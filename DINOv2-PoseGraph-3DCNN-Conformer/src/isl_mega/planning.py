"""Latent semantic planning with explicit frame-evidence retrieval."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class PlanGroundingBlock(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(dim)
        self.memory_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.self_layer = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout=dropout,
                                                      activation="gelu", batch_first=True, norm_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, queries, memory, memory_mask, query_mask: torch.Tensor | None = None):
        q = self.query_norm(queries)
        m = self.memory_norm(memory)
        key_valid = memory_mask.bool()
        any_valid = key_valid.any(dim=1, keepdim=True)
        # MultiheadAttention returns NaNs on rows whose keys are all masked.
        safe_key_valid = key_valid.clone()
        if not any_valid.all():
            safe_key_valid[~any_valid.squeeze(1), 0] = True
        attended, weights = self.cross(q, m, m, key_padding_mask=~safe_key_valid,
                                       need_weights=True, average_attn_weights=True)
        attended = torch.where(any_valid[..., None], attended, torch.zeros_like(attended))
        weights = torch.where(any_valid[..., None], weights, torch.zeros_like(weights))
        queries = queries + attended
        src_pad = (~query_mask.bool()) if query_mask is not None else None
        queries = self.self_layer(queries, src_key_padding_mask=src_pad)
        queries = queries + self.ffn(self.ffn_norm(queries))
        if query_mask is not None:
            queries = torch.where(query_mask.bool()[..., None], queries, torch.zeros_like(queries))
        return queries, weights


class LatentPlanGrounder(nn.Module):
    """Build an ordered, non-collapsed latent plan bound to video frame evidence.

    When ``dynamic_routing=True``, only a clip-length-appropriate budget of
    plan slots remain active (padded slots are zeroed and masked in T5). When
    ``diverse_ordered=True``, attention orthogonality and monotonic coverage
    diagnostics/regularizers are emitted in ``forward_with_aux``.
    """

    def __init__(
        self,
        dim: int,
        num_queries: int = 16,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.1,
        diverse_ordered: bool = False,
        dynamic_routing: bool = False,
        min_queries: int = 2,
        order_margin: float = 0.0,
        target_stride: int = 4,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.diverse_ordered = diverse_ordered
        self.dynamic_routing = dynamic_routing
        self.min_queries = min(num_queries, max(1, min_queries))
        self.order_margin = order_margin
        self.target_stride = max(1, target_stride)
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        if diverse_ordered:
            # Ordered slot positional bias distinguishes early vs late plan concepts cleanly.
            self.slot_pos = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        else:
            self.slot_pos = None
        if dynamic_routing:
            self.budget_gate = nn.Linear(dim, 1)
            nn.init.constant_(self.budget_gate.bias, 1.5)
        else:
            self.budget_gate = None
        self.blocks = nn.ModuleList([PlanGroundingBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def _compute_plan_mask(self, memory_mask: torch.Tensor) -> torch.Tensor:
        b = memory_mask.shape[0]
        k = self.num_queries
        if not self.dynamic_routing:
            return torch.ones((b, k), dtype=torch.bool, device=memory_mask.device)
        valid_len = memory_mask.sum(dim=1)
        active = torch.clamp(
            torch.div(valid_len + self.target_stride - 1, self.target_stride, rounding_mode="floor"),
            min=self.min_queries,
            max=k,
        )
        return torch.arange(k, device=memory_mask.device)[None] < active[:, None]

    def _grounding_losses(
        self,
        plan: torch.Tensor,
        grounding: torch.Tensor,
        plan_mask: torch.Tensor,
        memory_mask: torch.Tensor,
        timestamps: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, k, t = grounding.shape
        device = plan.device
        if k < 2:
            zero = plan.sum() * 0.0
            return zero, zero

        # 1. Attention + embedding diversity loss (prevents query collapse).
        g_norm = F.normalize(grounding.float(), dim=-1, eps=1e-6)
        p_norm = F.normalize(plan.float(), dim=-1, eps=1e-6)
        eye = torch.eye(k, dtype=torch.bool, device=device)[None]
        pair_ok = plan_mask[:, :, None] & plan_mask[:, None, :] & (~eye)
        denom = pair_ok.sum().clamp_min(1.0)
        attn_gram = torch.bmm(g_norm, g_norm.transpose(1, 2))
        emb_gram = torch.bmm(p_norm, p_norm.transpose(1, 2))
        diversity_loss = (
            (attn_gram.square() * pair_ok).sum() / denom
            + 0.25 * (emb_gram.square() * pair_ok).sum() / denom
        )

        # 2. Intrinsic clip-relative temporal ordering loss (padding invariant).
        if t <= 1:
            order_loss = plan.sum() * 0.0
        else:
            if timestamps is not None:
                valid_len = memory_mask.sum(dim=1).clamp_min(1)
                last_idx = (valid_len - 1)[:, None]
                t0 = timestamps[:, :1]
                t1 = timestamps.gather(1, last_idx)
                dur = (t1 - t0).clamp_min(1e-5)
                tau = ((timestamps - t0) / dur).clamp(0.0, 1.0).float()
            else:
                valid_span = (memory_mask.sum(dim=1) - 1).clamp_min(1).float()
                tau = (torch.arange(t, device=device, dtype=torch.float32)[None] / valid_span[:, None]).clamp(0.0, 1.0)
            tau = torch.where(memory_mask.bool(), tau, torch.zeros_like(tau))
            row_mass = grounding.float().sum(dim=-1).clamp_min(1e-6)
            centers = (grounding.float() * tau[:, None, :]).sum(dim=-1) / row_mass
            has_span = (memory_mask.sum(dim=1) >= 2)[:, None]
            step_ok = plan_mask[:, :-1] & plan_mask[:, 1:] & has_span
            violation = F.relu(centers[:, :-1] - centers[:, 1:] + self.order_margin)
            order_loss = (violation * step_ok).sum() / step_ok.sum().clamp_min(1.0)
        return diversity_loss.to(plan.dtype), order_loss.to(plan.dtype)

    def forward_with_aux(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ):
        b = memory.shape[0]
        plan_mask = self._compute_plan_mask(memory_mask)
        queries = self.queries.expand(b, -1, -1)
        if self.slot_pos is not None:
            queries = queries + self.slot_pos.expand(b, -1, -1)
        attentions = []
        query_mask_arg = plan_mask if self.dynamic_routing else None
        for block in self.blocks:
            queries, weights = block(queries, memory, memory_mask, query_mask=query_mask_arg)
            attentions.append(weights)
        plan = self.norm(queries)
        if self.budget_gate is not None:
            gate = torch.sigmoid(self.budget_gate(plan))
            fixed_min = torch.arange(self.num_queries, device=plan.device)[None] < self.min_queries
            scale = torch.where(fixed_min[..., None], torch.ones_like(gate), gate)
            plan = plan * scale
        plan = torch.where(plan_mask[..., None], plan, torch.zeros_like(plan))

        grounding = torch.stack(attentions).mean(0) if attentions else memory.new_zeros(
            (b, plan.shape[1], memory.shape[1]))
        grounding = grounding * memory_mask[:, None].to(grounding.dtype)
        grounding = grounding * plan_mask[:, :, None].to(grounding.dtype)

        diversity_loss, order_loss = self._grounding_losses(
            plan, grounding, plan_mask, memory_mask, timestamps
        )
        aux = {
            "plan_mask": plan_mask,
            "diversity_loss": diversity_loss,
            "order_loss": order_loss,
        }
        return plan, grounding, aux

    def forward(
        self,
        memory: torch.Tensor,
        memory_mask: torch.Tensor,
        timestamps: torch.Tensor | None = None,
    ):
        plan, grounding, _ = self.forward_with_aux(memory, memory_mask, timestamps=timestamps)
        return plan, grounding
