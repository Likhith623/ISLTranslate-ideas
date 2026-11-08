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

    def forward(self, queries, memory, memory_mask):
        q = self.query_norm(queries)
        m = self.memory_norm(memory)
        attended, weights = self.cross(q, m, m, key_padding_mask=~memory_mask.bool(),
                                       need_weights=True, average_attn_weights=True)
        queries = queries + attended
        queries = self.self_layer(queries)
        queries = queries + self.ffn(self.ffn_norm(queries))
        return queries, weights


class LatentPlanGrounder(nn.Module):
    """Build a small ordered latent plan, then bind each plan token to video frames.

    The outputs are appended to the frame memory passed into the pretrained text
    encoder. ``grounding`` is an attention diagnostic, not a verified human gloss.
    """

    def __init__(self, dim: int, num_queries: int = 16, layers: int = 2, heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(1, num_queries, dim) * 0.02)
        self.blocks = nn.ModuleList([PlanGroundingBlock(dim, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, memory: torch.Tensor, memory_mask: torch.Tensor):
        b = memory.shape[0]
        queries = self.queries.expand(b, -1, -1)
        attentions = []
        for block in self.blocks:
            queries, weights = block(queries, memory, memory_mask)
            attentions.append(weights)
        plan = self.norm(queries)
        # Average layers into [B,K,T] evidence attention for inspection.
        grounding = torch.stack(attentions).mean(0) if attentions else memory.new_zeros(
            (b, plan.shape[1], memory.shape[1]))
        grounding = grounding * memory_mask[:, None].to(grounding.dtype)
        return plan, grounding
