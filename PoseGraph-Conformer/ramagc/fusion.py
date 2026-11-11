"""Per-frame anatomical interaction and reliability-aware region fusion."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RegionFusionConfig


class RegionInteraction(nn.Module):
    """Self-attention over left hand, right hand, upper body and optional face."""

    def __init__(self, config: RegionFusionConfig, max_regions: int = 4):
        super().__init__()
        if max_regions not in (3, 4):
            raise ValueError("max_regions must be 3 or 4")
        self.config = config
        self.region_embed = nn.Embedding(max_regions, config.region_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.region_dim,
            nhead=config.num_heads,
            dim_feedforward=config.region_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer,
            num_layers=config.num_interaction_layers,
            enable_nested_tensor=False,
        )
        self.norm = nn.LayerNorm(config.region_dim)

    def forward(self, region_tokens: torch.Tensor, region_valid_mask: torch.Tensor) -> torch.Tensor:
        if region_tokens.ndim != 4:
            raise ValueError("region_tokens must have shape [B,T,K,D]")
        b, t, k, d = region_tokens.shape
        if d != self.config.region_dim or k > self.region_embed.num_embeddings:
            raise ValueError("Region token dimensions do not match fusion configuration")
        if region_valid_mask.shape != (b, t, k):
            raise ValueError("region_valid_mask must have shape [B,T,K]")

        position_ids = torch.arange(k, device=region_tokens.device)
        x = region_tokens + self.region_embed(position_ids)[None, None]
        x = x.reshape(b * t, k, d)
        valid = region_valid_mask.reshape(b * t, k).bool()
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_valid = torch.where(any_valid, valid, torch.ones_like(valid))
        out = self.transformer(x, src_key_padding_mask=~safe_valid)
        out = self.norm(out).reshape(b, t, k, d)
        return torch.where(region_valid_mask[..., None], out, torch.zeros_like(out))


class ReliabilityGate(nn.Module):
    """Quality-conditioned soft gating over available anatomical regions.

    Quality descriptors, in order, are mean available detector confidence,
    confidence-availability ratio, observed-point ratio, imputed-point ratio,
    position-valid ratio and velocity-valid ratio.
    """

    def __init__(self, config: RegionFusionConfig, max_regions: int = 4):
        super().__init__()
        if max_regions not in (3, 4):
            raise ValueError("max_regions must be 3 or 4")
        self.config = config
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.region_dim + config.quality_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.region_dim * 2, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.missing_observation_embed = nn.Parameter(torch.randn(1, 1, config.fused_dim) * 0.02)

    def forward(
        self,
        raw_region_tokens: torch.Tensor,
        interacted_region_tokens: torch.Tensor,
        quality_descriptors: torch.Tensor,
        region_valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if raw_region_tokens.ndim != 4:
            raise ValueError("raw_region_tokens must have shape [B,T,K,D]")
        b, t, k, d = raw_region_tokens.shape
        if d != self.config.region_dim:
            raise ValueError("Region dimension does not match reliability gate configuration")
        if interacted_region_tokens.shape != raw_region_tokens.shape:
            raise ValueError("interacted_region_tokens must match raw_region_tokens")
        if quality_descriptors.shape != (b, t, k, self.config.quality_feature_dim):
            raise ValueError("quality_descriptors shape mismatch")
        if region_valid_mask.shape != (b, t, k):
            raise ValueError("region_valid_mask shape mismatch")

        valid = region_valid_mask.bool()
        logits = self.gate_mlp(torch.cat([raw_region_tokens, quality_descriptors], dim=-1)).squeeze(-1)
        has_any = valid.any(dim=-1, keepdim=True)
        logits = logits.masked_fill(~valid, torch.finfo(logits.dtype).min)
        safe_logits = torch.where(has_any, logits, torch.zeros_like(logits))
        weights = F.softmax(safe_logits, dim=-1)
        weights = torch.where(has_any, weights, torch.zeros_like(weights))

        gated = (raw_region_tokens * weights[..., None]).sum(dim=2)
        counts = valid.sum(dim=-1, keepdim=True).clamp_min(1).to(raw_region_tokens.dtype)
        interacted_mean = (interacted_region_tokens * valid[..., None]).sum(dim=2) / counts
        fused = self.fusion_proj(torch.cat([gated, interacted_mean], dim=-1))
        missing = self.missing_observation_embed.expand(b, t, -1)
        return torch.where(has_any, fused, missing)
