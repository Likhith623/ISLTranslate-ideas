"""Cross-Region Interaction and Reliability-Aware Gated Fusion.

Models interactions between anatomical regions (two-hand coordination, hand-to-body,
hand-to-face) and applies reliability gating based on observation and validity statistics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RegionFusionConfig


class RegionInteraction(nn.Module):
    """
    Per-frame interaction Transformer across anatomical regions:
    [Left Hand, Right Hand, Upper Body, (Face)].
    """
    def __init__(self, config: RegionFusionConfig, max_regions: int = 4):
        super().__init__()
        self.config = config
        self.region_embed = nn.Embedding(max_regions, config.region_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.region_dim,
            nhead=config.num_heads,
            dim_feedforward=config.region_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_interaction_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(config.region_dim)

    def forward(self, region_tokens: torch.Tensor, region_valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            region_tokens: [B, T, K, region_dim] where K is number of regions (3 or 4)
            region_valid_mask: [B, T, K] boolean mask indicating if region has valid observations
        Returns:
            interacted_tokens: [B, T, K, region_dim]
        """
        B, T, K, D = region_tokens.shape
        BT = B * T
        
        region_ids = torch.arange(K, device=region_tokens.device)
        h = region_tokens + self.region_embed(region_ids)[None, None, :, :]
        
        h_flat = h.view(BT, K, D)
        padding_mask = ~region_valid_mask.view(BT, K)
        
        # If all regions are invalid in a frame, unmask first element to prevent NaNs in softmax
        all_invalid = padding_mask.all(dim=-1, keepdim=True)
        safe_padding_mask = torch.where(all_invalid, torch.zeros_like(padding_mask), padding_mask)
        
        out_flat = self.transformer(h_flat, src_key_padding_mask=safe_padding_mask)
        out = self.norm(out_flat).view(B, T, K, D)
        
        # Mask out completely invalid positions
        out = out.masked_fill(~region_valid_mask[..., None], 0.0)
        return out


class ReliabilityGate(nn.Module):
    """
    Predicts reliability weights for each anatomical region based on
    spatial features and quality descriptors (observed ratio, validity ratio).
    """
    def __init__(self, config: RegionFusionConfig, max_regions: int = 4):
        super().__init__()
        self.config = config
        self.gate_mlp = nn.Sequential(
            nn.Linear(config.region_dim + config.quality_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1)
        )
        
        # Fusion projection: gate-weighted region summary + interacted region mean -> fused_dim (256)
        self.fusion_proj = nn.Sequential(
            nn.Linear(config.region_dim * 2, config.fused_dim),
            nn.LayerNorm(config.fused_dim),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Explicit learned representation for real frames where all regions are occluded
        self.missing_observation_embed = nn.Parameter(torch.randn(1, 1, config.fused_dim) * 0.02)

    def forward(
        self, 
        raw_region_tokens: torch.Tensor, 
        interacted_region_tokens: torch.Tensor, 
        quality_descriptors: torch.Tensor, 
        region_valid_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            raw_region_tokens: [B, T, K, region_dim]
            interacted_region_tokens: [B, T, K, region_dim]
            quality_descriptors: [B, T, K, quality_feature_dim] (obs_ratio, imp_ratio, pos_val, vel_val)
            region_valid_mask: [B, T, K] boolean mask
        Returns:
            fused_features: [B, T, fused_dim] (256 dimensions)
        """
        B, T, K, D = raw_region_tokens.shape
        
        # Concatenate region representation with quality descriptors to compute reliability logits
        gate_input = torch.cat([raw_region_tokens, quality_descriptors], dim=-1)  # [B, T, K, D + Q]
        logits = self.gate_mlp(gate_input).squeeze(-1)  # [B, T, K]
        
        # Mask regions with no usable landmarks
        logits = logits.masked_fill(~region_valid_mask, -1e9)
        
        has_any_region = region_valid_mask.any(dim=-1, keepdim=True)  # [B, T, 1]
        safe_logits = torch.where(~has_any_region, torch.zeros_like(logits), logits)
        
        alpha = F.softmax(safe_logits, dim=-1)  # [B, T, K]
        alpha = torch.where(~has_any_region, torch.zeros_like(alpha), alpha)
        
        # Gate-weighted region summary
        gated_summary = (raw_region_tokens * alpha[..., None]).sum(dim=2)  # [B, T, D]
        
        # Masked mean of interacted tokens
        valid_counts = region_valid_mask.sum(dim=-1, keepdim=True).clamp_min(1)  # [B, T, 1]
        interacted_mean = (interacted_region_tokens * region_valid_mask[..., None]).sum(dim=2) / valid_counts  # [B, T, D]
        
        # Combine and project
        combined = torch.cat([gated_summary, interacted_mean], dim=-1)  # [B, T, 2*D]
        fused = self.fusion_proj(combined)  # [B, T, fused_dim]
        
        # Substitute explicit missing representation if all regions are missing in a real frame
        fused = torch.where(~has_any_region, self.missing_observation_embed.expand(B, T, -1), fused)
        return fused

