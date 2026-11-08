"""Dual-Rate Temporal Encoding: Multi-Scale Temporal CNN, Pairwise Aggregation,
Conformer Backbone, and Local Fine-Detail Retrieval.

Processes every frame at original resolution, aggregates adjacent frames for global
modelling, and retains the fine memory for local detail retrieval.
Matches every specification in ideas.md.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiScaleTemporalConfig, ConformerConfig


def length_mask(lengths: torch.Tensor, maximum: int, device=None) -> torch.Tensor:
    """Create boolean mask [Batch, Time] where True indicates valid time steps."""
    return torch.arange(maximum, device=device or lengths.device)[None] < lengths.to(device)[:, None]


class MultiScaleTemporalBlock(nn.Module):
    """
    1D Multi-Scale Temporal Convolutional Block at stride 1 (ideas.md Section 4).
    
    Contains 4 parallel depthwise-separable branches with distinct kernel sizes
    and dilation factors:
      - Fine: k=3, d=1 (RF 3)
      - Short: k=5, d=1 (RF 5)
      - Medium: k=5, d=2 (RF 9)
      - Broad-local: k=9, d=2 (RF 17)
    Each branch uses depthwise Conv1D over 256 channels, pointwise projection to 64 channels,
    LayerNorm, and GELU. Concatenates 4 branches (256 dims), applies pointwise projection,
    dropout, residual connection, LayerNorm, and re-applies length mask.
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 256, 
                 branch_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        assert branch_dim * 4 == out_channels, "branch_dim * 4 must equal out_channels"
        
        # Branch 1: Fine (k=3, d=1)
        self.b1_dw = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1, dilation=1, groups=in_channels)
        self.b1_pw = nn.Linear(in_channels, branch_dim)
        self.b1_norm = nn.LayerNorm(branch_dim)
        
        # Branch 2: Short (k=5, d=1)
        self.b2_dw = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=2, dilation=1, groups=in_channels)
        self.b2_pw = nn.Linear(in_channels, branch_dim)
        self.b2_norm = nn.LayerNorm(branch_dim)
        
        # Branch 3: Medium (k=5, d=2)
        self.b3_dw = nn.Conv1d(in_channels, in_channels, kernel_size=5, padding=4, dilation=2, groups=in_channels)
        self.b3_pw = nn.Linear(in_channels, branch_dim)
        self.b3_norm = nn.LayerNorm(branch_dim)
        
        # Branch 4: Broad-local (k=9, d=2)
        self.b4_dw = nn.Conv1d(in_channels, in_channels, kernel_size=9, padding=8, dilation=2, groups=in_channels)
        self.b4_pw = nn.Linear(in_channels, branch_dim)
        self.b4_norm = nn.LayerNorm(branch_dim)
        
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, C] (256 dims)
            mask: [B, T] boolean mask
        Returns:
            out: [B, T, C]
        """
        res = x
        x_conv = x.transpose(1, 2)  # [B, C, T]
        
        # Each branch: depthwise Conv1D -> pointwise Linear -> LayerNorm -> GELU
        out1 = F.gelu(self.b1_norm(self.b1_pw(self.b1_dw(x_conv).transpose(1, 2))))
        out2 = F.gelu(self.b2_norm(self.b2_pw(self.b2_dw(x_conv).transpose(1, 2))))
        out3 = F.gelu(self.b3_norm(self.b3_pw(self.b3_dw(x_conv).transpose(1, 2))))
        out4 = F.gelu(self.b4_norm(self.b4_pw(self.b4_dw(x_conv).transpose(1, 2))))
        
        concat = torch.cat([out1, out2, out3, out4], dim=-1)  # [B, T, 256]
        projected = self.dropout(self.out_proj(concat))
        
        out = self.norm(res + projected)
        # Crucial guardrail: zero-out padded positions immediately to prevent LayerNorm bias leak
        out = torch.where(mask[..., None], out, torch.zeros_like(out))
        return out


class PairwiseAggregator(nn.Module):
    """
    Pairs adjacent valid frames into a compact coarse sequence of length ceil(T/2).
    Follows ideas.md Section 5.2 exactly:
      m_i = (a_i + b_i) / 2
      d_i = b_i - a_i
      For an odd-length segment, the final group contains one real frame:
      mean is that frame, difference is zero.
      Adds learned embedding of center timestamp, pair duration, and observation validity pattern.
    """
    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        # Concatenate [mean, difference] = 512 dims -> project to 256
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU()
        )
        self.time_proj = nn.Linear(2, out_channels)  # (pair_duration, center_timestamp)
        # Validity pattern embedding: 0 = regular pair (both frames present), 1 = single odd frame
        self.pattern_embed = nn.Embedding(2, out_channels)

    def forward(
        self, 
        features: torch.Tensor, 
        lengths: torch.Tensor, 
        timestamps: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            features: [B, T, C]
            lengths: [B]
            timestamps: [B, T] in seconds
        Returns:
            coarse_features: [B, T_coarse, C]
            coarse_lengths: [B]
            coarse_timestamps: [B, T_coarse]
        """
        B, T, C = features.shape
        T_coarse = (T + 1) // 2
        
        # Identify whether each pair position has both frames or is a single odd frame
        is_odd_length = (T % 2 != 0)
        
        if is_odd_length:
            features_padded = F.pad(features, (0, 0, 0, 1), value=0.0)
            timestamps_padded = F.pad(timestamps, (0, 1), value=0.0)
        else:
            features_padded = features
            timestamps_padded = timestamps
            
        a = features_padded[:, 0::2]  # [B, T_coarse, C]
        b = features_padded[:, 1::2]  # [B, T_coarse, C]
        
        t_a = timestamps_padded[:, 0::2]
        t_b = timestamps_padded[:, 1::2]
        
        # Check per-batch-item if the last token in coarse sequence is an odd single frame
        pair_idx = torch.arange(T_coarse, device=features.device)[None, :]  # [1, T_coarse]
        first_frame_idx = pair_idx * 2
        second_frame_idx = first_frame_idx + 1
        
        has_second_frame = second_frame_idx < lengths[:, None]  # [B, T_coarse]
        
        # For odd final frame: mean is a, difference is 0 (ideas.md Section 5.2)
        m = torch.where(has_second_frame[..., None], (a + b) / 2.0, a)
        d = torch.where(has_second_frame[..., None], b - a, torch.zeros_like(b))
        
        pair_duration = torch.where(has_second_frame, (t_b - t_a).clamp_min(0.0), torch.zeros_like(t_a))
        center_time = torch.where(has_second_frame, (t_a + t_b) / 2.0, t_a)
        time_feats = torch.stack([pair_duration, center_time], dim=-1)  # [B, T_coarse, 2]
        
        # Pattern index: 0 = regular pair, 1 = single odd frame
        pattern_idx = torch.where(has_second_frame, 0, 1)  # [B, T_coarse]
        pattern_feats = self.pattern_embed(pattern_idx)     # [B, T_coarse, C]
        
        combined = torch.cat([m, d], dim=-1)  # [B, T_coarse, 2*C]
        coarse = self.proj(combined) + self.time_proj(time_feats) + pattern_feats
        
        coarse_lengths = (lengths + 1) // 2
        coarse_mask = length_mask(coarse_lengths, T_coarse, features.device)
        coarse = torch.where(coarse_mask[..., None], coarse, torch.zeros_like(coarse))
        
        return coarse, coarse_lengths, center_time


class RelativeElapsedTimeBias(nn.Module):
    """
    Learned relative elapsed-time attention bias based on physical timestamp
    differences (delta_t in seconds), distinguishing nearby and distant events.
    Matches ideas.md Section 6.
    """
    def __init__(self, num_heads: int, num_buckets: int = 64, max_seconds: float = 20.0):
        super().__init__()
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_seconds = max_seconds
        self.bias_table = nn.Embedding(num_buckets, num_heads)

    def _time_to_bucket(self, delta_t: torch.Tensor) -> torch.Tensor:
        # delta_t is [B, T, T] in seconds (can be negative or positive)
        half_buckets = self.num_buckets // 2
        sign = (delta_t >= 0).long()
        abs_t = delta_t.abs().clamp(min=1e-6, max=self.max_seconds)
        
        # Log-spaced bucket assignment
        log_scaled = torch.log(abs_t / 1e-4) / math.log(self.max_seconds / 1e-4)
        bucket = (log_scaled * (half_buckets - 1)).long().clamp(0, half_buckets - 1)
        
        # Distinct buckets for forward and backward relative time
        return bucket + sign * half_buckets

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestamps: [B, T]
        Returns:
            bias: [B, num_heads, T, T]
        """
        delta_t = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # [B, T, T]
        buckets = self._time_to_bucket(delta_t)  # [B, T, T]
        bias = self.bias_table(buckets)  # [B, T, T, num_heads]
        return bias.permute(0, 3, 1, 2)  # [B, num_heads, T, T]


class ConformerBlock(nn.Module):
    """
    Standard Macaron-style Conformer Block (ideas.md Section 6):
      1. Half-step Feed-Forward module
      2. Multi-Head Self-Attention with relative elapsed-time bias
      3. Depthwise Temporal Convolution module:
         pointwise expansion -> GLU -> depthwise Conv1D -> LayerNorm -> SiLU/GELU -> pointwise projection
      4. Second half-step Feed-Forward module
      5. Final LayerNorm
    """
    def __init__(self, config: ConformerConfig):
        super().__init__()
        d = config.d_model
        
        # 1. Macaron FFN 1
        self.ffn1 = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d),
            nn.Dropout(config.dropout)
        )
        self.ffn1_norm = nn.LayerNorm(d)
        
        # 2. Multi-Head Self-Attention
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.attn_norm = nn.LayerNorm(d)
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # 3. Depthwise Temporal Convolution Module (with explicit normalization)
        padding = (config.depthwise_kernel_size - 1) // 2
        self.conv_in_norm = nn.LayerNorm(d)
        self.conv_pw1 = nn.Linear(d, d * 2)  # For GLU activation
        self.conv_dw = nn.Conv1d(d, d, kernel_size=config.depthwise_kernel_size, padding=padding, groups=d)
        self.conv_mid_norm = nn.LayerNorm(d)
        self.conv_pw2 = nn.Linear(d, d)
        self.conv_dropout = nn.Dropout(config.dropout)
        
        # 4. Macaron FFN 2
        self.ffn2 = nn.Sequential(
            nn.Linear(d, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d),
            nn.Dropout(config.dropout)
        )
        self.ffn2_norm = nn.LayerNorm(d)
        
        # 5. Final LayerNorm
        self.final_norm = nn.LayerNorm(d)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_bias: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T, d_model]
            mask: [B, T] boolean mask
            time_bias: [B, num_heads, T, T] relative elapsed-time attention bias
        """
        B, T, D = x.shape
        
        # 1. Macaron FFN 1 (half step: 0.5 * ffn)
        x = x + 0.5 * self.ffn1(self.ffn1_norm(x)) * mask[..., None]
        
        # 2. MHSA with Relative Time Bias
        normed = self.attn_norm(x)
        q = self.q_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if time_bias is not None:
            scores = scores + time_bias[:, :, :T, :T]
            
        key_padding_mask = ~mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
        has_valid_keys = mask.unsqueeze(1).unsqueeze(2).any(dim=-1, keepdim=True)
        scores = scores.masked_fill(key_padding_mask, -1e9)
        safe_scores = torch.where(~has_valid_keys, torch.zeros_like(scores), scores)
        attn_weights = F.softmax(safe_scores, dim=-1)
        attn_weights = torch.where(~has_valid_keys, torch.zeros_like(attn_weights), attn_weights)
        attn_weights = self.attn_dropout(attn_weights)
        
        attn_ctx = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, T, D)
        x = x + self.attn_out(attn_ctx) * mask[..., None]
        
        # 3. Depthwise Temporal Convolution Module
        c_in = self.conv_in_norm(x)
        c_glu = F.glu(self.conv_pw1(c_in), dim=-1)  # [B, T, D]
        c_dw = self.conv_dw(c_glu.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        c_act = F.gelu(self.conv_mid_norm(c_dw))
        c_out = self.conv_dropout(self.conv_pw2(c_act))
        x = x + c_out * mask[..., None]
        
        # 4. Macaron FFN 2 (half step: 0.5 * ffn)
        x = x + 0.5 * self.ffn2(self.ffn2_norm(x)) * mask[..., None]
        
        # 5. Final LayerNorm (guaranteed zero bias leak on padded frames)
        out = self.final_norm(x)
        return torch.where(mask[..., None], out, torch.zeros_like(out))


class FineDetailRetrieval(nn.Module):
    """
    Local cross-attention from coarse Conformer tokens to fine memory Z (ideas.md Section 5.3).
    
    Each coarse token queries a local window of the fine memory (the two frames in its pair
    and immediately adjacent frames). Enforces local window band masking.
    """
    def __init__(self, d_model: int = 256, num_heads: int = 4, local_window_radius: int = 2, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.local_window_radius = local_window_radius
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Sigmoid())

    def forward(
        self, 
        coarse_tokens: torch.Tensor, 
        fine_memory: torch.Tensor, 
        coarse_mask: torch.Tensor, 
        fine_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            coarse_tokens: [B, Tc, d_model]
            fine_memory: [B, Tf, d_model]
            coarse_mask: [B, Tc]
            fine_mask: [B, Tf]
        """
        B, Tc, D = coarse_tokens.shape
        _, Tf, _ = fine_memory.shape
        
        q = self.q_proj(coarse_tokens).view(B, Tc, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(fine_memory).view(B, Tf, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(fine_memory).view(B, Tf, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Raw cross-attention scores: [B, H, Tc, Tf]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Local window mask: each coarse token i corresponds to fine frames around 2*i + 0.5
        coarse_indices = torch.arange(Tc, device=coarse_tokens.device)[:, None]  # [Tc, 1]
        fine_indices = torch.arange(Tf, device=fine_memory.device)[None, :]      # [1, Tf]
        center_fine_pos = coarse_indices * 2.0 + 0.5
        
        # Frame distance in fine time steps
        dist = (fine_indices - center_fine_pos).abs()  # [Tc, Tf]
        in_local_window = dist <= (self.local_window_radius + 0.5)  # [Tc, Tf]
        
        # Combine local window mask with key padding mask from fine_mask
        valid_key_mask = fine_mask.unsqueeze(1).unsqueeze(2) & in_local_window.unsqueeze(0).unsqueeze(0)  # [B, 1, Tc, Tf]
        has_valid_keys = valid_key_mask.any(dim=-1, keepdim=True)  # [B, 1, Tc, 1]
        scores = scores.masked_fill(~valid_key_mask, -1e9)
        safe_scores = torch.where(~has_valid_keys, torch.zeros_like(scores), scores)
        attn_weights = F.softmax(safe_scores, dim=-1)
        attn_weights = torch.where(~has_valid_keys, torch.zeros_like(attn_weights), attn_weights)
        attn_weights = self.dropout(attn_weights)
        
        retrieved = torch.matmul(attn_weights, v).transpose(1, 2).reshape(B, Tc, D)
        retrieved = self.out_proj(retrieved)
        
        # Gated residual update: coarse + gate * retrieved
        gate_val = self.gate(torch.cat([coarse_tokens, retrieved], dim=-1))
        refined = self.norm(coarse_tokens + gate_val * retrieved)
        return torch.where(coarse_mask[..., None], refined, torch.zeros_like(refined))
