"""Dual-rate temporal modelling for RA-MAGC.

The stride-one front-end creates a retained fine memory. A learned pairwise
aggregator creates the shorter global sequence. The Conformer models global
context, and a genuinely local gather-based cross-attention retrieves fine detail.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import MultiScaleTemporalConfig, ConformerConfig


def length_mask(lengths: torch.Tensor, maximum: int, device: torch.device | None = None) -> torch.Tensor:
    """Return ``[B,T]`` boolean mask; True marks valid sequence positions."""
    if lengths.ndim != 1:
        raise ValueError("lengths must be one-dimensional [B]")
    if maximum < 0:
        raise ValueError("maximum must be non-negative")
    target_device = lengths.device if device is None else device
    return torch.arange(maximum, device=target_device).unsqueeze(0) < lengths.to(target_device).unsqueeze(1)


class MultiScaleTemporalBlock(nn.Module):
    """Stride-one residual temporal block with four configurable depthwise branches."""

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        branch_dim: int = 64,
        dropout: float = 0.1,
        kernel_sizes: tuple[int, int, int, int] = (3, 5, 5, 9),
        dilations: tuple[int, int, int, int] = (1, 1, 2, 2),
    ):
        super().__init__()
        if in_channels != out_channels:
            raise ValueError("Residual temporal block requires in_channels == out_channels")
        if branch_dim * 4 != out_channels:
            raise ValueError("branch_dim * 4 must equal out_channels")
        if len(kernel_sizes) != 4 or len(dilations) != 4:
            raise ValueError("Exactly four temporal branches are required")
        if any(k <= 0 or k % 2 == 0 for k in kernel_sizes) or any(d <= 0 for d in dilations):
            raise ValueError("Kernels must be positive odd values and dilations positive")

        branches = []
        for kernel, dilation in zip(kernel_sizes, dilations):
            padding = dilation * (kernel - 1) // 2
            branches.append(
                nn.ModuleDict(
                    {
                        "depthwise": nn.Conv1d(
                            in_channels,
                            in_channels,
                            kernel_size=kernel,
                            padding=padding,
                            dilation=dilation,
                            groups=in_channels,
                        ),
                        "pointwise": nn.Linear(in_channels, branch_dim),
                        "norm": nn.LayerNorm(branch_dim),
                    }
                )
            )
        self.branches = nn.ModuleList(branches)
        self.out_proj = nn.Linear(out_channels, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.in_channels:
            raise ValueError(f"x must have shape [B,T,{self.in_channels}]")
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        mask = mask.bool()
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        residual = x
        x_conv = x.transpose(1, 2)
        outputs = []
        for branch in self.branches:
            z = branch["depthwise"](x_conv).transpose(1, 2)
            z = branch["pointwise"](z)
            z = F.gelu(branch["norm"](z))
            outputs.append(z)
        merged = torch.cat(outputs, dim=-1)
        merged = torch.where(mask[..., None], merged, torch.zeros_like(merged))
        update = self.dropout(self.out_proj(merged))
        out = self.norm(residual + update)
        return torch.where(mask[..., None], out, torch.zeros_like(out))


class PairwiseAggregator(nn.Module):
    """Aggregate adjacent fine tokens into a ceil(T/2) global sequence.

    Each pair is represented by its mean and signed difference. The temporal
    metadata uses pair duration and centre time *relative to the clip start*,
    avoiding dependence on arbitrary absolute recording-clock offsets.
    """

    def __init__(self, in_channels: int = 256, out_channels: int = 256):
        super().__init__()
        if in_channels <= 0 or out_channels <= 0:
            raise ValueError("Aggregator channel dimensions must be positive")
        self.proj = nn.Sequential(
            nn.Linear(in_channels * 2, out_channels),
            nn.LayerNorm(out_channels),
            nn.GELU(),
        )
        self.time_proj = nn.Linear(2, out_channels)
        self.pattern_embed = nn.Embedding(2, out_channels)

    def forward(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        timestamps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 3:
            raise ValueError("features must have shape [B,T,C]")
        b, t, _ = features.shape
        if t < 1:
            raise ValueError("Cannot aggregate an empty temporal sequence")
        if lengths.shape != (b,) or torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError("lengths must be [B] with values in [1,T]")
        if timestamps.shape != (b, t):
            raise ValueError("timestamps must have shape [B,T]")

        valid = length_mask(lengths, t, features.device)
        valid_times = timestamps.to(device=features.device, dtype=torch.float64)
        if not torch.isfinite(valid_times[valid]).all():
            raise ValueError("Valid timestamps must be finite")
        if t > 1:
            pair_is_valid = torch.arange(t - 1, device=features.device)[None, :] < (lengths - 1).to(features.device)[:, None]
            dt = valid_times[:, 1:] - valid_times[:, :-1]
            if torch.any(pair_is_valid & (dt <= 0)):
                raise ValueError("Valid timestamps must be strictly increasing")

        features = torch.where(valid[..., None], features, torch.zeros_like(features))
        tc = (t + 1) // 2
        if t % 2:
            features = F.pad(features, (0, 0, 0, 1), value=0.0)
            valid_times = F.pad(valid_times, (0, 1), value=0.0)

        a, b_frame = features[:, 0::2], features[:, 1::2]
        ta, tb = valid_times[:, 0::2], valid_times[:, 1::2]
        pair_idx = torch.arange(tc, device=features.device)[None, :]
        has_second = (2 * pair_idx + 1) < lengths.to(features.device)[:, None]

        mean = torch.where(has_second[..., None], (a + b_frame) * 0.5, a)
        difference = torch.where(has_second[..., None], b_frame - a, torch.zeros_like(a))
        duration = torch.where(has_second, (tb - ta).clamp_min(0.0), torch.zeros_like(ta))
        centre_abs = torch.where(has_second, (ta + tb) * 0.5, ta)
        start_time = valid_times[:, :1]
        centre_elapsed = centre_abs - start_time
        time_features = torch.stack([duration, centre_elapsed], dim=-1).to(dtype=features.dtype)
        centre_elapsed = centre_elapsed.to(dtype=features.dtype)
        pattern = torch.where(has_second, 0, 1).long()

        coarse = self.proj(torch.cat([mean, difference], dim=-1))
        coarse = coarse + self.time_proj(time_features) + self.pattern_embed(pattern)
        coarse_lengths = (lengths + 1) // 2
        coarse_mask = length_mask(coarse_lengths, tc, features.device)
        coarse = torch.where(coarse_mask[..., None], coarse, torch.zeros_like(coarse))
        coarse_elapsed = torch.where(coarse_mask, centre_elapsed, torch.zeros_like(centre_elapsed))
        return coarse, coarse_lengths, coarse_elapsed


class RelativeElapsedTimeBias(nn.Module):
    """Signed log-bucket relative-time bias for temporal self-attention."""

    def __init__(self, num_heads: int, num_buckets: int = 64, max_seconds: float = 20.0):
        super().__init__()
        if num_heads <= 0 or num_buckets < 4 or num_buckets % 2:
            raise ValueError("num_heads must be positive and num_buckets even and >=4")
        if max_seconds <= 1e-4:
            raise ValueError("max_seconds must be > 1e-4")
        self.num_heads = num_heads
        self.num_buckets = num_buckets
        self.max_seconds = max_seconds
        self.bias_table = nn.Embedding(num_buckets, num_heads)

    def _time_to_bucket(self, delta_t: torch.Tensor) -> torch.Tensor:
        half = self.num_buckets // 2
        sign_offset = (delta_t >= 0).long() * half
        magnitude = delta_t.abs().clamp(min=1e-6, max=self.max_seconds)
        scaled = torch.log(magnitude / 1e-4) / math.log(self.max_seconds / 1e-4)
        bucket = torch.floor(scaled * (half - 1)).long().clamp(0, half - 1)
        return bucket + sign_offset

    def forward(self, timestamps: torch.Tensor) -> torch.Tensor:
        if timestamps.ndim != 2:
            raise ValueError("timestamps must have shape [B,T]")
        delta = timestamps.unsqueeze(2) - timestamps.unsqueeze(1)  # query time minus key time
        indices = self._time_to_bucket(delta)
        return self.bias_table(indices).permute(0, 3, 1, 2).contiguous()


class ConformerBlock(nn.Module):
    """Macaron-style Conformer with explicit padding-safe depthwise convolution."""

    def __init__(self, config: ConformerConfig):
        super().__init__()
        d = config.d_model
        if d % config.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if config.depthwise_kernel_size % 2 == 0:
            raise ValueError("depthwise_kernel_size must be odd to preserve sequence length")
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads

        def ffn() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(d, config.dim_feedforward),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.dim_feedforward, d),
                nn.Dropout(config.dropout),
            )

        self.ffn1 = ffn()
        self.ffn1_norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.attn_out = nn.Linear(d, d)
        self.attn_norm = nn.LayerNorm(d)
        self.attn_dropout = nn.Dropout(config.dropout)

        self.conv_in_norm = nn.LayerNorm(d)
        self.conv_pw1 = nn.Linear(d, 2 * d)
        self.conv_dw = nn.Conv1d(
            d,
            d,
            kernel_size=config.depthwise_kernel_size,
            padding=(config.depthwise_kernel_size - 1) // 2,
            groups=d,
        )
        self.conv_mid_norm = nn.LayerNorm(d)
        self.conv_pw2 = nn.Linear(d, d)
        self.conv_dropout = nn.Dropout(config.dropout)

        self.ffn2 = ffn()
        self.ffn2_norm = nn.LayerNorm(d)
        self.final_norm = nn.LayerNorm(d)
        self.d_model = d

    def forward(self, x: torch.Tensor, mask: torch.Tensor, time_bias: torch.Tensor | None = None) -> torch.Tensor:
        if x.ndim != 3 or x.shape[-1] != self.d_model:
            raise ValueError(f"x must have shape [B,T,{self.d_model}]")
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [B,T]")
        b, t, d = x.shape
        mask = mask.bool()
        x = torch.where(mask[..., None], x, torch.zeros_like(x))

        # Macaron FFN 1.
        ffn_in = torch.where(mask[..., None], self.ffn1_norm(x), torch.zeros_like(x))
        x = x + 0.5 * self.ffn1(ffn_in)
        x = torch.where(mask[..., None], x, torch.zeros_like(x))

        # Multi-head self-attention.
        normed = torch.where(mask[..., None], self.attn_norm(x), torch.zeros_like(x))
        q = self.q_proj(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(normed).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if time_bias is not None:
            if time_bias.shape != (b, self.num_heads, t, t):
                raise ValueError("time_bias must have shape [B,num_heads,T,T]")
            scores = scores + time_bias
        key_mask = mask[:, None, None, :]
        has_keys = mask.any(dim=-1)[:, None, None, None]
        scores = scores.masked_fill(~key_mask, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        weights = F.softmax(safe_scores, dim=-1)
        weights = torch.where(has_keys, weights, torch.zeros_like(weights))
        weights = self.attn_dropout(weights)
        context = torch.matmul(weights, v).transpose(1, 2).reshape(b, t, d)
        x = x + self.attn_out(context)
        x = torch.where(mask[..., None], x, torch.zeros_like(x))

        # Convolution module. Re-mask after every biased operation so padded
        # positions cannot affect valid boundary positions in the depthwise conv.
        conv_in = self.conv_in_norm(x)
        conv_in = torch.where(mask[..., None], conv_in, torch.zeros_like(conv_in))
        glu = F.glu(self.conv_pw1(conv_in), dim=-1)
        glu = torch.where(mask[..., None], glu, torch.zeros_like(glu))
        depthwise = self.conv_dw(glu.transpose(1, 2)).transpose(1, 2)
        depthwise = torch.where(mask[..., None], depthwise, torch.zeros_like(depthwise))
        conv_act = F.gelu(self.conv_mid_norm(depthwise))
        conv_act = torch.where(mask[..., None], conv_act, torch.zeros_like(conv_act))
        conv_out = self.conv_dropout(self.conv_pw2(conv_act))
        conv_out = torch.where(mask[..., None], conv_out, torch.zeros_like(conv_out))
        x = x + conv_out
        x = torch.where(mask[..., None], x, torch.zeros_like(x))

        ffn_in = torch.where(mask[..., None], self.ffn2_norm(x), torch.zeros_like(x))
        x = x + 0.5 * self.ffn2(ffn_in)
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        return torch.where(mask[..., None], self.final_norm(x), torch.zeros_like(x))


class FineDetailRetrieval(nn.Module):
    """Gather-based local cross-attention from coarse tokens into fine memory.

    For coarse token i, the default six-frame window is indices 2i-2 through
    2i+3 (clipped to the available sequence). This computes O(T_coarse * W)
    attention rather than materializing a dense O(T_coarse * T_fine) matrix.
    """

    def __init__(
        self,
        d_model: int = 256,
        num_heads: int = 4,
        local_window_radius: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if local_window_radius < 0:
            raise ValueError("local_window_radius must be non-negative")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.radius = local_window_radius
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

    def forward(
        self,
        coarse_tokens: torch.Tensor,
        fine_memory: torch.Tensor,
        coarse_mask: torch.Tensor,
        fine_mask: torch.Tensor,
    ) -> torch.Tensor:
        if coarse_tokens.ndim != 3 or fine_memory.ndim != 3:
            raise ValueError("coarse_tokens and fine_memory must have shape [B,T,D]")
        b, tc, d = coarse_tokens.shape
        bf, tf, df = fine_memory.shape
        if b != bf or d != self.d_model or df != self.d_model or tf < 1:
            raise ValueError("Incompatible coarse/fine memory shapes")
        if coarse_mask.shape != (b, tc) or fine_mask.shape != (b, tf):
            raise ValueError("coarse_mask/fine_mask shape mismatch")

        q = self.q_proj(coarse_tokens).view(b, tc, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(fine_memory).view(b, tf, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(fine_memory).view(b, tf, self.num_heads, self.head_dim).transpose(1, 2)

        coarse_idx = torch.arange(tc, device=coarse_tokens.device) * 2
        offsets = torch.arange(-self.radius, self.radius + 2, device=coarse_tokens.device)
        raw_idx = coarse_idx[:, None] + offsets[None, :]
        in_range = (raw_idx >= 0) & (raw_idx < tf)
        safe_idx = raw_idx.clamp(0, tf - 1)
        window = safe_idx.shape[1]

        gather_idx = safe_idx[None, None, :, :, None].expand(b, self.num_heads, tc, window, self.head_dim)
        k_expanded = k.unsqueeze(2).expand(b, self.num_heads, tc, tf, self.head_dim)
        v_expanded = v.unsqueeze(2).expand(b, self.num_heads, tc, tf, self.head_dim)
        k_local = torch.gather(k_expanded, dim=3, index=gather_idx)
        v_local = torch.gather(v_expanded, dim=3, index=gather_idx)

        fine_valid = fine_mask.bool()[:, safe_idx]  # [B,Tc,W]
        local_valid = fine_valid & in_range[None]
        allowed = local_valid[:, None, :, :]
        has_keys = allowed.any(dim=-1, keepdim=True)
        scores = (q.unsqueeze(3) * k_local).sum(dim=-1) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        weights = F.softmax(safe_scores, dim=-1)
        weights = torch.where(has_keys, weights, torch.zeros_like(weights))
        weights = self.dropout(weights)
        retrieved = (weights[..., None] * v_local).sum(dim=3).transpose(1, 2).reshape(b, tc, d)
        retrieved = self.out_proj(retrieved)

        gate = self.gate(torch.cat([coarse_tokens, retrieved], dim=-1))
        refined = self.norm(coarse_tokens + gate * retrieved)
        return torch.where(coarse_mask.bool()[..., None], refined, torch.zeros_like(refined))
