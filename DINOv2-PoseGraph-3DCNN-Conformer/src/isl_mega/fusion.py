"""Quality-aware cross-modal interaction and gated stream fusion."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import FusionConfig


STREAM_NAMES = (
    "left_hand_rgb", "right_hand_rgb", "face_rgb", "body_rgb",
    "left_hand_pose", "right_hand_pose", "body_pose", "face_pose",
    "motion", "fingerspelling",
)


class ContinuousTimeLagBias(nn.Module):
    """Continuous elapsed-time Fourier/polynomial lag bias across asynchronous streams."""

    def __init__(self, num_heads: int, num_streams: int, window_seconds: float):
        super().__init__()
        self.num_heads = num_heads
        self.num_streams = num_streams
        self.window_seconds = max(float(window_seconds), 1e-4)
        self.dt_mlp = nn.Sequential(
            nn.Linear(7, num_heads * 2),
            nn.GELU(),
            nn.Linear(num_heads * 2, num_heads),
        )
        # Signed stream-specific anticipation/delay slope and stream bias.
        self.stream_slope = nn.Parameter(torch.zeros(num_heads, num_streams))
        self.stream_bias = nn.Parameter(torch.zeros(num_heads, num_streams))

    def forward(self, dt: torch.Tensor) -> torch.Tensor:
        # dt: [B, T, K] -> bias: [B, T, H, K * S]
        tau = (dt / self.window_seconds).clamp(-2.0, 2.0)
        pi_tau = math.pi * tau
        feats = torch.stack([
            tau,
            tau * tau,
            tau.abs(),
            torch.sin(pi_tau),
            torch.cos(pi_tau),
            torch.sin(2.0 * pi_tau),
            torch.cos(2.0 * pi_tau),
        ], dim=-1)
        dt_bias = self.dt_mlp(feats).permute(0, 1, 3, 2).unsqueeze(-1)  # [B,T,H,K,1]
        slope = self.stream_slope[None, None, :, None, :] * tau[:, :, None, :, None]
        bias = dt_bias + slope + self.stream_bias[None, None, :, None, :]
        return bias.reshape(dt.shape[0], dt.shape[1], self.num_heads, dt.shape[2] * self.num_streams)


class ReliabilityAwareFusion(nn.Module):
    """Per-frame cross-stream attention followed by quality-conditioned gating.

    Supports both discrete frame-offset lag attention (v1 baseline) and
    continuous-time (elapsed seconds) asynchronous alignment paired with
    calibrated structural reliability gating (v2).
    """

    def __init__(self, input_dims: dict[str, int], config: FusionConfig):
        super().__init__()
        if set(input_dims) != set(STREAM_NAMES):
            raise ValueError(f"input_dims must contain exactly {STREAM_NAMES}")
        self.names = STREAM_NAMES
        self.projections = nn.ModuleDict({k: nn.Sequential(nn.Linear(v, config.dim), nn.LayerNorm(config.dim))
                                          for k, v in input_dims.items()})
        self.type_embedding = nn.Parameter(torch.randn(1, 1, len(self.names), config.dim) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model=config.dim, nhead=config.num_heads,
                                           dim_feedforward=config.dim * 4, dropout=config.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.interaction = nn.TransformerEncoder(layer, num_layers=config.layers, enable_nested_tensor=False)
        self.quality_gate = nn.Sequential(nn.Linear(config.dim + config.quality_dim, config.dim // 2),
                                          nn.GELU(), nn.Linear(config.dim // 2, 1))
        if config.reliability_uncertainty_gate:
            self.reliability_net = nn.Sequential(
                nn.Linear(config.quality_dim, max(config.quality_dim * 2, 8)),
                nn.GELU(),
                nn.Linear(max(config.quality_dim * 2, 8), 1),
            )
        else:
            self.reliability_net = None
        self.lag_heads = config.num_heads
        self.lag_dh = config.dim // config.num_heads
        self.lag_q = nn.Linear(config.dim, config.dim)
        self.lag_k = nn.Linear(config.dim, config.dim)
        self.lag_v = nn.Linear(config.dim, config.dim)
        self.lag_out = nn.Linear(config.dim, config.dim)
        if config.continuous_time_lag:
            self.continuous_lag_bias = ContinuousTimeLagBias(
                config.num_heads, len(self.names), config.lag_window_seconds
            )
            self.lag_bias = None
        else:
            self.continuous_lag_bias = None
            self.lag_bias = nn.Parameter(torch.zeros(config.num_heads, 5))  # offsets -2..+2 frames
        self.output = nn.Sequential(nn.Linear(config.dim * 2, config.dim), nn.LayerNorm(config.dim), nn.GELU())
        self.null = nn.Parameter(torch.zeros(1, 1, config.dim))
        self.config = config

    def forward_with_aux(
        self,
        streams: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        quality: dict[str, torch.Tensor],
        timestamps: torch.Tensor | None = None,
    ):
        if set(streams) != set(self.names) or set(available) != set(self.names) or set(quality) != set(self.names):
            raise ValueError("streams, available and quality must contain exactly the documented 10 streams")
        values, masks, qualities = [], [], []
        ref = streams[self.names[0]]
        b, t = ref.shape[:2]
        if timestamps is not None and timestamps.shape != (b, t):
            raise ValueError("timestamps must have shape [B,T]")
        for name in self.names:
            x = streams[name]
            if x.shape[:2] != (b, t):
                raise ValueError(f"Stream {name} has inconsistent batch/time dimensions")
            if available[name].shape != (b, t) or quality[name].shape != (b, t, self.config.quality_dim):
                raise ValueError(f"Availability/quality shape mismatch for {name}")
            values.append(self.projections[name](x))
            masks.append(available[name].bool())
            qualities.append(quality[name])
        projected = torch.stack(values, dim=2)
        token = projected + self.type_embedding
        mask = torch.stack(masks, dim=2)
        q = torch.stack(qualities, dim=2)
        # TransformerEncoder cannot receive an all-masked row. Inject a null token
        # only for such rows, then zero its weight in the final output.
        flat = token.reshape(b * t, len(self.names), -1)
        flat_mask = mask.reshape(b * t, len(self.names))
        all_missing = ~flat_mask.any(-1)
        flat_mask_safe = flat_mask.clone()
        flat = flat.clone()
        if all_missing.any():
            flat[all_missing, 0] = self.null[0, 0]
            flat_mask_safe[all_missing, 0] = True
        interacted = self.interaction(flat, src_key_padding_mask=~flat_mask_safe).reshape(b, t, len(self.names), -1)
        gate_in = torch.cat([interacted, q], dim=-1)
        logits = self.quality_gate(gate_in).squeeze(-1)
        if self.reliability_net is not None:
            reliability = torch.sigmoid(self.reliability_net(q).squeeze(-1))
            reliability = torch.where(mask, reliability, torch.zeros_like(reliability))
            logits = logits + torch.log(reliability.clamp_min(1e-4))
        else:
            reliability = mask.to(logits.dtype)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        logits = torch.where(mask.any(-1, keepdim=True), logits, torch.zeros_like(logits))
        weights = torch.softmax(logits, dim=-1)
        weights = torch.where(mask, weights, torch.zeros_like(weights))
        pooled = (interacted * weights[..., None]).sum(dim=2)
        pooled = torch.where(mask.any(-1, keepdim=True), pooled, self.null.expand(b, t, -1))

        if self.continuous_lag_bias is not None:
            r = self.config.lag_radius_frames
            offsets = torch.arange(-r, r + 1, device=pooled.device)
            time_index = torch.arange(t, device=pooled.device)[:, None] + offsets[None, :]
            in_range = (time_index >= 0) & (time_index < t)
            safe_index = time_index.clamp(0, t - 1)
            times = (
                timestamps if timestamps is not None
                else torch.arange(t, device=pooled.device, dtype=pooled.dtype)[None].expand(b, t) / 25.0
            )
            dt = times[:, safe_index] - times[:, :, None]  # [B,T,K]
            time_ok = (dt.abs() <= self.config.lag_window_seconds) | (offsets[None, None, :] == 0)
            neighbours = interacted[:, safe_index]
            neighbour_mask = mask[:, safe_index] & in_range[None, :, :, None] & time_ok[..., None]
            k_off = len(offsets)
            bias_term = self.continuous_lag_bias(dt)
        else:
            offsets = torch.arange(-2, 3, device=pooled.device)
            time_index = torch.arange(t, device=pooled.device)[:, None] + offsets[None, :]
            in_range = (time_index >= 0) & (time_index < t)
            safe_index = time_index.clamp(0, t - 1)
            dt = (offsets[None, None, :].to(pooled.dtype) / 25.0).expand(b, t, -1)
            neighbours = interacted[:, safe_index]  # [B,T,5,S,D]
            neighbour_mask = mask[:, safe_index] & in_range[None, :, :, None]
            k_off = 5
            bias_term = self.lag_bias[:, (offsets + 2)].repeat_interleave(len(self.names), dim=-1)[None, None]

        keys = neighbours.reshape(b, t, k_off * len(self.names), -1)
        key_mask = neighbour_mask.reshape(b, t, k_off * len(self.names))
        q_lag = self.lag_q(pooled).view(b, t, self.lag_heads, self.lag_dh).unsqueeze(-2)  # [B,T,H,1,Dh]
        k_lag = self.lag_k(keys).view(b, t, -1, self.lag_heads, self.lag_dh).permute(0, 1, 3, 2, 4)
        v_lag = self.lag_v(keys).view(b, t, -1, self.lag_heads, self.lag_dh).permute(0, 1, 3, 2, 4)
        score = (q_lag * k_lag).sum(-1) / (self.lag_dh ** 0.5)
        score = score + bias_term
        score = score.masked_fill(~key_mask[:, :, None, :], torch.finfo(score.dtype).min)
        any_keys = key_mask.any(-1)[:, :, None, None]
        score = torch.where(any_keys, score, torch.zeros_like(score))
        lag_weights = torch.softmax(score, dim=-1)
        lag_weights = torch.where(any_keys, lag_weights, torch.zeros_like(lag_weights))
        aligned = (lag_weights.unsqueeze(-1) * v_lag).sum(-2).reshape(b, t, -1)
        fused = self.output(torch.cat([pooled, self.lag_out(aligned)], dim=-1))

        dt_expanded = dt.repeat_interleave(len(self.names), dim=-1)[:, :, None, :]
        expected_lag = (lag_weights * dt_expanded).sum(-1)
        aux = {
            "reliability": reliability,
            "expected_lag_seconds": expected_lag,
            "stream_mask": mask,
        }
        return fused, weights, interacted, projected, aux

    def forward(
        self,
        streams: dict[str, torch.Tensor],
        available: dict[str, torch.Tensor],
        quality: dict[str, torch.Tensor],
        timestamps: torch.Tensor | None = None,
    ):
        fused, weights, interacted, projected, _ = self.forward_with_aux(
            streams, available, quality, timestamps=timestamps
        )
        return fused, weights, interacted, projected
