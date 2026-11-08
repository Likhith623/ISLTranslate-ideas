"""Full-resolution multi-scale front end, coarse global transformer and detail retrieval."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import TemporalConfig


def make_length_mask(lengths: torch.Tensor, maximum: int) -> torch.Tensor:
    return torch.arange(maximum, device=lengths.device)[None] < lengths[:, None]


class MultiScaleTemporalStem(nn.Module):
    """Parallel temporal convolutions retaining every input time position."""

    def __init__(self, dim: int, kernels: tuple[int, ...], dilations: tuple[int, ...], dropout: float):
        super().__init__()
        if dim % len(kernels):
            raise ValueError("temporal dim must be divisible by number of convolution branches")
        branch = dim // len(kernels)
        self.branches = nn.ModuleList([
            nn.Conv1d(dim, branch, kernel_size=k, dilation=d, padding=(k // 2) * d)
            for k, d in zip(kernels, dilations)
        ])
        self.project = nn.Conv1d(dim, dim, 1)
        self.norm_in = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or mask.shape != x.shape[:2]:
            raise ValueError("stem expects x [B,T,D], mask [B,T]")
        mask = mask.bool()
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        normed = torch.where(mask[..., None], self.norm_in(x), torch.zeros_like(x))
        z = normed.transpose(1, 2)
        branches = [conv(z) for conv in self.branches]
        y = self.project(torch.cat(branches, dim=1)).transpose(1, 2)
        y = torch.where(mask[..., None], y, torch.zeros_like(y))
        out = self.norm_out(x + self.dropout(F.gelu(y)))
        return torch.where(mask[..., None], out, torch.zeros_like(out))


class RelativeTimeBias(nn.Module):
    def __init__(self, heads: int, buckets: int, max_seconds: float):
        super().__init__()
        self.heads, self.buckets, self.max_seconds = heads, buckets, max_seconds
        self.table = nn.Embedding(buckets, heads)

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        delta = times[:, :, None] - times[:, None, :]
        half = self.buckets // 2
        magnitude = delta.abs().clamp(1e-5, self.max_seconds)
        scaled = torch.log(magnitude / 1e-5) / math.log(self.max_seconds / 1e-5)
        bucket = torch.floor(scaled * (half - 1)).long().clamp(0, half - 1)
        bucket = bucket + (delta >= 0).long() * half
        return self.table(bucket).permute(0, 3, 1, 2).contiguous()


class GlobalTemporalBlock(nn.Module):
    """Pre-LN relative-time self-attention + local depthwise convolution."""

    def __init__(self, dim: int, heads: int, ffn_dim: int, dropout: float, conv_kernel: int = 15):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.conv_in = nn.Linear(dim, dim * 2)
        self.depthwise = nn.Conv1d(dim, dim, conv_kernel, padding=conv_kernel // 2, groups=dim)
        self.conv_out = nn.Linear(dim, dim)
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(ffn_dim, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, time_bias):
        b, t, d = x.shape
        mask = mask.bool()
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        n = torch.where(mask[..., None], self.norm1(x), torch.zeros_like(x))
        q, k, v = self.qkv(n).chunk(3, dim=-1)
        q = q.view(b, t, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, t, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, t, self.heads, self.dh).transpose(1, 2)
        score = q @ k.transpose(-2, -1) / math.sqrt(self.dh) + time_bias
        allowed = mask[:, None, None, :]
        score = score.masked_fill(~allowed, torch.finfo(score.dtype).min)
        weights = torch.softmax(score, dim=-1)
        context = (self.attn_drop(weights) @ v).transpose(1, 2).reshape(b, t, d)
        x = x + self.dropout(self.attn_out(context))
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        # Convolution sees zeros at padding and masks biased intermediates before mixing.
        a, gate = self.conv_in(torch.where(mask[..., None], self.norm2(x), torch.zeros_like(x))).chunk(2, -1)
        z = a * torch.sigmoid(gate)
        z = torch.where(mask[..., None], z, torch.zeros_like(z))
        z = self.depthwise(z.transpose(1, 2)).transpose(1, 2)
        z = torch.where(mask[..., None], F.gelu(z), torch.zeros_like(z))
        x = x + self.dropout(self.conv_out(z))
        x = torch.where(mask[..., None], x, torch.zeros_like(x))
        x = x + self.ffn(torch.where(mask[..., None], self.norm3(x), torch.zeros_like(x)))
        return torch.where(mask[..., None], x, torch.zeros_like(x))


class FineDetailRetrieval(nn.Module):
    """Local cross-attention from coarse global tokens to original fine tokens."""

    def __init__(self, dim: int, heads: int, window: int = 7, stride: int = 2):
        super().__init__()
        if dim % heads or window % 2 == 0:
            raise ValueError("dim must divide heads and window must be odd")
        if stride < 2:
            raise ValueError("detail retrieval stride must be >= 2")
        self.heads, self.dh, self.window, self.stride = heads, dim // heads, window, stride
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, coarse: torch.Tensor, fine: torch.Tensor, lengths: torch.Tensor):
        b, nc, d = coarse.shape
        t = fine.shape[1]
        radius = self.window // 2
        nominal_centres = torch.arange(nc, device=fine.device) * self.stride + (self.stride - 1) // 2
        # Centre each sample's partial final group on its own last valid frame;
        # a shared padded batch length must not move the query off that sample.
        centres = torch.minimum(nominal_centres[None], (lengths - 1)[:, None]).clamp(0, t - 1)
        offsets = torch.arange(-radius, radius + 1, device=fine.device)
        idx = centres[:, :, None] + offsets[None, None, :]
        in_range = (idx >= 0) & (idx < t)
        idx_safe = idx.clamp(0, t - 1)
        gather_index = idx_safe.reshape(b, -1).unsqueeze(-1).expand(-1, -1, d)
        gathered = torch.gather(fine, dim=1, index=gather_index).reshape(b, nc, self.window, d)
        valid = in_range & (idx_safe < lengths[:, None, None])
        q = self.q(coarse).view(b, nc, self.heads, self.dh).transpose(1, 2).unsqueeze(3)
        k = self.k(gathered).view(b, nc, self.window, self.heads, self.dh).permute(0, 3, 1, 2, 4)
        v = self.v(gathered).view(b, nc, self.window, self.heads, self.dh).permute(0, 3, 1, 2, 4)
        score = (q * k).sum(-1) / math.sqrt(self.dh)
        score = score.masked_fill(~valid[:, None], torch.finfo(score.dtype).min)
        any_valid = valid.any(-1)[:, None, :, None]
        score = torch.where(any_valid, score, torch.zeros_like(score))
        weight = torch.softmax(score, dim=-1)
        weight = torch.where(any_valid, weight, torch.zeros_like(weight))
        context = (weight.unsqueeze(-1) * v).sum(-2).transpose(1, 2).reshape(b, nc, d)
        return self.norm(coarse + self.out(context))


class HierarchicalTemporalEncoder(nn.Module):
    """Fine-rate local path + half-rate global path + local detail retrieval.

    No input frame is discarded: the fine branch remains length T. The global
    branch has ceil(T/stride) tokens and is broadcast back to the fine timeline.
    """

    def __init__(self, config: TemporalConfig):
        super().__init__()
        self.config = config
        self.stem = MultiScaleTemporalStem(config.dim, config.local_kernels,
                                           config.local_dilations, config.dropout)
        self.coarse_projection = nn.Sequential(nn.Linear(config.dim * 2, config.dim),
                                                nn.LayerNorm(config.dim), nn.GELU())
        self.time_projection = nn.Linear(2, config.dim)
        self.global_blocks = nn.ModuleList([
            GlobalTemporalBlock(config.dim, config.heads, config.ffn_dim, config.dropout)
            for _ in range(config.global_layers)
        ])
        self.time_bias = RelativeTimeBias(config.heads, config.relative_time_buckets,
                                          config.max_elapsed_seconds)
        self.detail = FineDetailRetrieval(config.dim, config.heads, config.detail_window, config.coarse_stride)
        self.final = nn.Sequential(nn.Linear(config.dim * 2, config.dim), nn.LayerNorm(config.dim), nn.GELU())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, timestamps: torch.Tensor):
        if x.ndim != 3 or lengths.shape != (x.shape[0],) or timestamps.shape != x.shape[:2]:
            raise ValueError("temporal input shapes must be [B,T,D], [B], [B,T]")
        b, t, d = x.shape
        if torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError("lengths must be in [1,T]")
        fine_mask = make_length_mask(lengths, t)
        if not torch.isfinite(timestamps[fine_mask]).all():
            raise ValueError("valid timestamps must be finite")
        valid_dt = timestamps[:, 1:] - timestamps[:, :-1]
        pairs = torch.arange(t - 1, device=x.device)[None] < (lengths - 1)[:, None]
        if t > 1 and torch.any(pairs & (valid_dt <= 0)):
            raise ValueError("valid timestamps must be strictly increasing")
        fine = self.stem(x, fine_mask)
        stride = self.config.coarse_stride
        pad = (stride - (t % stride)) % stride
        if pad:
            fine_pad = F.pad(fine, (0, 0, 0, pad))
            time_pad = F.pad(timestamps, (0, pad), value=0.0)
        else:
            fine_pad, time_pad = fine, timestamps
        tp = fine_pad.shape[1]
        grouped = fine_pad.reshape(b, tp // stride, stride, d)
        # Pair mean and first-to-last difference provide a learned summary while
        # the fine-rate tensor remains available to the detail path.
        group_valid = (torch.arange(tp, device=x.device)[None, :] < lengths[:, None]).reshape(b, -1, stride)
        group_count = group_valid.sum(dim=-1).clamp_min(1)
        grouped_mean = (grouped * group_valid[..., None]).sum(dim=2) / group_count[..., None]
        # Difference uses the final *valid* item in each group. This matters
        # for a partial final group and for batches with differently sized clips.
        has_multiple = group_valid.sum(dim=-1) > 1
        last_index = (group_valid.sum(dim=-1) - 1).clamp_min(0)
        last_valid = torch.gather(
            grouped, dim=2,
            index=last_index[..., None, None].expand(-1, -1, 1, d),
        ).squeeze(2)
        first_valid = grouped[:, :, 0]
        grouped_diff = torch.where(has_multiple[..., None], last_valid - first_valid,
                                   torch.zeros_like(first_valid))
        coarse = self.coarse_projection(torch.cat([grouped_mean, grouped_diff], dim=-1))
        times_g = time_pad.reshape(b, -1, stride)
        # group_valid is reused from the masked temporal aggregation above.
        coarse_times = (times_g * group_valid).sum(dim=-1) / group_count
        coarse_lengths = torch.div(lengths + stride - 1, stride, rounding_mode="floor")
        coarse_mask = make_length_mask(coarse_lengths, coarse.shape[1])
        elapsed = coarse_times - timestamps[:, :1]
        scale = torch.stack([elapsed, torch.log1p(elapsed.clamp_min(0))], dim=-1)
        coarse = coarse + self.time_projection(scale.to(coarse.dtype))
        coarse = torch.where(coarse_mask[..., None], coarse, torch.zeros_like(coarse))
        bias = self.time_bias(coarse_times)
        for block in self.global_blocks:
            coarse = block(coarse, coarse_mask, bias)
        retrieved = self.detail(coarse, fine, lengths)
        global_up = retrieved.repeat_interleave(stride, dim=1)[:, :t]
        out = self.final(torch.cat([fine, global_up], dim=-1))
        out = torch.where(fine_mask[..., None], out, torch.zeros_like(out))
        return out, fine_mask, {"fine": fine, "global": coarse, "global_times": coarse_times,
                                "global_mask": coarse_mask, "retrieved_global": retrieved}
