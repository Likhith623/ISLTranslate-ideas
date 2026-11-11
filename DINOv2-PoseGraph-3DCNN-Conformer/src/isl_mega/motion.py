"""Full-resolution RGB spatiotemporal motion pathway using 3-D convolutions."""

from __future__ import annotations

import torch
from torch import nn

from .config import MotionConfig


class MotionExpert(nn.Module):
    """Encode body-crop appearance changes without temporal downsampling.

    The input is the current body crop concatenated with its absolute frame
    difference. Two 3-D convolutions operate over (time, height, width), both
    with temporal stride one. Spatial pooling retains one feature per original
    frame. A depthwise temporal refinement then models longer local motion.
    """

    def __init__(self, config: MotionConfig):
        super().__init__()
        d = config.hidden_dim
        self.spatial1 = nn.Conv3d(6, d // 2, kernel_size=(3, 5, 5), stride=(1, 2, 2),
                                  padding=(1, 2, 2), bias=False)
        self.spatial2 = nn.Conv3d(d // 2, d, kernel_size=(3, 3, 3), stride=(1, 2, 2),
                                  padding=(1, 1, 1), bias=False)
        k = config.temporal_kernel
        self.temporal = nn.Sequential(
            nn.Conv1d(d, d, k, padding=k // 2, groups=d),
            nn.GELU(),
            nn.Conv1d(d, d, 1),
            nn.Dropout(config.dropout),
        )
        self.norm = nn.LayerNorm(d)
        self.output_dim = d

    def forward(
        self,
        body_crops: torch.Tensor,
        frame_mask: torch.Tensor,
        body_available: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if body_crops.ndim != 6 or body_crops.shape[2] != 1:
            raise ValueError("body_crops must have shape [B,T,1,3,H,W]")
        x = body_crops[:, :, 0]
        b, t, c, h, w = x.shape
        if c != 3 or frame_mask.shape != (b, t):
            raise ValueError("Expected RGB body crops and frame_mask [B,T]")
        if body_available is None:
            body_available = frame_mask
        if body_available.shape != (b, t):
            raise ValueError("body_available must have shape [B,T]")

        valid = frame_mask.bool() & body_available.bool()
        x = torch.where(valid[..., None, None, None], x, torch.zeros_like(x))
        diff = torch.zeros_like(x)
        if t > 1:
            pair_valid = valid[:, 1:] & valid[:, :-1]
            delta = (x[:, 1:] - x[:, :-1]).abs()
            diff[:, 1:] = torch.where(pair_valid[..., None, None, None], delta,
                                      torch.zeros_like(delta))

        video = torch.cat([x, diff], dim=2).permute(0, 2, 1, 3, 4).contiguous()
        temporal_mask = valid[:, None, :, None, None]
        z = torch.nn.functional.gelu(self.spatial1(video))
        z = torch.where(temporal_mask, z, torch.zeros_like(z))
        z = torch.nn.functional.gelu(self.spatial2(z))
        z = torch.where(temporal_mask, z, torch.zeros_like(z))
        features = z.mean(dim=(-1, -2)).transpose(1, 2)
        if features.shape[1] != t:
            raise RuntimeError("Motion encoder must preserve the input frame count")
        temporal = self.temporal(features.transpose(1, 2)).transpose(1, 2)
        out = self.norm(features + temporal)
        return torch.where(valid[..., None], out, torch.zeros_like(out))
