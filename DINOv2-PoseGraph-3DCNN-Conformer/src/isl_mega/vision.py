"""RGB region extraction and specialist appearance encoders.

ROI order is fixed: face, left hand, right hand, upper-body/global signer view.
All boxes are normalized xyxy coordinates in [0, 1].
"""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import VisionConfig


ROI_NAMES = ("face", "left_hand", "right_hand", "body")


def crop_regions(
    frames: torch.Tensor,
    boxes: torch.Tensor,
    box_valid: torch.Tensor,
    output_size: int,
) -> torch.Tensor:
    """Bilinearly crop 4 ROIs per frame using grid_sample.

    Args:
      frames: [B,T,3,H,W], float RGB in [0,1].
      boxes: [B,T,4,4], normalized xyxy in ROI_NAMES order.
      box_valid: [B,T,4] boolean.
    Returns:
      crops: [B,T,4,3,output_size,output_size]; invalid ROIs are exact zeros.
    """
    if frames.ndim != 5 or frames.shape[2] != 3:
        raise ValueError("frames must have shape [B,T,3,H,W]")
    b, t, _, h, w = frames.shape
    if boxes.shape != (b, t, 4, 4) or box_valid.shape != (b, t, 4):
        raise ValueError("boxes/box_valid must have shapes [B,T,4,4] and [B,T,4]")
    if not torch.isfinite(boxes).all():
        raise ValueError("ROI boxes must be finite")
    valid = box_valid.bool()
    xyxy = boxes.clamp(0.0, 1.0)
    if torch.any(valid & ((xyxy[..., 2] <= xyxy[..., 0]) | (xyxy[..., 3] <= xyxy[..., 1]))):
        raise ValueError("Valid ROI boxes must have positive width and height")

    # Flatten frame/ROI pairs; grid_sample remains differentiable w.r.t. frame pixels.
    image = frames[:, :, None].expand(-1, -1, 4, -1, -1, -1).reshape(b * t * 4, 3, h, w)
    box = xyxy.reshape(-1, 4)
    n = box.shape[0]
    u = torch.linspace(0.0, 1.0, output_size, device=frames.device, dtype=frames.dtype)
    v = torch.linspace(0.0, 1.0, output_size, device=frames.device, dtype=frames.dtype)
    yy, xx = torch.meshgrid(v, u, indexing="ij")
    x = box[:, 0, None, None] + xx[None] * (box[:, 2] - box[:, 0])[:, None, None]
    y = box[:, 1, None, None] + yy[None] * (box[:, 3] - box[:, 1])[:, None, None]
    grid = torch.stack((x * 2.0 - 1.0, y * 2.0 - 1.0), dim=-1)
    sampled = F.grid_sample(image, grid, mode="bilinear", padding_mode="border", align_corners=True)
    sampled = sampled * valid.reshape(n, 1, 1, 1).to(sampled.dtype)
    return sampled.reshape(b, t, 4, 3, output_size, output_size)


class PatchVisionTransformer(nn.Module):
    """Native ViT encoder used for tests and sign-specific self-supervised training."""

    def __init__(self, image_size: int, patch_size: int, dim: int, depth: int, heads: int,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        n = (image_size // patch_size) ** 2
        self.patch = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos = nn.Parameter(torch.zeros(1, n + 1, dim))
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout, activation="gelu", batch_first=True, norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(dim)
        self.dim = dim
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("vision images must have shape [N,3,H,W]")
        x = self.patch(images).flatten(2).transpose(1, 2)
        cls = self.cls.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        if x.shape[1] != self.pos.shape[1]:
            # Interpolate patch positions for different crop dimensions.
            cls_pos = self.pos[:, :1]
            patch_pos = self.pos[:, 1:]
            old = int(math.sqrt(patch_pos.shape[1]))
            new_h, new_w = self.patch(images).shape[-2:]
            patch_pos = patch_pos.reshape(1, old, old, self.dim).permute(0, 3, 1, 2)
            patch_pos = F.interpolate(patch_pos, size=(new_h, new_w), mode="bicubic", align_corners=False)
            patch_pos = patch_pos.flatten(2).transpose(1, 2)
            pos = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos = self.pos
        return self.norm(self.encoder(x + pos)[:, 0])


class TimmBackbone(nn.Module):
    """Optional pretrained DINOv2/timm wrapper; never downloads unless requested."""

    def __init__(self, model_name: str, pretrained: bool):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("Install ISL-MEGA with `pip install -e '.[vision]'` for timm backbones") from exc
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=0,
                                       dynamic_img_size=True)
        self.dim = int(getattr(self.model, "num_features"))
        pretrained_cfg = getattr(self.model, "pretrained_cfg", {}) or {}
        mean = pretrained_cfg.get("mean", (0.485, 0.456, 0.406))
        std = pretrained_cfg.get("std", (0.229, 0.224, 0.225))
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        images = (images - self.mean.to(images)) / self.std.to(images)
        out = self.model(images)
        if isinstance(out, (tuple, list)):
            out = out[-1]
        if isinstance(out, dict):
            out = out.get("x_norm_clstoken", out.get("features", next(iter(out.values()))))
        if out.ndim == 3:
            out = out[:, 0] if out.shape[1] > 1 else out[:, 0]
        if out.ndim != 2:
            raise RuntimeError(f"timm backbone must return [N,D], received {tuple(out.shape)}")
        return out


class VisualBackbone(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        if config.backend == "native":
            self.encoder = PatchVisionTransformer(
                config.crop_size, config.patch_size, config.dim, config.depth,
                config.heads, config.mlp_ratio, config.dropout,
            )
            input_dim = config.dim
        elif config.backend == "timm":
            self.encoder = TimmBackbone(config.timm_model_name, config.pretrained)
            input_dim = self.encoder.dim
        else:
            raise ValueError(f"Unsupported vision backend: {config.backend}")
        self.projection = nn.Identity() if input_dim == config.dim else nn.Linear(input_dim, config.dim)
        self.norm = nn.LayerNorm(config.dim)
        if config.freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.norm(self.projection(self.encoder(images)))


class AppearanceExpert(nn.Module):
    """Specialist hand, face and upper-body visual streams.

    Left/right hands share a hand encoder; face and body have independent
    encoders/adapters. Setting separate_hand_face_body=False shares all weights.
    """

    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        self.hand = VisualBackbone(config)
        if config.separate_hand_face_body:
            self.face = VisualBackbone(config)
            self.body = VisualBackbone(config)
        else:
            self.face = self.hand
            self.body = self.hand
        self.output_dim = config.dim

    def forward(self, crops: torch.Tensor, roi_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        if crops.ndim != 6 or crops.shape[2] != 4:
            raise ValueError("crops must have shape [B,T,4,3,H,W]")
        if roi_valid.shape != crops.shape[:3]:
            raise ValueError("roi_valid must have shape [B,T,4]")
        b, t, r, c, h, w = crops.shape
        flat = crops.reshape(b * t * r, c, h, w)
        valid = roi_valid.reshape(b * t * r).bool()
        outputs = {}
        for name, index, backbone in (
            ("face", 0, self.face), ("left_hand", 1, self.hand),
            ("right_hand", 2, self.hand), ("body", 3, self.body),
        ):
            x = crops[:, :, index].reshape(b * t, c, h, w)
            m = roi_valid[:, :, index].reshape(b * t).bool()
            # Encode only present regions to avoid wasting work; scatter back into T.
            encoded = x.new_zeros((b * t, self.output_dim))
            if m.any():
                encoded[m] = backbone(x[m])
            outputs[name] = encoded.reshape(b, t, self.output_dim)
        return outputs


def roi_quality(
    crops: torch.Tensor,
    boxes: torch.Tensor,
    box_valid: torch.Tensor,
    box_scores: torch.Tensor | None = None,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Five per-ROI descriptors: presence, area, crop sharpness, detector score, frame validity.

    ``box_scores`` should contain detector confidence when available. If the
    detector did not expose a score, callers may provide the documented
    presence proxy; do not interpret that proxy as calibrated confidence.
    """
    if crops.ndim != 6 or crops.shape[2] != 4:
        raise ValueError("crops must be [B,T,4,3,H,W]")
    b, t = crops.shape[:2]
    if boxes.shape != (b, t, 4, 4) or box_valid.shape != (b, t, 4):
        raise ValueError("boxes/box_valid shapes do not match crops")
    if frame_mask is None:
        frame_mask = torch.ones((b, t), dtype=torch.bool, device=crops.device)
    if frame_mask.shape != (b, t):
        raise ValueError("frame_mask must be [B,T]")
    valid = box_valid.bool() & frame_mask[..., None].bool()
    area = ((boxes[..., 2] - boxes[..., 0]).clamp_min(0) *
            (boxes[..., 3] - boxes[..., 1]).clamp_min(0)).clamp(0, 1)
    gray = crops.mean(dim=3, keepdim=True)
    dx = (gray[..., :, 1:] - gray[..., :, :-1]).abs().mean(dim=(-1, -2, -3))
    dy = (gray[..., 1:, :] - gray[..., :-1, :]).abs().mean(dim=(-1, -2, -3))
    sharp = ((dx + dy) * 0.5).clamp(0, 1)
    if box_scores is None:
        score = valid.to(crops.dtype)  # Explicit uncalibrated availability proxy.
    else:
        if box_scores.shape != (b, t, 4):
            raise ValueError("box_scores must be [B,T,4]")
        score = torch.nan_to_num(box_scores, nan=0.0).clamp(0, 1)
    present = valid.to(crops.dtype)
    return torch.stack([present, area, sharp, score, frame_mask[..., None].expand(-1, -1, 4).to(crops.dtype)], dim=-1)
