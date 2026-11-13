"""Actual local RGB observations, encoded without assuming pose determines texture."""

import torch
from torch import nn


class CropEncoder(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True):
        super().__init__()
        self.cfg = cfg
        if cfg.visual_backend == "cnn":
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 24, 3, stride=2, padding=1),
                nn.GroupNorm(4, 24),
                nn.GELU(),
                nn.Conv2d(24, 48, 3, stride=2, padding=1),
                nn.GroupNorm(4, 48),
                nn.GELU(),
                nn.Conv2d(48, 64, 3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d(cfg.visual_grid),
                nn.Flatten(),
            )
            channels = 64 * cfg.visual_grid ** 2
        elif cfg.visual_backend == "dinov2":
            from transformers import AutoConfig, AutoModel

            if initialize_pretrained:
                self.backbone = AutoModel.from_pretrained(
                    cfg.visual_model, revision=cfg.visual_revision, trust_remote_code=False
                )
                cfg.visual_config = self.backbone.config.to_dict()
            else:
                raw = dict(cfg.visual_config)
                kind = raw.pop("model_type")
                self.backbone = AutoModel.from_config(AutoConfig.for_model(kind, **raw))
            if self.backbone.config.model_type != "dinov2":
                raise ValueError("visual_model must be a DINOv2 model")
            channels = self.backbone.config.hidden_size
        else:
            raise ValueError("CropEncoder requires cnn or dinov2")
        self.projection = nn.Linear(channels, cfg.visual_dim)
        if cfg.visual_frozen:
            self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.cfg.visual_frozen:
            self.backbone.eval()
        return self

    def forward(self, rgb, mask, lengths):
        if rgb.ndim != 6 or rgb.shape[2:4] != (self.cfg.visual_regions, 3):
            raise ValueError("rgb must be [B,T,R,3,H,W]")
        if mask.shape != rgb.shape[:3] or mask.dtype != torch.bool:
            raise ValueError("rgb_mask must be boolean [B,T,R]")
        valid = mask & (torch.arange(rgb.shape[1], device=rgb.device)[None, :, None] < lengths[:, None, None])
        flat = rgb.reshape(-1, *rgb.shape[3:])
        index = valid.flatten().nonzero().flatten()
        output = rgb.new_zeros(flat.shape[0], self.cfg.visual_dim)
        if index.numel():
            selected = flat.index_select(0, index)
            if self.cfg.visual_backend == "dinov2" and min(selected.shape[-2:]) < self.backbone.config.patch_size:
                raise ValueError("DINOv2 crop must cover at least one patch")
            if not torch.isfinite(selected).all() or ((selected < 0) | (selected > 1)).any():
                raise ValueError("observed RGB crops must be finite in [0,1]")
            encoded = []
            for chunk in selected.split(self.cfg.visual_chunk_size):
                if self.cfg.visual_backend == "dinov2":
                    mean = chunk.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
                    std = chunk.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
                    hidden = self.backbone(pixel_values=(chunk - mean) / std).last_hidden_state
                    feature = hidden[:, 1:].mean(1)
                else:
                    feature = self.backbone(chunk)
                encoded.append(self.projection(feature))
            values = torch.cat(encoded).to(output)
            output = output.index_copy(0, index, values)
        return output.reshape(*rgb.shape[:3], self.cfg.visual_dim), valid
