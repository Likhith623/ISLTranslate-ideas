"""Padding-invariant temporal CNN, packed BiLSTM and optional synchronized RGB fusion."""

from contextlib import nullcontext

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from isl_slt.config import Config


def length_mask(lengths: torch.Tensor, maximum: int, device=None) -> torch.Tensor:
    return torch.arange(maximum, device=device or lengths.device)[None] < lengths.to(device)[:, None]


class TemporalCNN(nn.Module):
    def __init__(self, input_dim: int, channels: tuple[int, int], kernel: int, reduction: int, dropout: float):
        super().__init__()
        self.strides = (1, 2) if reduction == 2 else (2, 2)
        self.convolutions = nn.ModuleList()
        self.norms = nn.ModuleList()
        for width, stride in zip(channels, self.strides):
            self.convolutions.append(nn.Conv1d(input_dim, width, kernel, stride=stride, padding=kernel // 2))
            self.norms.append(nn.LayerNorm(width))
            input_dim = width
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor, lengths: torch.Tensor):
        values = values * length_mask(lengths, values.shape[1], values.device)[..., None]
        for convolution, norm, stride in zip(self.convolutions, self.norms, self.strides):
            values = convolution(values.transpose(1, 2)).transpose(1, 2)
            lengths = (lengths + stride - 1) // stride
            mask = length_mask(lengths, values.shape[1], values.device)
            values = self.dropout(torch.nn.functional.gelu(norm(values))) * mask[..., None]
        return values, lengths


class VisualEncoder(nn.Module):
    def __init__(self, input_dim: int, config: Config, initialize_rgb: bool = True):
        super().__init__()
        model = config.model
        self.cnn = TemporalCNN(input_dim, model.cnn_channels, model.kernel_size, model.downsample, model.dropout)
        self.reduction, self.recurrent_fp32 = model.downsample, config.runtime.recurrent_fp32
        self.rgb_encoder = None
        fusion_dim = model.cnn_channels[-1]
        if config.rgb.mode != "off":
            views = 3 if config.rgb.hand_crops else 1
            self.rgb_projection = nn.Sequential(nn.Linear(views * 513, 256), nn.LayerNorm(256), nn.GELU())
            fusion_dim += 256
            if config.rgb.mode == "images":
                from isl_slt.features.rgb import RGBEncoder
                self.rgb_encoder = RGBEncoder(config.rgb.weights_path if initialize_rgb else None)
        else:
            self.rgb_projection = None
        self.recurrent = nn.LSTM(fusion_dim, model.lstm_hidden, num_layers=model.lstm_layers,
                                 batch_first=True, bidirectional=True, dropout=model.dropout)
        self.norm = nn.LayerNorm(model.lstm_hidden * 2)
        self.dropout = nn.Dropout(model.dropout)
        self.output_dim = model.lstm_hidden * 2

    def forward(self, features: torch.Tensor, lengths: torch.Tensor, rgb=None, rgb_mask=None, images=None):
        values, output_lengths = self.cnn(features, lengths)
        if self.rgb_projection is not None:
            if images is not None:
                if self.rgb_encoder is None:
                    raise ValueError("This model expects cached vectors, not images")
                rgb = self.rgb_encoder(images)
            if rgb is None or rgb_mask is None or rgb.shape[:2] != features.shape[:2]:
                raise ValueError("RGB fusion requires synchronized vectors/images and crop masks")
            # Convolution center indices are 0, reduction, 2*reduction, ... for symmetric padding.
            indices = (torch.arange(values.shape[1], device=values.device) * self.reduction).clamp(max=rgb.shape[1] - 1)
            sampled, valid = rgb[:, indices], rgb_mask[:, indices]
            augmented = torch.cat([sampled * valid[..., None], valid[..., None].to(sampled.dtype)], dim=-1)
            values = torch.cat([values, self.rgb_projection(augmented.flatten(2))], dim=-1)
        context = torch.autocast(values.device.type, enabled=False) if self.recurrent_fp32 else nullcontext()
        with context:
            recurrent_input = values.float() if self.recurrent_fp32 else values
            packed = pack_padded_sequence(recurrent_input, output_lengths.cpu(), batch_first=True, enforce_sorted=False)
            output, _ = self.recurrent(packed)
            states, _ = pad_packed_sequence(output, batch_first=True, total_length=values.shape[1])
            states = self.dropout(self.norm(states))
        mask = length_mask(output_lengths, states.shape[1], states.device)
        return states * mask[..., None], mask


def masked_mean(states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (states * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
