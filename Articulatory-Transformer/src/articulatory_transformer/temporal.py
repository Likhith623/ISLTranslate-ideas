"""
Axial Spatio-Temporal Conformer and Multi-Task Fingerspelling CTC Branch.

Processes factored 4-articulator representations [B, T, 4, D] using alternating:
1. Spatial 4x4 Self-Attention: captures instantaneous bilateral contacts,
   hand-to-face contacts, and torso-relative spatial positioning.
2. Full-Resolution Temporal Conformer: Macaron FFN + MHSA + Depthwise 1D Conv
   operating at full video frame rate without lossy downsampling.
3. Dedicated Dominant-Hand Fingerspelling CTC Head: provides lexical anchoring
   and prevents linguistic hallucination.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class Swish(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * torch.sigmoid(x)


    class ConformerConvModule(nn.Module):
        """
        Conformer 1D temporal convolution block.
        LayerNorm -> Pointwise Conv -> Gated Linear Unit (GLU) ->
        Depthwise 1D Conv -> BatchNorm1d -> Swish -> Pointwise Conv -> Dropout.
        """
        def __init__(self, dim: int, kernel_size: int = 31, dropout: float = 0.1):
            super().__init__()
            assert kernel_size % 2 == 1, "Kernel size must be odd for symmetric padding"
            padding = (kernel_size - 1) // 2
            
            self.norm = nn.LayerNorm(dim)
            self.pointwise1 = nn.Conv1d(dim, dim * 2, kernel_size=1)
            self.depthwise = nn.Conv1d(
                dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
            )
            self.batch_norm = nn.BatchNorm1d(dim)
            self.act = Swish()
            self.pointwise2 = nn.Conv1d(dim, dim, kernel_size=1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (N, T, D)
            Returns:
                out: (N, T, D)
            """
            res = x
            x = self.norm(x)
            # Permute to (N, D, T) for Conv1d
            x = x.transpose(1, 2)
            
            x = self.pointwise1(x)
            x = F.glu(x, dim=1)  # (N, D, T)
            
            x = self.depthwise(x)
            x = self.batch_norm(x)
            x = self.act(x)
            x = self.pointwise2(x)
            x = self.dropout(x)
            
            x = x.transpose(1, 2)  # (N, T, D)
            return res + x


    class ConformerFeedForward(nn.Module):
        """Macaron-style half-step feed forward network."""
        def __init__(self, dim: int, expansion_factor: int = 4, dropout: float = 0.1):
            super().__init__()
            inner_dim = dim * expansion_factor
            self.norm = nn.LayerNorm(dim)
            self.net = nn.Sequential(
                nn.Linear(dim, inner_dim),
                Swish(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, dim),
                nn.Dropout(dropout),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return 0.5 * self.net(self.norm(x))


    class AxialSpatioTemporalBlock(nn.Module):
        """
        Single Axial Spatio-Temporal Conformer Block.
        Alternates between intra-frame 4x4 spatial attention and temporal Conformer.
        """
        def __init__(
            self,
            dim: int = 256,
            num_spatial_heads: int = 4,
            num_temporal_heads: int = 8,
            conv_kernel_size: int = 31,
            ffn_expansion_factor: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.dim = dim
            
            # 1. Spatial Attention across 4 articulators within each frame
            self.spatial_norm = nn.LayerNorm(dim)
            self.spatial_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_spatial_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.spatial_dropout = nn.Dropout(dropout)

            # 2. Temporal Conformer for each articulator
            self.temporal_ffn1 = ConformerFeedForward(dim, ffn_expansion_factor, dropout)
            self.temporal_norm_attn = nn.LayerNorm(dim)
            self.temporal_attn = nn.MultiheadAttention(
                embed_dim=dim,
                num_heads=num_temporal_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.temporal_dropout = nn.Dropout(dropout)
            self.temporal_conv = ConformerConvModule(dim, conv_kernel_size, dropout)
            self.temporal_ffn2 = ConformerFeedForward(dim, ffn_expansion_factor, dropout)
            self.temporal_norm_out = nn.LayerNorm(dim)

        def forward(
            self,
            x: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                x: (B, T, 4, D)
                temporal_mask: Optional (B, T) boolean mask where True indicates valid frame
            Returns:
                out: (B, T, 4, D)
            """
            B, T, A, D = x.shape
            assert A == 4, f"Expected 4 articulators, got {A}"

            # Step 1: Spatial Attention across 4 articulators within each frame
            # Reshape to (B * T, 4, D)
            x_sp = x.view(B * T, A, D)
            sp_normed = self.spatial_norm(x_sp)
            sp_out, _ = self.spatial_attn(query=sp_normed, key=sp_normed, value=sp_normed)
            x_sp = x_sp + self.spatial_dropout(sp_out)
            x = x_sp.view(B, T, A, D)

            # Step 2: Temporal Conformer across T frames for each articulator
            # Permute to (B, A, T, D) then reshape to (B * A, T, D)
            x_tp = x.permute(0, 2, 1, 3).contiguous().view(B * A, T, D)
            
            # Key padding mask for temporal attention
            key_padding_mask = None
            if temporal_mask is not None:
                # temporal_mask is (B, T) -> expand to (B, A, T) -> (B * A, T)
                # Note: PyTorch key_padding_mask expects True for positions to IGNORE
                invalid_mask = ~temporal_mask
                key_padding_mask = (
                    invalid_mask.unsqueeze(1)
                    .expand(-1, A, -1)
                    .contiguous()
                    .view(B * A, T)
                )

            # Macaron FFN 1
            x_tp = x_tp + self.temporal_ffn1(x_tp)

            # Temporal Self-Attention
            tp_normed = self.temporal_norm_attn(x_tp)
            tp_attn_out, _ = self.temporal_attn(
                query=tp_normed,
                key=tp_normed,
                value=tp_normed,
                key_padding_mask=key_padding_mask,
            )
            x_tp = x_tp + self.temporal_dropout(tp_attn_out)

            # Temporal Convolution
            x_tp = self.temporal_conv(x_tp)

            # Macaron FFN 2
            x_tp = x_tp + self.temporal_ffn2(x_tp)
            x_tp = self.temporal_norm_out(x_tp)

            # Reshape back to (B, T, A, D)
            out = x_tp.view(B, A, T, D).permute(0, 2, 1, 3).contiguous()
            return out


    class AxialSpatioTemporalConformer(nn.Module):
        """
        Full stack of Axial Spatio-Temporal Conformer blocks with Dominant Hand CTC head.
        """
        def __init__(
            self,
            num_layers: int = 6,
            dim: int = 256,
            num_spatial_heads: int = 4,
            num_temporal_heads: int = 8,
            conv_kernel_size: int = 31,
            ffn_expansion_factor: int = 4,
            dropout: float = 0.1,
            enable_ctc: bool = True,
            ctc_vocab_size: int = 64,
        ):
            super().__init__()
            self.num_layers = num_layers
            self.dim = dim
            self.enable_ctc = enable_ctc

            self.layers = nn.ModuleList([
                AxialSpatioTemporalBlock(
                    dim=dim,
                    num_spatial_heads=num_spatial_heads,
                    num_temporal_heads=num_temporal_heads,
                    conv_kernel_size=conv_kernel_size,
                    ffn_expansion_factor=ffn_expansion_factor,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

            # Multi-Task CTC Head on Dominant Hand channel (articulator index 0)
            if enable_ctc:
                self.ctc_head = nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, ctc_vocab_size),
                )
            else:
                self.ctc_head = None

        def forward(
            self,
            x: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            Args:
                x: (B, T, 4, D)
                temporal_mask: Optional (B, T)
            Returns:
                x: (B, T, 4, D) contextualized spatio-temporal representations.
                ctc_logits: Optional (T, B, vocab_size) for CTCLoss.
            """
            for layer in self.layers:
                x = layer(x, temporal_mask=temporal_mask)

            ctc_logits = None
            if self.enable_ctc and self.ctc_head is not None:
                # Dominant Hand is index 0: shape (B, T, D)
                dh_trajectory = x[:, :, 0, :]
                logits = self.ctc_head(dh_trajectory)  # (B, T, vocab_size)
                # Transpose to (T, B, vocab_size) for standard torch.nn.CTCLoss
                ctc_logits = logits.transpose(0, 1)

            return x, ctc_logits
else:
    class AxialSpatioTemporalConformer:  # type: ignore
        def __init__(self, *args, **kwargs): pass

