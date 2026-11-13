"""
Factored 4-Articulator Token Representation.

Partitions 66 Joint-Visual Tokens into 4 linguistically grounded articulatory streams:
1. Dominant Hand (DH)   : 21 landmarks
2. Base Hand (BH)       : 21 landmarks
3. Face / Mouthing (FC) : 18 landmarks
4. Torso / Anchor (TR)  : 6 landmarks

Applies intra-articulator attention pooling to produce [B, T, 4, D],
preserving distinct articulator dynamics without premature 1D spatiotemporal collapse.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class ArticulatorAttentionPooler(nn.Module):
        """
        Pools variable number of joint tokens within an articulator group into a single token
        using a learned query vector and multi-head cross-attention.
        """
        def __init__(self, embed_dim: int = 256, num_heads: int = 4, dropout: float = 0.1):
            super().__init__()
            self.embed_dim = embed_dim
            self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.attn = nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.norm = nn.LayerNorm(embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 2, embed_dim),
            )
            self.norm_ffn = nn.LayerNorm(embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (N, num_joints, D) where N = B * T
            Returns:
                pooled: (N, D) single pooled vector
            """
            N, V, D = x.shape
            q = self.query.expand(N, -1, -1)  # (N, 1, D)
            
            # Cross-attention: query attends to all joints in the articulator
            attn_out, _ = self.attn(query=q, key=x, value=x)
            h = self.norm(q + attn_out)
            
            out = self.norm_ffn(h + self.ffn(h))
            return out.squeeze(1)  # (N, D)


    class FactoredArticulatorGrouper(nn.Module):
        """
        Groups 66 canonical Joint-Visual tokens into 4 distinct articulators:
        [Dominant Hand, Base Hand, Face, Torso].
        """
        def __init__(
            self,
            embed_dim: int = 256,
            aggregation_heads: int = 4,
            dropout: float = 0.1,
            left_hand_slice: Tuple[int, int] = (0, 21),
            right_hand_slice: Tuple[int, int] = (21, 42),
            body_slice: Tuple[int, int] = (42, 48),
            face_slice: Tuple[int, int] = (48, 66),
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.left_slice = left_hand_slice
            self.right_slice = right_hand_slice
            self.body_slice = body_slice
            self.face_slice = face_slice

            # Dedicated pooling attention for each articulator
            self.pool_dh = ArticulatorAttentionPooler(embed_dim, aggregation_heads, dropout)
            self.pool_bh = ArticulatorAttentionPooler(embed_dim, aggregation_heads, dropout)
            self.pool_fc = ArticulatorAttentionPooler(embed_dim, aggregation_heads, dropout)
            self.pool_tr = ArticulatorAttentionPooler(embed_dim, aggregation_heads, dropout)

            # Learned Articulator Type Embeddings (4 articulators)
            # 0: DH, 1: BH, 2: FC, 3: TR
            self.articulator_embed = nn.Parameter(torch.randn(1, 1, 4, embed_dim) * 0.02)
            self.norm = nn.LayerNorm(embed_dim)

        def forward(
            self,
            joint_tokens: torch.Tensor,
            is_right_dominant: torch.Tensor,
        ) -> torch.Tensor:
            """
            Args:
                joint_tokens: (B, T, 66, D)
                is_right_dominant: (B,) boolean tensor
            Returns:
                articulators: (B, T, 4, D)
            """
            B, T, V, D = joint_tokens.shape
            
            # Slices
            lh = joint_tokens[:, :, self.left_slice[0]:self.left_slice[1], :]   # (B, T, 21, D)
            rh = joint_tokens[:, :, self.right_slice[0]:self.right_slice[1], :]  # (B, T, 21, D)
            tr = joint_tokens[:, :, self.body_slice[0]:self.body_slice[1], :]   # (B, T, 6, D)
            fc = joint_tokens[:, :, self.face_slice[0]:self.face_slice[1], :]   # (B, T, 18, D)

            # Route hands according to dominance
            mask = is_right_dominant.view(B, 1, 1, 1).expand(-1, T, 21, D)
            dh = torch.where(mask, rh, lh)  # (B, T, 21, D)
            bh = torch.where(mask, lh, rh)  # (B, T, 21, D)

            # Flatten (B*T) for attention pooling
            N = B * T
            u_dh = self.pool_dh(dh.view(N, 21, D)).view(B, T, 1, D)
            u_bh = self.pool_bh(bh.view(N, 21, D)).view(B, T, 1, D)
            u_fc = self.pool_fc(fc.view(N, 18, D)).view(B, T, 1, D)
            u_tr = self.pool_tr(tr.view(N, 6, D)).view(B, T, 1, D)

            # Stack 4 articulators: [B, T, 4, D]
            tokens_4 = torch.cat([u_dh, u_bh, u_fc, u_tr], dim=2)
            
            # Add articulator type embedding
            out = self.norm(tokens_4 + self.articulator_embed)
            return out
else:
    class FactoredArticulatorGrouper:  # type: ignore
        def __init__(self, *args, **kwargs): pass

