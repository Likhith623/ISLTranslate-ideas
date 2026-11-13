"""
Articulatory Q-Former Resampler.

Condenses variable-length 4-articulator spatio-temporal representations [B, T, 4, D]
into a fixed number of linguistic prompt tokens (e.g. M=32) for cross-attention
in the text decoder (FLAN-T5 or LLaMA-3.2).
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
    class QFormerLayer(nn.Module):
        """
        Single Q-Former block:
        Self-Attention among queries -> Cross-Attention to spatio-temporal tokens -> FFN.
        """
        def __init__(
            self,
            query_dim: int = 512,
            kv_dim: int = 256,
            num_heads: int = 8,
            ffn_expansion: int = 4,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.query_dim = query_dim
            
            # 1. Query Self-Attention
            self.self_attn_norm = nn.LayerNorm(query_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=query_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.self_attn_dropout = nn.Dropout(dropout)

            # 2. Cross-Attention: Queries attend to Keys/Values from video encoder
            self.cross_attn_norm = nn.LayerNorm(query_dim)
            self.kv_proj_k = nn.Linear(kv_dim, query_dim)
            self.kv_proj_v = nn.Linear(kv_dim, query_dim)
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=query_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.cross_attn_dropout = nn.Dropout(dropout)

            # 3. FFN
            self.ffn_norm = nn.LayerNorm(query_dim)
            inner_dim = query_dim * ffn_expansion
            self.ffn = nn.Sequential(
                nn.Linear(query_dim, inner_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(inner_dim, query_dim),
                nn.Dropout(dropout),
            )

        def forward(
            self,
            queries: torch.Tensor,
            kv_tokens: torch.Tensor,
            kv_padding_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                queries: (B, M, D_q)
                kv_tokens: (B, 4*T, D_kv)
                kv_padding_mask: Optional (B, 4*T) boolean mask (True for padding to ignore)
            Returns:
                queries: (B, M, D_q)
            """
            # 1. Self-Attention
            q_norm = self.self_attn_norm(queries)
            sa_out, _ = self.self_attn(query=q_norm, key=q_norm, value=q_norm)
            queries = queries + self.self_attn_dropout(sa_out)

            # 2. Cross-Attention
            q_norm = self.cross_attn_norm(queries)
            k = self.kv_proj_k(kv_tokens)
            v = self.kv_proj_v(kv_tokens)
            ca_out, _ = self.cross_attn(
                query=q_norm,
                key=k,
                value=v,
                key_padding_mask=kv_padding_mask,
            )
            queries = queries + self.cross_attn_dropout(ca_out)

            # 3. FFN
            queries = queries + self.ffn(self.ffn_norm(queries))
            return queries


    class ArticulatoryQFormer(nn.Module):
        """
        Resamples [B, T, 4, D] articulatory spatiotemporal sequences into
        a fixed-length prompt tensor [B, num_queries, output_dim].
        """
        def __init__(
            self,
            num_queries: int = 32,
            input_dim: int = 256,
            query_dim: int = 512,
            output_dim: int = 768,
            num_layers: int = 4,
            num_heads: int = 8,
            dropout: float = 0.1,
        ):
            super().__init__()
            self.num_queries = num_queries
            self.input_dim = input_dim
            self.query_dim = query_dim
            self.output_dim = output_dim

            # Learnable linguistic query vectors
            self.query_embeddings = nn.Parameter(
                torch.randn(1, num_queries, query_dim) * 0.02
            )

            # Stack of Q-Former layers
            self.layers = nn.ModuleList([
                QFormerLayer(
                    query_dim=query_dim,
                    kv_dim=input_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ])

            self.final_norm = nn.LayerNorm(query_dim)
            self.output_proj = nn.Linear(query_dim, output_dim)

        def forward(
            self,
            articulatory_tokens: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
        ) -> torch.Tensor:
            """
            Args:
                articulatory_tokens: (B, T, 4, D)
                temporal_mask: Optional (B, T) boolean mask where True is valid frame
            Returns:
                prompts: (B, num_queries, output_dim)
            """
            B, T, A, D = articulatory_tokens.shape
            assert A == 4, f"Expected 4 articulators, got {A}"

            # Flatten across time and articulators: (B, 4*T, D)
            kv_tokens = articulatory_tokens.view(B, T * A, D)

            # Construct key padding mask for (B, 4*T)
            kv_padding_mask = None
            if temporal_mask is not None:
                # temporal_mask: (B, T), True=valid -> invalid is ~temporal_mask
                invalid_frames = ~temporal_mask  # (B, T)
                # Expand to (B, T, 4) then view (B, 4*T)
                kv_padding_mask = (
                    invalid_frames.unsqueeze(-1)
                    .expand(-1, -1, A)
                    .contiguous()
                    .view(B, T * A)
                )

            # Initialize queries for batch
            queries = self.query_embeddings.expand(B, -1, -1)  # (B, M, D_q)

            # Process through Q-Former layers
            for layer in self.layers:
                queries = layer(
                    queries=queries,
                    kv_tokens=kv_tokens,
                    kv_padding_mask=kv_padding_mask,
                )

            queries = self.final_norm(queries)
            prompt_tokens = self.output_proj(queries)  # (B, M, output_dim)
            return prompt_tokens
else:
    class ArticulatoryQFormer:  # type: ignore
        def __init__(self, *args, **kwargs): pass

