"""Anatomical graph and set encoders for RA-MAGC."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HandGraphConfig, BodyGraphConfig, FaceSetConfig


class EdgeAwareGraphAttention(nn.Module):
    r"""Multi-head graph attention with pair-specific edge-feature injection.

    For query node ``i`` and key node ``j`` the score is

        q_i^T (k_j + W_e e_ij) / sqrt(d_head).

    The pair axes are explicit; this is important because a transposed/broadcast
    einsum can silently produce plausible shapes while computing the wrong score.
    """

    def __init__(self, node_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        if node_dim <= 0 or num_heads <= 0 or node_dim % num_heads:
            raise ValueError("node_dim must be positive and divisible by num_heads")
        if edge_dim <= 0:
            raise ValueError("edge_dim must be positive")
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        edge_bias: torch.Tensor,
    ) -> torch.Tensor:
        """Return ``[BT,H,N_query,N_key]`` scores with the intended Q_i/K_j pairing."""
        if q.ndim != 4 or k.shape != q.shape:
            raise ValueError("q and k must have matching [BT,H,N,Dh] shapes")
        expected = (*q.shape[:2], q.shape[2], k.shape[2], q.shape[3])
        if edge_bias.shape != expected:
            raise ValueError(f"edge_bias must have shape {expected}, got {tuple(edge_bias.shape)}")
        pair_keys = k.unsqueeze(2) + edge_bias                 # [BT,H,N_query,N_key,Dh]
        scores = (q.unsqueeze(3) * pair_keys).sum(dim=-1)     # [BT,H,N_query,N_key]
        return scores / math.sqrt(self.head_dim)

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adjacency_mask: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.ndim != 3:
            raise ValueError("nodes must have shape [BT,N,D]")
        bt, n, dim = nodes.shape
        if dim != self.node_dim:
            raise ValueError(f"Expected node dimension {self.node_dim}, got {dim}")
        if edges.shape[:3] != (bt, n, n):
            raise ValueError("edges must have shape [BT,N,N,edge_dim]")
        if edges.shape[-1] != self.edge_proj.in_features:
            raise ValueError("edge feature dimension does not match edge_proj")
        if adjacency_mask.shape != (bt, n, n):
            raise ValueError("adjacency_mask must have shape [BT,N,N]")

        residual = nodes
        q = self.q_proj(nodes).view(bt, n, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(nodes).view(bt, n, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(nodes).view(bt, n, self.num_heads, self.head_dim).transpose(1, 2)
        edge_bias = self.edge_proj(edges).view(
            bt, n, n, self.num_heads, self.head_dim
        ).permute(0, 3, 1, 2, 4)

        scores = self.attention_scores(q, k, edge_bias)
        allowed = adjacency_mask.to(dtype=torch.bool).unsqueeze(1)  # [BT,1,Nq,Nk]
        has_keys = allowed.any(dim=-1, keepdim=True)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        safe_scores = torch.where(has_keys, scores, torch.zeros_like(scores))
        weights = F.softmax(safe_scores, dim=-1)
        weights = torch.where(has_keys, weights, torch.zeros_like(weights))
        weights = self.dropout(weights)

        context = torch.matmul(weights, v).transpose(1, 2).reshape(bt, n, dim)
        update = self.out_proj(context)
        return self.norm(residual + self.dropout(update))


class MaskedAttentionPooling(nn.Module):
    """Learned pooling over valid joints; all-invalid frames return exact zeros."""

    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score_net = nn.Sequential(nn.Linear(node_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))

    def forward(self, nodes: torch.Tensor, node_valid_mask: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 4 or node_valid_mask.shape != nodes.shape[:3]:
            raise ValueError("nodes/mask must have shapes [B,T,N,D] and [B,T,N]")
        valid = node_valid_mask.bool()
        scores = self.score_net(nodes).squeeze(-1).masked_fill(~valid, torch.finfo(nodes.dtype).min)
        any_valid = valid.any(dim=-1, keepdim=True)
        safe_scores = torch.where(any_valid, scores, torch.zeros_like(scores))
        weights = F.softmax(safe_scores, dim=-1)
        weights = torch.where(any_valid, weights, torch.zeros_like(weights))
        pooled = (nodes * weights.unsqueeze(-1)).sum(dim=2)
        return torch.where(any_valid, pooled, torch.zeros_like(pooled))


class HandGraphEncoder(nn.Module):
    """Shared-weight graph encoder for left/right 21-joint hands."""

    def __init__(self, config: HandGraphConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.joint_embed = nn.Embedding(config.num_joints, config.hidden_dim)
        self.side_embed = nn.Embedding(2, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeAwareGraphAttention(config.hidden_dim, config.edge_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adj_mask: torch.Tensor,
        valid_mask: torch.Tensor,
        side_idx: int,
    ) -> torch.Tensor:
        if side_idx not in (0, 1):
            raise ValueError("side_idx must be 0 (left) or 1 (right)")
        if nodes.ndim != 4 or nodes.shape[2:] != (21, self.config.in_node_dim):
            raise ValueError("hand nodes must have shape [B,T,21,13]")
        b, t, n, _ = nodes.shape
        if edges.shape != (b, t, n, n, self.config.edge_dim):
            raise ValueError("hand edges must have shape [B,T,21,21,4]")
        if adj_mask.shape != (b, t, n, n) or valid_mask.shape != (b, t, n):
            raise ValueError("hand adjacency/validity mask shape mismatch")

        side = self.side_embed.weight[side_idx]
        joint_ids = torch.arange(n, device=nodes.device)
        h = self.node_proj(nodes) + self.joint_embed(joint_ids)[None, None] + side[None, None, None]
        bt = b * t
        h = h.reshape(bt, n, -1)
        e = edges.reshape(bt, n, n, -1)
        a = adj_mask.reshape(bt, n, n)
        for layer in self.layers:
            h = layer(h, e, a)
        return self.pool(h.reshape(b, t, n, -1), valid_mask)


class UpperBodyGraphEncoder(nn.Module):
    """Graph encoder for shoulders, elbows and wrists."""

    def __init__(self, config: BodyGraphConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.joint_embed = nn.Embedding(config.num_joints, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                EdgeAwareGraphAttention(config.hidden_dim, config.edge_dim, config.num_heads, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(
        self,
        nodes: torch.Tensor,
        edges: torch.Tensor,
        adj_mask: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if nodes.ndim != 4 or nodes.shape[2:] != (6, self.config.in_node_dim):
            raise ValueError("body nodes must have shape [B,T,6,10]")
        b, t, n, _ = nodes.shape
        if edges.shape != (b, t, n, n, self.config.edge_dim):
            raise ValueError("body edges must have shape [B,T,6,6,4]")
        if adj_mask.shape != (b, t, n, n) or valid_mask.shape != (b, t, n):
            raise ValueError("body adjacency/validity mask shape mismatch")
        joint_ids = torch.arange(n, device=nodes.device)
        h = self.node_proj(nodes) + self.joint_embed(joint_ids)[None, None]
        bt = b * t
        h = h.reshape(bt, n, -1)
        e = edges.reshape(bt, n, n, -1)
        a = adj_mask.reshape(bt, n, n)
        for layer in self.layers:
            h = layer(h, e, a)
        return self.pool(h.reshape(b, t, n, -1), valid_mask)


class FaceSetEncoder(nn.Module):
    """Set-attention encoder for the selected 18 facial landmarks."""

    def __init__(self, config: FaceSetConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.point_embed = nn.Embedding(config.num_points, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(self, nodes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if nodes.ndim != 4 or nodes.shape[2:] != (18, self.config.in_node_dim):
            raise ValueError("face nodes must have shape [B,T,18,13]")
        b, t, n, _ = nodes.shape
        if valid_mask.shape != (b, t, n):
            raise ValueError("face validity mask must have shape [B,T,18]")
        h = self.node_proj(nodes) + self.point_embed(torch.arange(n, device=nodes.device))[None, None]
        flat = h.reshape(b * t, n, -1)
        valid = valid_mask.reshape(b * t, n).bool()
        any_valid = valid.any(dim=-1, keepdim=True)
        # Avoid all-masked self-attention rows; discard their outputs immediately after.
        safe_valid = torch.where(any_valid, valid, torch.ones_like(valid))
        encoded = self.transformer(flat, src_key_padding_mask=~safe_valid)
        encoded = torch.where(any_valid[..., None], encoded, torch.zeros_like(encoded))
        return self.pool(encoded.reshape(b, t, n, -1), valid_mask)
