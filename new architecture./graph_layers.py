"""Anatomical Graph and Set Encoders for Hand, Upper-Body, and Facial Streams.

Implements topology-aware message passing with physical bone-edge attributes,
joint-validity masking, and learned masked attention pooling.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HandGraphConfig, BodyGraphConfig, FaceSetConfig


class EdgeAwareGraphAttention(nn.Module):
    """
    Multi-head graph attention layer with bone-edge attribute injection.
    
    Given node features h_i and bone-edge features e_ij:
      attention_score(i, j) = (Q h_i)^T (K h_j + W_edge e_ij) / sqrt(d)
    """
    def __init__(self, node_dim: int, edge_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.node_dim = node_dim
        self.num_heads = num_heads
        self.head_dim = node_dim // num_heads
        assert self.head_dim * num_heads == node_dim, "node_dim must be divisible by num_heads"
        
        self.q_proj = nn.Linear(node_dim, node_dim)
        self.k_proj = nn.Linear(node_dim, node_dim)
        self.v_proj = nn.Linear(node_dim, node_dim)
        self.edge_proj = nn.Linear(edge_dim, node_dim)
        self.out_proj = nn.Linear(node_dim, node_dim)
        
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, adjacency_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [Batch * Time, Num_Nodes, node_dim]
            edges: [Batch * Time, Num_Nodes, Num_Nodes, edge_dim]
            adjacency_mask: [Batch * Time, Num_Nodes, Num_Nodes] boolean mask of valid connected edges
        """
        BT, N, D = nodes.shape
        res = nodes
        
        q = self.q_proj(nodes).view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)  # [BT, H, N, Dh]
        k = self.k_proj(nodes).view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)  # [BT, H, N, Dh]
        v = self.v_proj(nodes).view(BT, N, self.num_heads, self.head_dim).transpose(1, 2)  # [BT, H, N, Dh]
        
        edge_bias = self.edge_proj(edges).view(BT, N, N, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        
        # Compute attention scores: (q_i * (k_j + edge_ij))
        scores = torch.einsum("bhnd,bhmnd->bhnm", q, k.unsqueeze(2) + edge_bias) / math.sqrt(self.head_dim)
        
        # Mask non-adjacent or invalid joint connections
        if adjacency_mask is not None:
            mask = adjacency_mask.unsqueeze(1)  # [BT, 1, N, N]
            has_connections = mask.any(dim=-1, keepdim=True)  # [BT, 1, N, 1]
            scores = scores.masked_fill(~mask, -1e9)
            safe_scores = torch.where(~has_connections, torch.zeros_like(scores), scores)
            attn_weights = F.softmax(safe_scores, dim=-1)
            attn_weights = torch.where(~has_connections, torch.zeros_like(attn_weights), attn_weights)
        else:
            attn_weights = F.softmax(scores, dim=-1)
            
        attn_weights = self.dropout(attn_weights)
        
        out = torch.einsum("bhnm,bhmd->bhnd", attn_weights, v)
        out = out.transpose(1, 2).reshape(BT, N, D)
        out = self.out_proj(out)
        
        return self.norm(res + self.dropout(out))


class MaskedAttentionPooling(nn.Module):
    """
    Learned attention pooling across joints in an anatomical region.
    Guarantees missing/invalid joints do not leak zero-valued activations into region tokens.
    """
    def __init__(self, node_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.score_net = nn.Sequential(
            nn.Linear(node_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, nodes: torch.Tensor, node_valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [B, T, Num_Nodes, node_dim]
            node_valid_mask: [B, T, Num_Nodes] boolean mask of observed/valid joints
        Returns:
            pooled: [B, T, node_dim]
        """
        scores = self.score_net(nodes).squeeze(-1)  # [B, T, Num_Nodes]
        scores = scores.masked_fill(~node_valid_mask, -1e9)
        
        # If all joints in a frame are invalid, softmax will produce NaNs without safe handling
        all_invalid = ~node_valid_mask.any(dim=-1, keepdim=True)  # [B, T, 1]
        safe_scores = torch.where(all_invalid, torch.zeros_like(scores), scores)
        
        weights = F.softmax(safe_scores, dim=-1).unsqueeze(-1)  # [B, T, Num_Nodes, 1]
        pooled = (nodes * weights).sum(dim=2)  # [B, T, node_dim]
        
        # Zero out completely invalid frames
        pooled = torch.where(all_invalid, torch.zeros_like(pooled), pooled)
        return pooled


class HandGraphEncoder(nn.Module):
    """
    Shared-weight anatomical hand graph encoder for left and right hands.
    
    Shares parameters across hands while retaining lateral distinction via
    learned left/right side embeddings.
    """
    def __init__(self, config: HandGraphConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.joint_embed = nn.Embedding(config.num_joints, config.hidden_dim)
        self.side_embed = nn.Embedding(2, config.hidden_dim)  # 0: left, 1: right
        
        self.layers = nn.ModuleList([
            EdgeAwareGraphAttention(
                node_dim=config.hidden_dim,
                edge_dim=config.edge_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, 
                adj_mask: torch.Tensor, valid_mask: torch.Tensor, 
                side_idx: int) -> torch.Tensor:
        """
        Args:
            nodes: [B, T, 21, in_node_dim]
            edges: [B, T, 21, 21, edge_dim]
            adj_mask: [B, T, 21, 21]
            valid_mask: [B, T, 21]
            side_idx: 0 for left hand, 1 for right hand
        Returns:
            hand_token: [B, T, hidden_dim]
        """
        B, T, N, D = nodes.shape
        side_tensor = torch.tensor(side_idx, device=nodes.device)
        
        # Initial projection + joint embedding + lateral side embedding
        h = self.node_proj(nodes)  # [B, T, 21, hidden_dim]
        joint_ids = torch.arange(N, device=nodes.device)
        h = h + self.joint_embed(joint_ids)[None, None, :, :] + self.side_embed(side_tensor)[None, None, None, :]
        
        # Flatten B and T for graph attention layers
        BT = B * T
        h_flat = h.view(BT, N, -1)
        edges_flat = edges.view(BT, N, N, -1)
        adj_flat = adj_mask.view(BT, N, N)
        
        for layer in self.layers:
            h_flat = layer(h_flat, edges_flat, adj_flat)
            
        h_out = h_flat.view(B, T, N, -1)
        pooled = self.pool(h_out, valid_mask)
        return pooled


class UpperBodyGraphEncoder(nn.Module):
    """
    Upper-body skeleton graph encoder over 6 key joints:
    Left/Right Shoulders, Elbows, and Wrists.
    """
    def __init__(self, config: BodyGraphConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.joint_embed = nn.Embedding(config.num_joints, config.hidden_dim)
        
        self.layers = nn.ModuleList([
            EdgeAwareGraphAttention(
                node_dim=config.hidden_dim,
                edge_dim=config.edge_dim,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.num_layers)
        ])
        
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(self, nodes: torch.Tensor, edges: torch.Tensor, 
                adj_mask: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [B, T, 6, in_node_dim]
            edges: [B, T, 6, 6, edge_dim]
            adj_mask: [B, T, 6, 6]
            valid_mask: [B, T, 6]
        Returns:
            body_token: [B, T, hidden_dim]
        """
        B, T, N, D = nodes.shape
        h = self.node_proj(nodes) + self.joint_embed(torch.arange(N, device=nodes.device))[None, None, :, :]
        
        BT = B * T
        h_flat = h.view(BT, N, -1)
        edges_flat = edges.view(BT, N, N, -1)
        adj_flat = adj_mask.view(BT, N, N)
        
        for layer in self.layers:
            h_flat = layer(h_flat, edges_flat, adj_flat)
            
        h_out = h_flat.view(B, T, N, -1)
        return self.pool(h_out, valid_mask)


class FaceSetEncoder(nn.Module):
    """
    Sparse facial landmark set-attention encoder over 18 keypoints:
    Eyes, Eyebrows, and Mouth landmarks.
    """
    def __init__(self, config: FaceSetConfig):
        super().__init__()
        self.config = config
        self.node_proj = nn.Linear(config.in_node_dim, config.hidden_dim)
        self.point_embed = nn.Embedding(config.num_points, config.hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 2,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers, enable_nested_tensor=False)
        self.pool = MaskedAttentionPooling(config.hidden_dim)
        self.output_dim = config.hidden_dim

    def forward(self, nodes: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            nodes: [B, T, 18, in_node_dim]
            valid_mask: [B, T, 18]
        Returns:
            face_token: [B, T, hidden_dim]
        """
        B, T, N, D = nodes.shape
        h = self.node_proj(nodes) + self.point_embed(torch.arange(N, device=nodes.device))[None, None, :, :]
        
        BT = B * T
        h_flat = h.view(BT, N, -1)
        valid_flat = valid_mask.view(BT, N)
        has_any = valid_flat.any(dim=-1, keepdim=True)  # [BT, 1]
        
        # Safe padding mask: if all points in a frame are invalid, unmask all to prevent 0/0 NaNs in softmax
        safe_valid = torch.where(~has_any, torch.ones_like(valid_flat), valid_flat)
        padding_mask = ~safe_valid
        
        # Self-attention over facial keypoints
        out_flat = self.transformer(h_flat, src_key_padding_mask=padding_mask)
        # Broadcast [BT, 1, 1] across [BT, 18, D] (AUDIT_REPORT.md Section 3.2)
        out_flat = torch.where(~has_any.unsqueeze(-1), torch.zeros_like(out_flat), out_flat)
        out = out_flat.view(B, T, N, -1)
        return self.pool(out, valid_mask)

