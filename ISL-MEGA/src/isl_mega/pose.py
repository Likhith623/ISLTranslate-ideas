"""Anatomical graph pathway (66-point schema) with time-aware kinematics."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F

from .config import PoseConfig


# Input order: left hand 0:21, right hand 21:42, upper body 42:48, face 48:66.
LEFT = slice(0, 21)
RIGHT = slice(21, 42)
BODY = slice(42, 48)
FACE = slice(48, 66)

HAND_EDGES = tuple((a, b) for chain in (
    (0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
    (0, 13, 14, 15, 16), (0, 17, 18, 19, 20)
) for a, b in zip(chain, chain[1:]))
BODY_EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5))


def adjacency(num_nodes: int, edges: tuple[tuple[int, int], ...], device=None) -> torch.Tensor:
    a = torch.eye(num_nodes, dtype=torch.bool, device=device)
    for i, j in edges:
        a[i, j] = True
        a[j, i] = True
    return a


class EdgeGraphBlock(nn.Module):
    """Multi-head graph attention with pair-specific relative-coordinate bias."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge = nn.Sequential(nn.Linear(4, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, xy, valid, adj):
        # x [Nbatch,N,D], xy [Nbatch,N,2], valid [Nbatch,N]
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dh).transpose(1, 2)
        rel = xy[:, :, None, :] - xy[:, None, :, :]
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        edge = torch.cat([rel, dist, torch.ones_like(dist)], dim=-1)
        eb = self.edge(edge).view(b, n, n, self.heads, self.dh).permute(0, 3, 1, 2, 4)
        pair = k.unsqueeze(2) + eb
        scores = (q.unsqueeze(3) * pair).sum(-1) / math.sqrt(self.dh)
        allowed = adj[None, None] & valid[:, None, :, None] & valid[:, None, None, :]
        any_key = allowed.any(-1, keepdim=True)
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        scores = torch.where(any_key, scores, torch.zeros_like(scores))
        weights = torch.softmax(scores, dim=-1)
        weights = torch.where(any_key, weights, torch.zeros_like(weights))
        ctx = torch.matmul(self.dropout(weights), v).transpose(1, 2).reshape(b, n, d)
        x = x + self.dropout(self.out(ctx))
        x = torch.where(valid[..., None], x, torch.zeros_like(x))
        x = x + self.ffn(self.norm2(x))
        return torch.where(valid[..., None], x, torch.zeros_like(x))


class NodePool(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor, valid: torch.Tensor):
        score = self.score(x).squeeze(-1).masked_fill(~valid, torch.finfo(x.dtype).min)
        any_valid = valid.any(-1, keepdim=True)
        score = torch.where(any_valid, score, torch.zeros_like(score))
        w = torch.softmax(score, dim=-1)
        w = torch.where(any_valid, w, torch.zeros_like(w))
        return (x * w[..., None]).sum(-2), any_valid.squeeze(-1)


class RegionGraphEncoder(nn.Module):
    def __init__(self, n: int, dim: int, heads: int, layers: int, dropout: float,
                 edges: tuple[tuple[int, int], ...] | None = None, fully_connected: bool = False):
        super().__init__()
        self.n = n
        self.input = nn.Linear(11, dim)
        self.node_id = nn.Embedding(n, dim)
        self.layers = nn.ModuleList([EdgeGraphBlock(dim, heads, dropout) for _ in range(layers)])
        self.pool = NodePool(dim)
        self.register_buffer("adj", torch.ones(n, n, dtype=torch.bool) if fully_connected
                             else adjacency(n, edges or ()))

    def forward(self, features, xy, valid):
        b, t, n, _ = features.shape
        h = self.input(features) + self.node_id.weight[None, None]
        h = h.reshape(b * t, n, -1)
        p = xy.reshape(b * t, n, 2)
        m = valid.reshape(b * t, n).bool()
        for layer in self.layers:
            h = layer(h, p, m, self.adj)
        pooled, available = self.pool(h, m)
        return pooled.reshape(b, t, -1), available.reshape(b, t)


class PoseExpert(nn.Module):
    def __init__(self, config: PoseConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.left = RegionGraphEncoder(21, d, config.heads, config.layers, config.dropout, HAND_EDGES)
        self.right = RegionGraphEncoder(21, d, config.heads, config.layers, config.dropout, HAND_EDGES)
        self.body = RegionGraphEncoder(6, d, config.heads, config.layers, config.dropout, BODY_EDGES)
        self.face = RegionGraphEncoder(18, d, config.heads, config.layers, config.dropout, fully_connected=True)
        self.output_dim = d

    @staticmethod
    def _kinematic_features(xy, confidence, valid, confidence_known, timestamps):
        b, t, n, _ = xy.shape
        dt = timestamps[:, 1:] - timestamps[:, :-1]
        pair = valid[:, 1:] & valid[:, :-1] & (dt > 0)[..., None]
        velocity = torch.zeros_like(xy)
        if t > 1:
            raw = (xy[:, 1:] - xy[:, :-1]) / dt.clamp_min(1e-5)[..., None, None]
            velocity[:, 1:] = torch.where(pair[..., None], raw, torch.zeros_like(raw))
        acceleration = torch.zeros_like(xy)
        if t > 2:
            dt2 = dt[:, 1:].clamp_min(1e-5)
            good = pair[:, 1:] & pair[:, :-1]
            raw_a = (velocity[:, 2:] - velocity[:, 1:-1]) / dt2[..., None, None]
            acceleration[:, 2:] = torch.where(good[..., None], raw_a, torch.zeros_like(raw_a))
        # Bound extreme detector jitter without erasing the direction of motion.
        velocity = torch.tanh(velocity / 5.0)
        acceleration = torch.tanh(acceleration / 25.0)
        conf = torch.nan_to_num(confidence, nan=0.0).clamp(0, 1)
        ck = confidence_known.to(xy.dtype)
        features = torch.cat([xy, torch.zeros_like(xy), velocity, acceleration,
                              conf[..., None], valid[..., None].to(xy.dtype), ck[..., None].expand(-1, -1, n, -1)], dim=-1)
        return features

    def forward(self, pose_xy, pose_confidence, pose_valid, confidence_known, timestamps, frame_mask):
        if pose_xy.ndim != 4 or pose_xy.shape[2:] != (66, 2):
            raise ValueError("pose_xy must have shape [B,T,66,2]")
        b, t, _, _ = pose_xy.shape
        expected = (b, t, 66)
        if pose_confidence.shape != expected or pose_valid.shape != expected:
            raise ValueError("pose confidence/valid tensors must have shape [B,T,66]")
        if confidence_known.shape not in {expected, (b, t, 1)}:
            raise ValueError("confidence_known must be [B,T,66] or [B,T,1]")
        if timestamps.shape != (b, t) or frame_mask.shape != (b, t):
            raise ValueError("timestamps/frame_mask shape mismatch")
        valid = pose_valid.bool() & frame_mask[..., None].bool()
        # Apply the detector threshold only where confidence is genuinely known;
        # missing confidence metadata must not be misinterpreted as zero quality.
        if self.config.min_confidence > 0:
            known_mask = confidence_known.bool()
            if known_mask.shape[-1] == 1:
                known_mask = known_mask.expand(-1, -1, 66)
            valid = valid & (~known_mask | (pose_confidence >= self.config.min_confidence))
        xy = torch.nan_to_num(pose_xy, nan=0.0, posinf=0.0, neginf=0.0)
        conf = torch.nan_to_num(pose_confidence, nan=0.0).clamp(0, 1)
        if confidence_known.shape[-1] == 1:
            confidence_known = confidence_known.expand(-1, -1, 66)
        feat = self._kinematic_features(xy, conf, valid, confidence_known, timestamps)

        # Local coordinate anchors: hand wrist, shoulder midpoint, facial centroid.
        groups = {
            "left_hand": (LEFT, 0), "right_hand": (RIGHT, 0),
            "body": (BODY, None), "face": (FACE, None),
        }
        result, availability, quality = {}, {}, {}
        for name, (sl, wrist) in groups.items():
            gxy, gfeat, gvalid, gconf = xy[:, :, sl], feat[:, :, sl].clone(), valid[:, :, sl], conf[:, :, sl]
            if wrist is not None:
                anchor = gxy[:, :, wrist:wrist + 1]
                anchor_valid = gvalid[:, :, wrist:wrist + 1]
            elif name == "body":
                # The documented body order begins with left/right shoulders.
                # Prefer their midpoint; fall back to the visible body centroid
                # only when neither shoulder is available in that frame.
                shoulder_valid = gvalid[:, :, :2]
                shoulder_count = shoulder_valid.sum(dim=2, keepdim=True).clamp_min(1)
                shoulder_anchor = (gxy[:, :, :2] * shoulder_valid[..., None]).sum(dim=2, keepdim=True) / shoulder_count[..., None]
                all_count = gvalid.sum(dim=2, keepdim=True).clamp_min(1)
                body_centroid = (gxy * gvalid[..., None]).sum(dim=2, keepdim=True) / all_count[..., None]
                has_shoulder = shoulder_valid.any(dim=2, keepdim=True)
                anchor = torch.where(has_shoulder[..., None], shoulder_anchor, body_centroid)
                anchor_valid = gvalid.any(dim=2, keepdim=True)
            else:
                count = gvalid.sum(dim=2, keepdim=True).clamp_min(1)
                anchor = (gxy * gvalid[..., None]).sum(dim=2, keepdim=True) / count[..., None]
                anchor_valid = gvalid.any(dim=2, keepdim=True)
            rel = gxy - anchor
            rel = torch.where(anchor_valid[..., None], rel, torch.zeros_like(rel))
            gfeat[..., 2:4] = rel
            if name == "left_hand":
                encoded, avail = self.left(gfeat, gxy, gvalid)
            elif name == "right_hand":
                encoded, avail = self.right(gfeat, gxy, gvalid)
            elif name == "body":
                encoded, avail = self.body(gfeat, gxy, gvalid)
            else:
                encoded, avail = self.face(gfeat, gxy, gvalid)
            result[name] = encoded
            availability[name] = avail & frame_mask
            coverage = gvalid.float().mean(-1)
            conf_mean = (gconf * gvalid).sum(-1) / gvalid.sum(-1).clamp_min(1)
            known = confidence_known[:, :, sl].float().mean(-1)
            quality[name] = torch.stack([coverage, conf_mean, known, avail.float(), frame_mask.float()], dim=-1)
        return {"features": result, "available": availability, "quality": quality}
