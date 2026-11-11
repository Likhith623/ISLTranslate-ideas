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
# Wrist (0) and the five fingertips (4, 8, 12, 16, 20) carry primary
# phonological handshape/location/contact geometry across regions.
HAND_KEY_INDICES = (0, 4, 8, 12, 16, 20)
CROSS_RBF_SIGMAS = (0.10, 0.25, 0.60, 1.50)


def adjacency(num_nodes: int, edges: tuple[tuple[int, int], ...], device=None) -> torch.Tensor:
    a = torch.eye(num_nodes, dtype=torch.bool, device=device)
    for i, j in edges:
        a[i, j] = True
        a[j, i] = True
    return a


def cross_articulator_adjacency(
    n_left: int = 6,
    n_right: int = 6,
    n_body: int = 6,
    n_face: int = 18,
    device=None,
) -> torch.Tensor:
    """Bipartite phonological location graph linking hands to face, body and opposite hand."""
    total = n_left + n_right + n_body + n_face
    adj = torch.eye(total, dtype=torch.bool, device=device)
    s_l = slice(0, n_left)
    s_r = slice(n_left, n_left + n_right)
    s_non_hand = slice(n_left + n_right, total)
    # Left <-> right hand, and each hand <-> body/face location anchors.
    adj[s_l, s_r] = True
    adj[s_r, s_l] = True
    adj[s_l, s_non_hand] = True
    adj[s_non_hand, s_l] = True
    adj[s_r, s_non_hand] = True
    adj[s_non_hand, s_r] = True
    return adj


def estimate_torso_scale(xy: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Estimate per-frame torso scale from shoulder span with geometric fallbacks.

    Returns tensor of shape [B, T] safely bounded in [0.05, 2.0].
    """
    b_xy, b_valid = xy[:, :, BODY], valid[:, :, BODY]
    both_shoulders = b_valid[:, :, 0] & b_valid[:, :, 1]
    shoulder_span = torch.linalg.vector_norm(b_xy[:, :, 0] - b_xy[:, :, 1], dim=-1)

    def _spread(pts: torch.Tensor, m: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        big = torch.full_like(pts, -1e4)
        small = torch.full_like(pts, 1e4)
        hi = torch.where(m[..., None], pts, big).amax(dim=2)
        lo = torch.where(m[..., None], pts, small).amin(dim=2)
        diag = torch.linalg.vector_norm((hi - lo).clamp_min(0.0), dim=-1)
        return diag, m.sum(dim=2) >= 2

    body_diag, body_ok = _spread(b_xy, b_valid)
    face_diag, face_ok = _spread(xy[:, :, FACE], valid[:, :, FACE])
    scale = torch.ones_like(shoulder_span)
    scale = torch.where(face_ok, face_diag * 1.25, scale)
    scale = torch.where(body_ok, body_diag * 0.65, scale)
    scale = torch.where(both_shoulders, shoulder_span, scale)
    return scale.clamp(0.05, 2.0)


class EdgeGraphBlock(nn.Module):
    """Multi-head graph attention with pair-specific relative-coordinate bias."""

    def __init__(self, dim: int, heads: int, dropout: float, edge_feat_dim: int = 4):
        super().__init__()
        self.dim, self.heads, self.dh = dim, heads, dim // heads
        self.edge_feat_dim = edge_feat_dim
        self.qkv = nn.Linear(dim, 3 * dim)
        self.edge = nn.Sequential(nn.Linear(edge_feat_dim, dim), nn.Tanh(), nn.Linear(dim, dim))
        self.out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(dim * 4, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        xy: torch.Tensor,
        valid: torch.Tensor,
        adj: torch.Tensor,
        scale: torch.Tensor | None = None,
        edge_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x [Nbatch,N,D], xy [Nbatch,N,2], valid [Nbatch,N]
        b, n, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).chunk(3, dim=-1)
        q = q.view(b, n, self.heads, self.dh).transpose(1, 2)
        k = k.view(b, n, self.heads, self.dh).transpose(1, 2)
        v = v.view(b, n, self.heads, self.dh).transpose(1, 2)
        if edge_features is None:
            rel = xy[:, :, None, :] - xy[:, None, :, :]
            if scale is not None:
                rel = rel / scale.view(b, 1, 1, 1)
            dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
            edge = torch.cat([rel, dist, torch.ones_like(dist)], dim=-1)
        else:
            edge = edge_features
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

    def forward_nodes(
        self,
        features: torch.Tensor,
        xy: torch.Tensor,
        valid: torch.Tensor,
        scale: torch.Tensor | None = None,
        extra_emb: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, n, _ = features.shape
        h = self.input(features) + self.node_id.weight[None, None]
        if extra_emb is not None:
            h = h + extra_emb
        m = valid.reshape(b * t, n).bool()
        h = h.reshape(b * t, n, -1)
        h = torch.where(m[..., None], h, torch.zeros_like(h))
        p = xy.reshape(b * t, n, 2)
        s = scale.reshape(b * t) if scale is not None else None
        for layer in self.layers:
            h = layer(h, p, m, self.adj, scale=s)
        return h.reshape(b, t, n, -1), m.reshape(b, t, n)

    def forward(self, features, xy, valid, scale=None, extra_emb=None):
        b, t, n, _ = features.shape
        h, m = self.forward_nodes(features, xy, valid, scale=scale, extra_emb=extra_emb)
        pooled, available = self.pool(h.reshape(b * t, n, -1), m.reshape(b * t, n))
        return pooled.reshape(b, t, -1), available.reshape(b, t)


class CrossArticulatorGraph(nn.Module):
    """Phonology-grounded relational graph across hand tips/wrists, body and face."""

    def __init__(self, dim: int, heads: int, layers: int, dropout: float):
        super().__init__()
        self.region_emb = nn.Embedding(4, dim)
        self.blocks = nn.ModuleList([
            EdgeGraphBlock(dim, heads, dropout, edge_feat_dim=4 + len(CROSS_RBF_SIGMAS))
            for _ in range(layers)
        ])
        self.gate = nn.Parameter(torch.zeros(1))
        self.register_buffer("adj", cross_articulator_adjacency())
        self.register_buffer("rbf_sigmas", torch.tensor(CROSS_RBF_SIGMAS, dtype=torch.float32))

    def forward(
        self,
        node_feats: dict[str, torch.Tensor],
        xy: torch.Tensor,
        valid: torch.Tensor,
        scale: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        b, t = xy.shape[:2]
        idx = torch.tensor(HAND_KEY_INDICES, device=xy.device, dtype=torch.long)
        h_l = node_feats["left_hand"][:, :, idx]
        h_r = node_feats["right_hand"][:, :, idx]
        h_b = node_feats["body"]
        h_f = node_feats["face"]

        p_l = xy[:, :, LEFT][:, :, idx]
        p_r = xy[:, :, RIGHT][:, :, idx]
        p_b = xy[:, :, BODY]
        p_f = xy[:, :, FACE]

        v_l = valid[:, :, LEFT][:, :, idx]
        v_r = valid[:, :, RIGHT][:, :, idx]
        v_b = valid[:, :, BODY]
        v_f = valid[:, :, FACE]

        r_ids = torch.cat([
            torch.zeros(6, device=xy.device, dtype=torch.long),
            torch.ones(6, device=xy.device, dtype=torch.long),
            torch.full((6,), 2, device=xy.device, dtype=torch.long),
            torch.full((18,), 3, device=xy.device, dtype=torch.long),
        ])
        x = torch.cat([h_l, h_r, h_b, h_f], dim=2) + self.region_emb(r_ids)[None, None]
        p = torch.cat([p_l, p_r, p_b, p_f], dim=2).reshape(b * t, 36, 2)
        m = torch.cat([v_l, v_r, v_b, v_f], dim=2).reshape(b * t, 36).bool()
        s = scale.reshape(b * t, 1, 1, 1)

        rel = (p[:, :, None, :] - p[:, None, :, :]) / s
        dist = torch.linalg.vector_norm(rel, dim=-1, keepdim=True)
        rbf = torch.exp(-0.5 * (dist / self.rbf_sigmas.to(dist).view(1, 1, 1, -1)).square())
        edge_feat = torch.cat([rel, dist, rbf, torch.ones_like(dist)], dim=-1)

        h_flat = x.reshape(b * t, 36, -1)
        h0 = h_flat
        for block in self.blocks:
            h_flat = block(h_flat, p, m, self.adj, edge_features=edge_feat)
        delta = torch.tanh(self.gate) * (h_flat - h0)
        delta = torch.where(m[..., None], delta, torch.zeros_like(delta)).reshape(b, t, 36, -1)

        out = {k: v.clone() for k, v in node_feats.items()}
        out["left_hand"][:, :, idx] = out["left_hand"][:, :, idx] + delta[:, :, 0:6]
        out["right_hand"][:, :, idx] = out["right_hand"][:, :, idx] + delta[:, :, 6:12]
        out["body"] = out["body"] + delta[:, :, 12:18]
        out["face"] = out["face"] + delta[:, :, 18:36]
        return out


class PoseExpert(nn.Module):
    def __init__(self, config: PoseConfig):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        if config.share_hand_encoder:
            self.left = RegionGraphEncoder(21, d, config.heads, config.layers, config.dropout, HAND_EDGES)
            self.right = self.left
            self.hand_side_emb = nn.Embedding(2, d)
        else:
            self.left = RegionGraphEncoder(21, d, config.heads, config.layers, config.dropout, HAND_EDGES)
            self.right = RegionGraphEncoder(21, d, config.heads, config.layers, config.dropout, HAND_EDGES)
            self.hand_side_emb = None
        self.body = RegionGraphEncoder(6, d, config.heads, config.layers, config.dropout, BODY_EDGES)
        self.face = RegionGraphEncoder(18, d, config.heads, config.layers, config.dropout, fully_connected=True)
        self.cross_graph = (
            CrossArticulatorGraph(d, config.heads, config.cross_articulator_layers, config.dropout)
            if config.cross_articulator_graph else None
        )
        self.output_dim = d

    @staticmethod
    def _kinematic_features(xy, confidence, valid, confidence_known, timestamps,
                            scale: torch.Tensor | None = None):
        b, t, n, _ = xy.shape
        dt = timestamps[:, 1:] - timestamps[:, :-1]
        pair = valid[:, 1:] & valid[:, :-1] & (dt > 0)[..., None]
        velocity = torch.zeros_like(xy)
        if t > 1:
            raw = (xy[:, 1:] - xy[:, :-1]) / dt.clamp_min(1e-5)[..., None, None]
            if scale is not None:
                raw = raw / scale[:, 1:, None, None]
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
        torso_scale = estimate_torso_scale(xy, valid) if self.config.normalize_by_shoulder_span else None
        feat = self._kinematic_features(xy, conf, valid, confidence_known, timestamps, scale=torso_scale)

        # Local coordinate anchors: hand wrist, shoulder midpoint, facial centroid.
        groups = {
            "left_hand": (LEFT, 0), "right_hand": (RIGHT, 0),
            "body": (BODY, None), "face": (FACE, None),
        }
        node_feats, group_valid_map = {}, {}
        result, availability, quality = {}, {}, {}
        encoders = {"left_hand": self.left, "right_hand": self.right, "body": self.body, "face": self.face}
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
            if torso_scale is not None:
                rel = rel / torso_scale[..., None, None]
            rel = torch.where(anchor_valid[..., None], rel, torch.zeros_like(rel))
            gfeat[..., 2:4] = rel

            enc_xy = gxy
            side_emb = None
            if self.config.share_hand_encoder and name in {"left_hand", "right_hand"}:
                side_id = 0 if name == "left_hand" else 1
                side_emb = self.hand_side_emb.weight[side_id].view(1, 1, 1, -1)
                if name == "right_hand":
                    # Mirror right hand into left-hand chirality inside the shared graph.
                    gfeat = gfeat.clone()
                    gfeat[..., 0] = 1.0 - gfeat[..., 0]
                    gfeat[..., 2] = -gfeat[..., 2]
                    gfeat[..., 4] = -gfeat[..., 4]
                    gfeat[..., 6] = -gfeat[..., 6]
                    enc_xy = gxy.clone()
                    enc_xy[..., 0] = 1.0 - enc_xy[..., 0]

            enc = encoders[name]
            if self.cross_graph is not None and hasattr(enc, "forward_nodes"):
                nodes, m = enc.forward_nodes(gfeat, enc_xy, gvalid, scale=torso_scale, extra_emb=side_emb)
                node_feats[name] = nodes
                group_valid_map[name] = (enc, m)
            elif hasattr(enc, "forward_nodes"):
                encoded, avail = enc(gfeat, enc_xy, gvalid, scale=torso_scale, extra_emb=side_emb)
                result[name] = encoded
                availability[name] = avail & frame_mask
            else:
                encoded, avail = enc(gfeat, enc_xy, gvalid)
                result[name] = encoded
                availability[name] = avail & frame_mask

            coverage = gvalid.float().mean(-1)
            conf_mean = (gconf * gvalid).sum(-1) / gvalid.sum(-1).clamp_min(1)
            known = confidence_known[:, :, sl].float().mean(-1)
            quality[name] = torch.stack([coverage, conf_mean, known, gvalid.any(-1).float(), frame_mask.float()], dim=-1)

        if self.cross_graph is not None:
            cross_scale = torso_scale if torso_scale is not None else xy.new_ones((b, t))
            updated = self.cross_graph(node_feats, xy, valid, cross_scale)
            for name, (enc, m) in group_valid_map.items():
                n_nodes = updated[name].shape[2]
                pooled, avail = enc.pool(updated[name].reshape(b * t, n_nodes, -1), m.reshape(b * t, n_nodes))
                result[name] = pooled.reshape(b, t, -1)
                availability[name] = avail.reshape(b, t) & frame_mask
                quality[name][..., 3] = avail.reshape(b, t).float()
        return {"features": result, "available": availability, "quality": quality}
