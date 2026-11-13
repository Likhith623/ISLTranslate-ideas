"""
Kinematic Dominance Routing and Dynamic Torso-Midline Normalization.

This module implements two foundational phonological principles of sign language:
1. Dynamic Torso-Midline Centering:
   Normalizes 3D keypoints with respect to the signer's dynamic shoulder midpoint
   p_mid(t) = (p_42(t) + p_43(t)) / 2 and EMA-smoothed shoulder width w_bar(t).
2. Kinematic Hand Dominance Routing:
   Evaluates kinetic energy E = sum_t ||v(t)||^2 of Left vs. Right hand trajectories.
   Maps the active hand to 'Dominant Hand' (DH) and the reference hand to 'Base Hand' (BH).
   Appends learned role embeddings to eliminate left/right signer variance while
   preserving asymmetric lexical morphology.
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


# ---------------------------------------------------------------------------
# Pure Python / NumPy Mathematical Reference Implementations
# ---------------------------------------------------------------------------

def compute_torso_midline_pure(
    landmarks_seq: List[List[List[float]]],
    left_shoulder_idx: int = 42,
    right_shoulder_idx: int = 43,
    ema_momentum: float = 0.95,
    eps: float = 1e-6,
) -> Tuple[List[List[List[float]]], List[float]]:
    """
    Torso-midline normalization in pure Python for reference and zero-dependency verification.
    
    Args:
        landmarks_seq: [T, 66, C] list of frames.
        left_shoulder_idx: Canonical index of left shoulder (42).
        right_shoulder_idx: Canonical index of right shoulder (43).
        ema_momentum: Smoothing factor for shoulder width.
        eps: Epsilon to prevent zero division.
    Returns:
        normalized_seq: [T, 66, C] list of normalized frames.
        smoothed_widths: [T] list of shoulder widths.
    """
    T = len(landmarks_seq)
    if T == 0:
        return [], []
    
    # First pass: compute midpoints and instantaneous shoulder widths
    midpoints = []
    widths = []
    for t in range(T):
        p_ls = landmarks_seq[t][left_shoulder_idx]
        p_rs = landmarks_seq[t][right_shoulder_idx]
        mid = [(a + b) / 2.0 for a, b in zip(p_ls, p_rs)]
        midpoints.append(mid)
        # 2D/3D Euclidean distance
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(p_ls[:2], p_rs[:2])))
        widths.append(max(dist, eps))
        
    # Temporal smoothing of widths (EMA)
    smoothed_widths = [widths[0]]
    for t in range(1, T):
        sm = ema_momentum * smoothed_widths[-1] + (1.0 - ema_momentum) * widths[t]
        smoothed_widths.append(max(sm, eps))
        
    # Second pass: normalize
    normalized_seq = []
    for t in range(T):
        w = smoothed_widths[t]
        mid = midpoints[t]
        frame_norm = []
        for pt in landmarks_seq[t]:
            # Subtract midpoint, scale by shoulder width
            pt_norm = [(coord - m) / w for coord, m in zip(pt, mid)]
            frame_norm.append(pt_norm)
        normalized_seq.append(frame_norm)
        
    return normalized_seq, smoothed_widths


def compute_hand_energies_pure(
    landmarks_seq: List[List[List[float]]],
    left_hand_slice: Tuple[int, int] = (0, 21),
    right_hand_slice: Tuple[int, int] = (21, 42),
) -> Tuple[float, float, str]:
    """
    Compute kinetic energy for Left and Right hands across sequence.
    
    Returns:
        (e_left, e_right, dominant_hand) where dominant_hand is 'right' or 'left'.
    """
    T = len(landmarks_seq)
    if T < 2:
        return 0.0, 0.0, "right"
        
    e_left = 0.0
    e_right = 0.0
    
    l_start, l_end = left_hand_slice
    r_start, r_end = right_hand_slice
    
    for t in range(1, T):
        # Left hand velocity energy
        for j in range(l_start, l_end):
            p_curr = landmarks_seq[t][j]
            p_prev = landmarks_seq[t - 1][j]
            v_sq = sum((c - p) ** 2 for c, p in zip(p_curr[:3], p_prev[:3]))
            e_left += v_sq
            
        # Right hand velocity energy
        for j in range(r_start, r_end):
            p_curr = landmarks_seq[t][j]
            p_prev = landmarks_seq[t - 1][j]
            v_sq = sum((c - p) ** 2 for c, p in zip(p_curr[:3], p_prev[:3]))
            e_right += v_sq
            
    dominant = "right" if e_right >= e_left else "left"
    return e_left, e_right, dominant


# ---------------------------------------------------------------------------
# PyTorch Production Module
# ---------------------------------------------------------------------------

if _HAS_TORCH:
    class TorsoMidlineNormalizer(nn.Module):
        """
        Differentiable, batched Torso-Midline Normalization layer.
        Normalizes (B, T, 66, C) coordinates relative to dynamic shoulder midpoint.
        """
        def __init__(
            self,
            left_shoulder_idx: int = 42,
            right_shoulder_idx: int = 43,
            ema_momentum: float = 0.95,
            eps: float = 1e-6,
        ):
            super().__init__()
            self.left_shoulder_idx = left_shoulder_idx
            self.right_shoulder_idx = right_shoulder_idx
            self.ema_momentum = ema_momentum
            self.eps = eps

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                x: Tensor of shape (B, T, 66, C) or (T, 66, C).
            Returns:
                norm_x: Normalized coordinates of same shape.
                scale: Shoulder width scale of shape (B, T, 1, 1).
            """
            is_unbatched = x.dim() == 3
            if is_unbatched:
                x = x.unsqueeze(0)
                
            B, T, V, C = x.shape
            
            p_ls = x[:, :, self.left_shoulder_idx, :]  # (B, T, C)
            p_rs = x[:, :, self.right_shoulder_idx, :]  # (B, T, C)
            
            midpoint = (p_ls + p_rs) / 2.0  # (B, T, C)
            midpoint = midpoint.unsqueeze(2)  # (B, T, 1, C)
            
            # Instantaneous shoulder distance in xy plane
            diff = p_ls[:, :, :2] - p_rs[:, :, :2]
            inst_width = torch.linalg.norm(diff, dim=-1, keepdim=True).clamp(min=self.eps)  # (B, T, 1)
            
            # Smooth width across time using causal filter or median
            # Using causal EMA smoothing:
            smoothed = []
            curr = inst_width[:, 0, :]
            for t in range(T):
                curr = self.ema_momentum * curr + (1.0 - self.ema_momentum) * inst_width[:, t, :]
                smoothed.append(curr)
            width = torch.stack(smoothed, dim=1).unsqueeze(-1)  # (B, T, 1, 1)
            
            norm_x = (x - midpoint) / width
            
            if is_unbatched:
                norm_x = norm_x.squeeze(0)
                width = width.squeeze(0)
                
            return norm_x, width


    class KinematicDominanceRouter(nn.Module):
        """
        Dynamically partitions Left and Right hand keypoints into Dominant Hand (DH)
        and Base Hand (BH) based on utterance-level kinetic energy.
        Appends learned role embeddings to disambiguate active vs reference articulator.
        """
        def __init__(
            self,
            embed_dim: int = 256,
            left_hand_slice: Tuple[int, int] = (0, 21),
            right_hand_slice: Tuple[int, int] = (21, 42),
            use_role_embeddings: bool = True,
        ):
            super().__init__()
            self.embed_dim = embed_dim
            self.left_slice = left_hand_slice
            self.right_slice = right_hand_slice
            self.use_role_embeddings = use_role_embeddings
            
            if use_role_embeddings:
                # Learned role embeddings for dominant and base hands
                self.dominant_role_embed = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
                self.base_role_embed = nn.Parameter(torch.randn(1, 1, 1, embed_dim) * 0.02)
            else:
                self.dominant_role_embed = None
                self.base_role_embed = None

        def compute_energy(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Compute total kinetic energy of Left and Right hands.
            Args:
                x: (B, T, 66, C) normalized keypoint tensor.
            Returns:
                e_left, e_right: (B,) energy per batch element.
            """
            B, T, V, C = x.shape
            if T < 2:
                zeros = torch.zeros(B, device=x.device, dtype=x.dtype)
                return zeros, zeros
                
            v = x[:, 1:, :, :3] - x[:, :-1, :, :3]  # (B, T-1, 66, 3)
            v_sq = torch.sum(v ** 2, dim=-1)         # (B, T-1, 66)
            
            e_left = v_sq[:, :, self.left_slice[0]:self.left_slice[1]].sum(dim=(1, 2))
            e_right = v_sq[:, :, self.right_slice[0]:self.right_slice[1]].sum(dim=(1, 2))
            
            return e_left, e_right

        def route_keypoints(
            self, x: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Route keypoints into (B, T, 21, C) dominant and base hands.
            
            Returns:
                dh_pts: Dominant hand keypoints (B, T, 21, C)
                bh_pts: Base hand keypoints (B, T, 21, C)
                is_right_dominant: Boolean mask (B,) indicating if right hand is dominant
            """
            B, T, V, C = x.shape
            e_left, e_right = self.compute_energy(x)
            
            # Decision: right is dominant if e_right >= e_left
            is_right_dom = (e_right >= e_left)  # (B,)
            
            lh = x[:, :, self.left_slice[0]:self.left_slice[1], :]   # (B, T, 21, C)
            rh = x[:, :, self.right_slice[0]:self.right_slice[1], :]  # (B, T, 21, C)
            
            # Select per batch element
            mask = is_right_dom.view(B, 1, 1, 1).expand(-1, T, 21, C)
            dh_pts = torch.where(mask, rh, lh)
            bh_pts = torch.where(mask, lh, rh)
            
            return dh_pts, bh_pts, is_right_dom

        def apply_role_embeddings(
            self, dh_tokens: torch.Tensor, bh_tokens: torch.Tensor
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Add learned role embeddings to dominant and base hand feature tokens.
            """
            if self.use_role_embeddings and self.dominant_role_embed is not None:
                dh_tokens = dh_tokens + self.dominant_role_embed
                bh_tokens = bh_tokens + self.base_role_embed
            return dh_tokens, bh_tokens
else:
    # Fallback dummy class signatures if torch is absent
    class TorsoMidlineNormalizer:  # type: ignore
        def __init__(self, *args, **kwargs): pass
    class KinematicDominanceRouter:  # type: ignore
        def __init__(self, *args, **kwargs): pass

