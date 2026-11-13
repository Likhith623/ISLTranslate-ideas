"""
Deformable Keypoint-to-Patch Binding Module.

Binds continuous spatial visual patch representations directly to articulated
skeleton landmarks via bilinear grid sampling and learned 2D coordinate offsets.
Generates 66 Joint-Visual Tokens: [B, T, 66, D].
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
    class DeformablePatchBinder(nn.Module):
        """
        Binds 2D/3D skeletal keypoints to local DINOv2 visual patch representations.
        
        Args:
            visual_dim: Channel dimension of spatial patches (e.g. 384 for DINOv2-ViT-S).
            pose_dim: Coordinate channels per landmark (typically 3: x, y, z or confidence).
            hidden_dim: Unified joint embedding dimension (e.g. 256).
            num_joints: Number of canonical landmarks (66).
            max_offset: Maximum deformable continuous offset in normalized grid units.
            dropout: Dropout probability.
            modality_dropout_prob: Probability of independently dropping vision or pose.
        """
        def __init__(
            self,
            visual_dim: int = 384,
            pose_dim: int = 3,
            hidden_dim: int = 256,
            num_joints: int = 66,
            max_offset: float = 0.15,
            dropout: float = 0.1,
            modality_dropout_prob: float = 0.15,
        ):
            super().__init__()
            self.visual_dim = visual_dim
            self.pose_dim = pose_dim
            self.hidden_dim = hidden_dim
            self.num_joints = num_joints
            self.max_offset = max_offset
            self.modality_dropout_prob = modality_dropout_prob

            # Deformable offset predictor: takes pose coords -> outputs (dx, dy)
            self.offset_net = nn.Sequential(
                nn.Linear(pose_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 2),
                nn.Tanh(),
            )

            # Projections to unified embedding space
            self.vis_proj = nn.Linear(visual_dim, hidden_dim)
            self.pose_proj = nn.Linear(pose_dim, hidden_dim)
            
            # Learned landmark identity embeddings: 66 canonical joints
            self.joint_embed = nn.Parameter(torch.randn(1, 1, num_joints, hidden_dim) * 0.02)
            
            self.norm = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def sample_visual_patches(
            self,
            patch_grid: torch.Tensor,
            sample_coords: torch.Tensor,
        ) -> torch.Tensor:
            """
            Bilinear sampling of patch grid at continuous 2D coordinates.
            
            Args:
                patch_grid: (B, T, H_grid, W_grid, C_vis)
                sample_coords: (B, T, V, 2) normalized to [-1, 1]
            Returns:
                sampled_features: (B, T, V, C_vis)
            """
            B, T, Hg, Wg, C = patch_grid.shape
            V = sample_coords.shape[2]
            
            # Flatten B and T for grid_sample:
            # grid_sample expects input (N, C, H_in, W_in) and grid (N, H_out, W_out, 2)
            flat_patches = patch_grid.view(B * T, Hg, Wg, C).permute(0, 3, 1, 2)  # (B*T, C, Hg, Wg)
            
            # Reshape coords to (B*T, 1, V, 2)
            flat_coords = sample_coords.view(B * T, 1, V, 2)
            
            sampled = F.grid_sample(
                flat_patches,
                flat_coords,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )  # (B*T, C, 1, V)
            
            # Reshape back to (B, T, V, C)
            sampled = sampled.squeeze(2).permute(0, 2, 1)  # (B*T, V, C)
            return sampled.view(B, T, V, C)

        def forward(
            self,
            patch_grid: torch.Tensor,
            landmarks_norm: torch.Tensor,
            landmarks_raw_xy: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Args:
                patch_grid: (B, T, H_grid, W_grid, C_vis) from DINOv2.
                landmarks_norm: (B, T, 66, 3) torso-normalized keypoints.
                landmarks_raw_xy: Optional (B, T, 66, 2) in normalized image coordinates [0, 1].
                                  If None, estimates from xy of normalized keypoints.
            Returns:
                joint_tokens: (B, T, 66, D)
                offsets: (B, T, 66, 2) predicted deformable offsets
            """
            B, T, V, C = landmarks_norm.shape
            assert V == self.num_joints, f"Expected {self.num_joints} landmarks, got {V}"
            
            # Predict 2D continuous offsets
            offsets = self.offset_net(landmarks_norm) * self.max_offset  # (B, T, V, 2)
            
            # Base sampling coordinates in [-1, 1] range
            if landmarks_raw_xy is not None:
                base_grid = (landmarks_raw_xy[:, :, :, :2] * 2.0) - 1.0
            else:
                # Clamp normalized coords to [-1, 1] range as spatial query
                base_grid = torch.clamp(landmarks_norm[:, :, :, :2], -1.0, 1.0)
                
            sample_coords = torch.clamp(base_grid + offsets, -1.0, 1.0)
            
            # Sample continuous features from visual patch grid
            sampled_vis = self.sample_visual_patches(patch_grid, sample_coords)  # (B, T, V, C_vis)
            
            # Linear projections
            h_vis = self.vis_proj(sampled_vis)        # (B, T, V, D)
            h_pose = self.pose_proj(landmarks_norm)    # (B, T, V, D)
            
            # Modality Dropout during training to avoid single-modality collapse
            if self.training and self.modality_dropout_prob > 0.0:
                rand_val = torch.rand(1, device=patch_grid.device).item()
                if rand_val < self.modality_dropout_prob / 2.0:
                    # Drop vision: rely purely on pose
                    h_vis = torch.zeros_like(h_vis)
                elif rand_val < self.modality_dropout_prob:
                    # Drop pose: rely purely on vision
                    h_pose = torch.zeros_like(h_pose)
                    
            # Fuse with identity embedding
            joint_tokens = self.norm(h_vis + h_pose + self.joint_embed)
            joint_tokens = self.dropout(joint_tokens)
            
            return joint_tokens, offsets
else:
    class DeformablePatchBinder:  # type: ignore
        def __init__(self, *args, **kwargs): pass

