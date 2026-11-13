"""Anatomically anchored SO(3)/translation/positive-scale canonicalization."""
import torch
from torch import Tensor, nn


class BodyFrame(nn.Module):
    """Use the first reliable nondegenerate anchor frame; preserve torso motion.

    Coordinates must share one Euclidean 3D coordinate system. Reflections,
    perspective transformations, moving cameras and non-rigid scale are excluded.
    """

    def __init__(self, anchors=(0, 1, 2), min_confidence=0.5, policy="first_reliable"):
        super().__init__()
        self.anchors = tuple(anchors)
        self.min_confidence = min_confidence
        if policy not in {"first", "first_reliable"}:
            raise ValueError("unknown anchor policy")
        self.policy = policy

    def eligible_frames(self, pose: Tensor, confidence: Tensor, lengths: Tensor | None = None) -> Tensor:
        if pose.ndim != 4 or pose.shape[-1] != 3 or confidence.shape != pose.shape[:-1]:
            raise ValueError("expected pose [B,T,J,3] and confidence [B,T,J]")
        a = pose[:, :, self.anchors, :]
        c = confidence[:, :, self.anchors]
        finite = torch.isfinite(a).all(dim=(-1, -2)) & torch.isfinite(c).all(-1)
        clean = torch.where(torch.isfinite(a), a, 0)
        left, right, neck = clean.unbind(2)
        shoulder, up = right - left, neck - (left + right) / 2
        width = torch.linalg.vector_norm(shoulder, dim=-1)
        relative_height = torch.linalg.vector_norm(torch.linalg.cross(shoulder, up, dim=-1), dim=-1)
        valid = finite & (c >= self.min_confidence).all(-1) & (width > 1e-8)
        valid &= relative_height > 1e-5 * width.square()
        if lengths is not None:
            valid &= torch.arange(pose.shape[1], device=pose.device)[None] < lengths[:, None]
        if self.policy == "first":
            valid[:, 1:] = False
        return valid

    def reference_indices(self, pose: Tensor, confidence: Tensor, lengths: Tensor | None = None) -> Tensor:
        valid = self.eligible_frames(pose, confidence, lengths)
        if not valid.any(1).all():
            raise ValueError("no reliable nondegenerate body frame: missing anchors or collinear geometry")
        return valid.to(torch.int64).argmax(1)

    def forward(self, pose: Tensor, confidence: Tensor, lengths: Tensor | None = None) -> Tensor:
        index = self.reference_indices(pose, confidence, lengths)
        a = pose[torch.arange(pose.shape[0], device=pose.device), index][:, self.anchors]
        left, right, neck = a.unbind(1)
        origin = (left + right) / 2
        shoulder = right - left
        width = torch.linalg.vector_norm(shoulder, dim=-1, keepdim=True)
        if (width <= 1e-8).any():
            raise ValueError("degenerate shoulder baseline")
        e1 = shoulder / width
        up = neck - origin
        up = up - (up * e1).sum(-1, keepdim=True) * e1
        height = torch.linalg.vector_norm(up, dim=-1, keepdim=True)
        if (height / width <= 1e-5).any():
            raise ValueError("neck collinear with shoulders")
        e2 = up / height
        e3 = torch.linalg.cross(e1, e2, dim=-1)
        basis = torch.stack((e1, e2, e3), dim=-1)
        centered = (pose - origin[:, None, None, :]) / width[:, None, None, :]
        return torch.einsum("btjc,bcd->btjd", centered, basis)
