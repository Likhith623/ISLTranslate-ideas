"""Sign-aware global-to-local self-distillation pretraining.

This implements an ISL-MEGA student/teacher variant inspired by SignDINO's
teacher-global / student-local hand-face principle; it is not a reproduction
of the published SignDINO recipe or checkpoint.
"""

from __future__ import annotations

import copy
import torch
from torch import nn
import torch.nn.functional as F

from .vision import VisualBackbone
from .config import VisionConfig


class DINOProjectionHead(nn.Module):
    def __init__(self, dim: int, hidden: int = 2048, bottleneck: int = 256, prototypes: int = 4096):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(),
                                 nn.Linear(hidden, bottleneck))
        # A plain linear prototype head avoids legacy weight_norm's non-leaf
        # parameter deepcopy issue and is compatible with EMA teacher cloning.
        self.last = nn.Linear(bottleneck, prototypes, bias=False)

    def forward(self, x):
        return F.linear(F.normalize(self.mlp(x), dim=-1), F.normalize(self.last.weight, dim=-1))


class SignAwareDistillation(nn.Module):
    """Teacher sees a global frame; student sees local face/hand crops."""

    def __init__(self, config: VisionConfig, prototypes: int = 65536,
                 student_temperature: float = 0.1, teacher_temperature: float = 0.04,
                 center_momentum: float = 0.9):
        super().__init__()
        self.student = VisualBackbone(config)
        self.teacher = copy.deepcopy(self.student)
        self.student_head = DINOProjectionHead(config.dim, prototypes=prototypes)
        self.teacher_head = copy.deepcopy(self.student_head)
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)
        self.register_buffer("center", torch.zeros(1, prototypes))
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.center_momentum = center_momentum

    def train(self, mode: bool = True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_head.eval()
        return self

    def forward(self, global_frames: torch.Tensor, local_crops: torch.Tensor,
                local_valid: torch.Tensor | None = None):
        if global_frames.ndim != 4 or local_crops.ndim != 5:
            raise ValueError("global_frames [B,3,H,W], local_crops [B,K,3,H,W]")
        b, k, c, h, w = local_crops.shape
        if local_valid is None:
            local_valid = torch.ones((b, k), dtype=torch.bool, device=local_crops.device)
        if local_valid.shape != (b, k):
            raise ValueError("local_valid must have shape [B,K]")
        local = local_crops.reshape(b * k, c, h, w)
        keep = local_valid.reshape(-1).bool()
        if not keep.any():
            return local_crops.sum() * 0.0, global_frames.new_zeros((1, self.center.shape[-1]))
        student_logits = self.student_head(self.student(local[keep]))
        with torch.no_grad():
            teacher_logits = self.teacher_head(self.teacher(global_frames))
            teacher_prob = F.softmax(
                (teacher_logits.float() - self.center.float()) / self.teacher_temperature, dim=-1
            )
        owner = torch.arange(b, device=local.device)[:, None].expand(b, k).reshape(-1)[keep]
        targets = teacher_prob.index_select(0, owner)
        student_logp = F.log_softmax(student_logits.float() / self.student_temperature, dim=-1)
        loss = -(targets * student_logp).sum(-1).mean()
        return loss, teacher_logits.detach().float().mean(0, keepdim=True)

    @torch.no_grad()
    def update_teacher(self, momentum: float = 0.996):
        if not 0.0 <= momentum < 1.0:
            raise ValueError("EMA momentum must be in [0,1)")
        for student, teacher in zip(self.student.parameters(), self.teacher.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)
        for student, teacher in zip(self.student_head.parameters(), self.teacher_head.parameters()):
            teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)
        # Keep running/statistical buffers synchronized for backbones that use them.
        for student, teacher in zip(self.student.buffers(), self.teacher.buffers()):
            if teacher.is_floating_point():
                teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)
            else:
                teacher.copy_(student)
        for student, teacher in zip(self.student_head.buffers(), self.teacher_head.buffers()):
            if teacher.is_floating_point():
                teacher.mul_(momentum).add_(student, alpha=1.0 - momentum)
            else:
                teacher.copy_(student)

    @torch.no_grad()
    def update_center(self, batch_center: torch.Tensor):
        self.center.mul_(self.center_momentum).add_(
            batch_center.to(device=self.center.device, dtype=self.center.dtype),
            alpha=1.0 - self.center_momentum,
        )


def load_ssl_backbone(model, checkpoint_path: str, strict: bool = True):
    """Initialize every appearance stream from the saved sign-adapted backbone."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint.get("student_backbone", checkpoint.get("student", checkpoint))
    targets = [model.appearance.hand]
    if model.appearance.face is not model.appearance.hand:
        targets.append(model.appearance.face)
    if model.appearance.body is not model.appearance.hand:
        targets.append(model.appearance.body)
    return [target.load_state_dict(state, strict=strict) for target in targets]
