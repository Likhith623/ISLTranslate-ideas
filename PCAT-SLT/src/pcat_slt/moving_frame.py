"""Time-aware local body coordinates; global motion is retained by a separate path."""

import torch
from torch import nn


class MovingBodyFrame(nn.Module):
    def __init__(self, anchors, threshold=0.5, time_constant=0.25):
        super().__init__()
        self.anchors, self.threshold, self.tau = anchors, threshold, time_constant

    def forward(self, global_features, lengths, timestamps):
        if timestamps is None:
            raise ValueError("dual body frames require physical timestamps")
        p, c = global_features[..., :3], global_features[..., 3]
        b, t = p.shape[:2]
        origin = p.new_zeros(b, 3)
        e1 = p.new_tensor([1.0, 0.0, 0.0]).expand(b, -1)
        e2 = p.new_tensor([0.0, 1.0, 0.0]).expand(b, -1)
        width = p.new_ones(b, 1)
        coords, motions = [], []
        for i in range(t):
            left, right, neck = p[:, i, self.anchors].unbind(1)
            o = (left + right) / 2
            shoulder = right - left
            w = shoulder.norm(dim=-1, keepdim=True)
            x = shoulder / w.clamp_min(1e-8)
            up = neck - o
            up = up - (up * x).sum(-1, keepdim=True) * x
            h = up.norm(dim=-1, keepdim=True)
            reliable = (
                (c[:, i, self.anchors] >= self.threshold).all(-1)
                & (i < lengths)
                & (w[:, 0] > 1e-8)
                & (h[:, 0] > 1e-5 * w[:, 0])
            )
            if i:
                dt = (timestamps[:, i] - timestamps[:, i - 1]).to(p).clamp_min(0)
                weight = -torch.expm1(-dt / self.tau)
            else:
                weight = p.new_ones(b)
            weight = torch.where(reliable, weight, 0)[:, None]
            origin = origin + weight * (o - origin)
            width = width + weight * (w - width)
            candidate_x = e1 + weight * (x - e1)
            candidate_x = candidate_x / candidate_x.norm(dim=-1, keepdim=True).clamp_min(1e-8)
            y = up / h.clamp_min(1e-8)
            candidate_y = e2 + weight * (y - e2)
            candidate_y = candidate_y - (candidate_y * candidate_x).sum(-1, keepdim=True) * candidate_x
            # Revert on antipodal/degenerate interpolated axes instead of using a zero frame.
            ok = (candidate_y.norm(dim=-1, keepdim=True) > 1e-6) & (
                candidate_x.norm(dim=-1, keepdim=True) > 0.5
            )
            e1 = torch.where(ok, candidate_x, e1)
            e2 = torch.where(ok, candidate_y / candidate_y.norm(dim=-1, keepdim=True).clamp_min(1e-8), e2)
            basis = torch.stack((e1, e2, torch.linalg.cross(e1, e2, dim=-1)), -1)
            local = torch.einsum("bjc,bcd->bjd", (p[:, i] - origin[:, None]) / width[:, None], basis)
            coords.append(local)
            motions.append(torch.cat((origin, e1, e2, width.log(), reliable[:, None].to(p)), -1))
        valid = torch.arange(t, device=p.device)[None] < lengths[:, None]
        local = torch.stack(coords, 1)
        motion = torch.stack(motions, 1)
        return torch.where(valid[..., None, None], local, 0), torch.where(valid[..., None], motion, 0)
