"""Exact clipped polygonal signed areas and norm-preserving area transport.

No custom extensions or approximate numerical integration are required.
See ARCHITECTURE.md for the assumptions behind the invariants.
"""
from dataclasses import dataclass
import torch
from torch import Tensor, nn


@dataclass
class PathStatistics:
    start: Tensor             # [B,K,N]
    delta: Tensor             # [B,K,N]
    area: Tensor              # [B,K,N,N], skew-symmetric
    length: Tensor            # [B,1]
    timing: Tensor            # [B,K,2]: log1p elapsed seconds, fraction of duration


def relative_times(timestamps, like, valid):
    """Rebase before casting: epoch-valued FP64 times must not collapse in FP32."""
    if timestamps is None:
        return torch.arange(valid.shape[1], device=like.device, dtype=like.dtype).expand(valid.shape)
    if timestamps.shape != valid.shape:
        raise ValueError("timestamps must have shape [B,T]")
    source_valid = valid.to(timestamps.device)
    if not torch.isfinite(timestamps[source_valid]).all():
        raise ValueError("nonfinite timestamps")
    origin = timestamps[:, :1]
    clean = torch.where(source_valid, timestamps, origin)
    return (clean - origin).to(like)


def cayley(area: Tensor, gain: Tensor) -> Tensor:
    """Return (I - A)^-1(I + A), A = gain*H/(1+||H||_F)."""
    norm = torch.linalg.matrix_norm(area, ord="fro", keepdim=True)
    skew = gain * area / (1 + norm)
    eye = torch.eye(area.shape[-1], dtype=area.dtype, device=area.device)
    return torch.linalg.solve(eye - skew, eye + skew)


def cayley_action(area: Tensor, gain: Tensor, values: Tensor) -> Tensor:
    """Exact action with only the required right-hand sides, without forming Q."""
    skew = gain * area / (1 + torch.linalg.matrix_norm(area, keepdim=True))
    eye = torch.eye(area.shape[-1], dtype=area.dtype, device=area.device)
    return torch.linalg.solve(eye - skew, values + skew @ values)


def givens_transport(area: Tensor, gain: Tensor, values: Tensor) -> Tensor:
    """Solve-free orthogonal action; values [...,N,R]. Palindromic sweeps
    preserve reversal: G(-H)=G(H)^T. This is not a Cayley approximation.
    """
    n = area.shape[-1]
    bounded = gain * area / (1 + torch.linalg.matrix_norm(area, keepdim=True))
    x = values + bounded.sum(dim=(-1, -2), keepdim=True) * 0
    for parity, fraction in ((0, .5), (1, 1.), (0, .5)):
        if parity >= n - 1:
            continue
        left = torch.arange(parity, n - 1, 2, device=area.device)
        right = left + 1
        if not left.numel():
            continue
        angle = fraction * 2 * torch.atan(bounded[..., left, right])
        c, s = angle.cos()[..., None], angle.sin()[..., None]
        u, v = x.index_select(-2, left), x.index_select(-2, right)
        x = x.index_copy(-2, left, c * u + s * v)
        x = x.index_copy(-2, right, -s * u + c * v)
    return x


def polygonal_statistics_reference(z: Tensor, lengths: Tensor, bins: int, metric: Tensor,
                         timestamps: Tensor | None = None, temporal: bool = False,
                         augment_progress: bool = False, edge_lengths: Tensor | None = None) -> PathStatistics:
    """Integrate each segment clipped against common progress cells.

    Invariant to insertion of collinear vertices, including duplicates, with
    temporal=False. No epsilon is added to nonzero segment lengths.
    """
    if z.ndim != 3 or z.shape[1] < 2 or bins < 1 or metric.shape != (z.shape[-1],):
        raise ValueError("expected z [B,T>=2,N], positive bins, metric [N]")
    b, t, _ = z.shape
    if lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an integer vector [B]")
    if ((lengths < 2) | (lengths > t)).any() or (metric <= 0).any():
        raise ValueError("invalid lengths or nonpositive metric")
    valid = torch.arange(t, device=z.device)[None] < lengths[:, None]
    if not torch.isfinite(z[valid]).all() or not torch.isfinite(metric).all():
        raise ValueError("nonfinite path or metric")
    z = torch.where(valid[..., None], z, 0)
    edge_valid = valid[:, 1:]
    d = torch.where(edge_valid[..., None], z[:, 1:] - z[:, :-1], 0)
    ell = torch.linalg.vector_norm(d * metric.sqrt(), dim=-1) if edge_lengths is None else edge_lengths
    if ell.shape != d.shape[:2] or not torch.isfinite(ell).all() or (ell < 0).any():
        raise ValueError("invalid shared edge lengths")
    ell = torch.where(edge_valid, ell, 0)
    path_length = ell.sum(-1, keepdim=True)
    if augment_progress:
        rho = ell / torch.where(path_length > 0, path_length, 1)
        progress = torch.cat((z.new_zeros(b, 1), rho.cumsum(1)), 1)
        z = torch.cat((z, progress[..., None]), -1)
        d = torch.cat((d, rho[..., None]), -1)
    timestamps = relative_times(timestamps, z, valid)
    if timestamps.shape != (b, t):
        raise ValueError("timestamps must have shape [B,T]")
    if not torch.isfinite(timestamps[valid]).all():
        raise ValueError("nonfinite timestamps")
    dt = torch.where(edge_valid, timestamps[:, 1:] - timestamps[:, :-1], 0)
    if (dt[edge_valid] <= 0).any():
        raise ValueError("timestamps must strictly increase on each valid prefix")
    # Temporal windows are an explicitly non-invariant matched control.
    cell_length = dt if temporal else ell
    total = cell_length.sum(-1, keepdim=True)
    end = cell_length.cumsum(-1)
    begin = end - cell_length
    positive = cell_length > 0
    safe = torch.where(positive, cell_length, torch.ones_like(cell_length))
    starts, deltas, areas, elapsed = [], [], [], []
    # Streaming over K keeps peak working memory O(B*T*N + B*K*N^2), not O(B*T*K*N^2).
    for k in range(bins):
        lo, hi = total * (k / bins), total * ((k + 1) / bins)
        overlap = (torch.minimum(end, hi) - torch.maximum(begin, lo)).clamp_min(0)
        fraction = torch.where(positive, overlap / safe, 0)
        v = fraction[..., None] * d
        prefix = v.cumsum(1) - v
        cross = torch.einsum("bti,btj->bij", prefix, v)
        h = 0.5 * (cross - cross.transpose(-1, -2))
        before = torch.where(positive, ((lo - begin) / safe).clamp(0, 1), 0)
        start = z[:, 0] + (before[..., None] * d).sum(1)
        # A hold carries no path length. Attribute its physical duration to the
        # cell immediately following its progress position (last cell at L).
        hold_cell = ((begin / torch.where(total > 0, total, 1)) * bins).floor().long()
        hold_cell = hold_cell.clamp(0, bins - 1)
        duration_weight = torch.where(positive, fraction, (hold_cell == k).to(z.dtype))
        seconds = (duration_weight * dt).sum(-1)
        elapsed.append(seconds)
        starts.append(start)
        deltas.append(v.sum(1))
        areas.append(h)
    elapsed = torch.stack(elapsed, 1)
    total_time = dt.sum(-1, keepdim=True)
    timing = torch.stack((torch.log1p(elapsed), elapsed / total_time), dim=-1)
    return PathStatistics(torch.stack(starts, 1), torch.stack(deltas, 1),
                          torch.stack(areas, 1), path_length, timing)


def pyramid_statistics(z: Tensor, lengths: Tensor, scales: tuple[int, ...], metric: Tensor,
                       timestamps: Tensor | None = None, temporal: bool = False,
                       augment_progress: bool = False, edge_lengths: Tensor | None = None) -> PathStatistics:
    """Exact prefix-signature interval extraction; preprocess the path only once.

    Cost O(B*T*N^2 + B*sum(scales)*N^2), excluding small solves. Prefix
    subtraction can lose precision on long paths; the reference integrator
    remains available as an independent numerical oracle.
    """
    if z.ndim != 3 or z.shape[1] < 2 or metric.shape != (z.shape[-1],):
        raise ValueError("expected z [B,T>=2,N] and metric [N]")
    if not scales or any(type(k) is not int or k < 1 for k in scales):
        raise ValueError("scales must contain positive integer cell counts")
    b, t, _ = z.shape
    if lengths.shape != (b,) or lengths.dtype not in (torch.int32, torch.int64):
        raise ValueError("lengths must be an integer vector [B]")
    if ((lengths < 2) | (lengths > t)).any() or (metric <= 0).any():
        raise ValueError("invalid lengths or nonpositive metric")
    valid = torch.arange(t, device=z.device)[None] < lengths[:, None]
    if not torch.isfinite(z[valid]).all() or not torch.isfinite(metric).all():
        raise ValueError("nonfinite path or metric")
    z = torch.where(valid[..., None], z, 0)
    edges = valid[:, 1:]
    d = torch.where(edges[..., None], z[:, 1:] - z[:, :-1], 0)
    ell = torch.linalg.vector_norm(d * metric.sqrt(), dim=-1) if edge_lengths is None else edge_lengths
    if ell.shape != d.shape[:2] or not torch.isfinite(ell).all() or (ell < 0).any():
        raise ValueError("invalid shared edge lengths")
    ell = torch.where(edges, ell, 0)
    length = ell.sum(-1, keepdim=True)
    timestamps = relative_times(timestamps, z, valid)
    if timestamps.shape != (b, t) or not torch.isfinite(timestamps[valid]).all():
        raise ValueError("expected finite valid timestamps [B,T]")
    dt = torch.where(edges, timestamps[:, 1:] - timestamps[:, :-1], 0)
    if (dt[edges] <= 0).any():
        raise ValueError("timestamps must strictly increase on each valid prefix")
    if augment_progress:
        rho = ell / torch.where(length > 0, length, 1)
        d = torch.cat((d, rho[..., None]), dim=-1)
        initial = torch.cat((z[:, 0], z.new_zeros(b, 1)), dim=-1)
    else:
        initial = z[:, 0]
    n = d.shape[-1]
    coordinate = dt if temporal else ell
    end = coordinate.cumsum(1)
    begin = end - coordinate
    total = coordinate.sum(1, keepdim=True)
    positive = coordinate > 0
    safe = torch.where(positive, coordinate, 1)
    prefix = d.cumsum(1) - d
    cross = prefix[..., :, None] * d[..., None, :]
    edge_area = .5 * (cross - cross.transpose(-1, -2))
    area_prefix = edge_area.cumsum(1) - edge_area
    moving_time = torch.where(positive, dt, 0)
    time_prefix = moving_time.cumsum(1) - moving_time

    def boundary_values(target):
        # right=True skips zero-length runs and assigns boundaries consistently.
        index = torch.searchsorted(end.contiguous(), target.contiguous(), right=True).clamp_max(t - 2)
        edge_start = begin.gather(1, index)
        edge_length = safe.gather(1, index)
        fraction = ((target - edge_start) / edge_length).clamp(0, 1)
        fraction = torch.where(positive.gather(1, index), fraction, 0)
        vector_index = index[..., None].expand(-1, -1, n)
        p = prefix.gather(1, vector_index)
        step = fraction[..., None] * d.gather(1, vector_index)
        h = area_prefix.gather(1, index[..., None, None].expand(-1, -1, n, n))
        wedge = p[..., :, None] * step[..., None, :]
        h = h + .5 * (wedge - wedge.transpose(-1, -2))
        seconds = time_prefix.gather(1, index) + fraction * moving_time.gather(1, index)
        return p + step, h, seconds

    starts, deltas, areas, timing = [], [], [], []
    for cells in scales:
        grid = torch.linspace(0, 1, cells + 1, device=z.device, dtype=z.dtype)
        position, h, seconds = boundary_values(total * grid)
        delta = position[:, 1:] - position[:, :-1]
        correction = position[:, :-1, :, None] * delta[..., None, :]
        area = h[:, 1:] - h[:, :-1] - .5 * (correction - correction.transpose(-1, -2))
        area = .5 * (area - area.transpose(-1, -2))
        elapsed = seconds[:, 1:] - seconds[:, :-1]
        hold_cell = ((begin / torch.where(total > 0, total, 1)) * cells).floor().long()
        hold_cell = hold_cell.clamp(0, cells - 1)
        hold_time = torch.where(positive, 0, dt)
        elapsed = elapsed.scatter_add(1, hold_cell, hold_time).clamp_min(0)
        starts.append(initial[:, None] + position[:, :-1])
        deltas.append(delta)
        areas.append(area)
        timing.append(torch.stack((torch.log1p(elapsed), elapsed / dt.sum(1, keepdim=True)), -1))
    return PathStatistics(torch.cat(starts, 1), torch.cat(deltas, 1), torch.cat(areas, 1),
                          length, torch.cat(timing, 1))


def polygonal_statistics(z: Tensor, lengths: Tensor, bins: int, metric: Tensor,
                         timestamps: Tensor | None = None, temporal: bool = False) -> PathStatistics:
    return pyramid_statistics(z, lengths, (bins,), metric, timestamps, temporal)


class ProgressCoupledAreaTransport(nn.Module):
    """Atomic PCAT operator: learned common clock -> area -> Cayley transport.

    Input [B,T,N], output [B,K,width]. The internal N dimensions preserve
    articulator blocks; cross-block signed areas drive orthogonal mixing.
    """

    def __init__(self, dimensions: int, bins: int, width: int, metric_floor=0.05,
                 clock="learned", mode="pcat", timing=False, scales=None,
                 augment_progress=False, metric_condition_limit=16.0,
                 integration="prefix", integration_precision="float32", transport="cayley",
                 sketch_dim=0, sketch_heads=1, adaptive_step=0., adaptive_min_bins=2):
        super().__init__()
        self.bins, self.mode, self.timing = bins, mode, timing
        self.metric_floor = metric_floor
        self.scales = tuple(scales) if scales else (bins,)
        self.token_count = sum(self.scales)
        self.augment_progress = augment_progress
        self.metric_condition_limit = metric_condition_limit
        self.integration, self.integration_precision = integration, integration_precision
        self.transport_kind = transport
        self.adaptive_step, self.adaptive_min_bins = adaptive_step, adaptive_min_bins
        self.sketch_heads = sketch_heads
        if transport not in {"cayley", "givens"}:
            raise ValueError("unknown transport")
        if sketch_dim:
            if not 1 <= sketch_dim <= dimensions or sketch_heads < 1:
                raise ValueError("invalid area sketch")
            generator = torch.Generator().manual_seed(1729)
            projection = torch.stack([torch.linalg.qr(torch.randn(dimensions, sketch_dim, generator=generator))[0]
                                      for _ in range(sketch_heads)])
        else:
            if sketch_heads != 1:
                raise ValueError("multiple sketch heads require sketch_dim")
            projection = None
        self.register_buffer("projection", projection)
        self.token_count *= sketch_heads
        if integration not in {"prefix", "clipped"} or integration_precision not in {"float32", "float64"}:
            raise ValueError("invalid integration backend or precision")
        self.raw_metric = nn.Parameter(torch.zeros(dimensions), requires_grad=(clock == "learned"))
        self.clock = clock
        self.raw_gain = nn.Parameter(torch.tensor(0.5), requires_grad=(mode in {"pcat", "temporal"}))
        transport_dimensions = (sketch_dim or dimensions) + int(augment_progress)
        self.register_buffer("upper", torch.triu_indices(transport_dimensions, transport_dimensions, offset=1))
        # Keep dimensions fixed across ablations, so downstream parameter counts match.
        size = (3 * transport_dimensions + transport_dimensions * (transport_dimensions - 1) // 2
                + 1 + (2 if timing else 0))
        self.readout = nn.Sequential(nn.Linear(size, width), nn.GELU(), nn.LayerNorm(width))

    @property
    def metric(self):
        if self.clock == "euclidean":
            return torch.ones_like(self.raw_metric)
        # A fixed positive floor alone does not bound condition after normalization.
        # A bounded log-range does, even for arbitrarily large raw parameters.
        half_range = .5 * torch.log(self.raw_metric.new_tensor(self.metric_condition_limit))
        w = torch.exp(half_range * self.raw_metric.tanh()) + self.metric_floor
        return w / w.mean()  # fixes an otherwise unidentifiable global clock scale

    def statistics(self, z, lengths, timestamps=None, scales=None):
        scales = self.scales if scales is None else scales
        metric = self.metric.to(z)
        if self.projection is not None:
            valid = torch.arange(z.shape[1], device=z.device)[None] < lengths[:, None]
            clean = torch.where(valid[..., None], z, 0)
            delta = torch.where(valid[:, 1:, None], clean[:, 1:] - clean[:, :-1], 0)
            shared = torch.linalg.vector_norm(delta * metric.sqrt(), dim=-1)
            parts = [self._integrate(clean @ p.to(z), lengths, timestamps, scales,
                                     z.new_ones(p.shape[-1]), shared) for p in self.projection]
            return self._join(parts)
        return self._integrate(z, lengths, timestamps, scales, metric)

    @staticmethod
    def _join(parts):
        return PathStatistics(torch.cat([s.start for s in parts], 1),
                              torch.cat([s.delta for s in parts], 1),
                              torch.cat([s.area for s in parts], 1), parts[0].length,
                              torch.cat([s.timing for s in parts], 1))

    def _integrate(self, z, lengths, timestamps, scales, metric, shared=None):
        if self.integration == "clipped":
            return self._join([polygonal_statistics_reference(z, lengths, k, metric, timestamps,
                               self.mode == "temporal", self.augment_progress, shared) for k in scales])
        return pyramid_statistics(z, lengths, scales, metric, timestamps, self.mode == "temporal",
                                  self.augment_progress, shared)

    def forward(self, z: Tensor, lengths: Tensor, timestamps: Tensor | None = None,
                return_details: bool = False):
        tokens, mask, details = self.forward_with_mask(z, lengths, timestamps, return_details)
        if return_details:
            details["padding_mask"] = mask
            return tokens, details
        return tokens

    def forward_with_mask(self, z, lengths, timestamps=None, return_details=False):
        if self.adaptive_step:
            rows, diagnostics = [], []
            for i in range(z.shape[0]):
                n = int(lengths[i])
                clean = z[i:i+1, :n]
                with torch.no_grad():
                    length = torch.linalg.vector_norm((clean[:, 1:] - clean[:, :-1])
                                                     * self.metric.to(clean).sqrt(), dim=-1).sum()
                    cells = max(self.adaptive_min_bins, min(self.bins, int(torch.ceil(length / self.adaptive_step))))
                scales = (*self.scales[:-1], cells)
                tokens, details = self._forward_fixed(clean, lengths[i:i+1],
                    None if timestamps is None else timestamps[i:i+1, :n], return_details, scales)
                rows.append(tokens[0])
                diagnostics.append(details)
            count = torch.tensor([r.shape[0] for r in rows], device=z.device)
            output = nn.utils.rnn.pad_sequence(rows, batch_first=True)
            mask = torch.arange(output.shape[1], device=z.device)[None] >= count[:, None]
            return output, mask, {"examples": diagnostics, "cell_counts": count}
        output, details = self._forward_fixed(z, lengths, timestamps, return_details)
        return output, torch.zeros(output.shape[:2], device=output.device, dtype=torch.bool), details

    def _forward_fixed(self, z, lengths, timestamps, return_details, scales=None):
        # Small matrix solves must run in FP32 (FP64 retained for mathematical tests).
        dtype = torch.float64 if z.dtype == torch.float64 or self.integration_precision == "float64" else torch.float32
        # MPS has no float64 and some integration kernels are version-dependent.
        # Explicit differentiable CPU integration avoids silent unsupported ops.
        integration_device = torch.device("cpu") if z.device.type == "mps" else z.device
        with torch.autocast(device_type=z.device.type, enabled=False):
            stats = self.statistics(z.to(device=integration_device, dtype=dtype),
                                    lengths.to(integration_device),
                                    None if timestamps is None else timestamps.to(integration_device), scales)
            h = stats.area
            eye = torch.eye(h.shape[-1], device=h.device, dtype=dtype)
            gain = self.raw_gain.tanh().to(h)
            midpoint = stats.start + 0.5 * stats.delta
            q = None
            if self.mode in {"pcat", "temporal"} and self.transport_kind == "givens":
                transported = givens_transport(h, gain, midpoint[..., None]).squeeze(-1)
                if return_details:
                    q = givens_transport(h, gain, eye.expand_as(h))
            else:
                if self.mode in {"pcat", "temporal"}:
                    transported = cayley_action(h, gain, midpoint[..., None]).squeeze(-1)
                    if return_details:
                        q = cayley(h, gain)
                else:
                    transported = midpoint
                    if return_details:
                        q = eye.expand_as(h)
            # Signature baseline gets raw level-2 area; PCAT gets bounded area.
            area_feature = h if self.mode == "signature" else h / (
                1 + torch.linalg.matrix_norm(h, ord="fro", keepdim=True))
            if self.mode == "first_order":
                area_feature = torch.zeros_like(h)
            upper = self.upper.to(h.device)
            area_feature = area_feature[..., upper[0], upper[1]]
            log_length = torch.log1p(stats.length)[:, None].expand(-1, stats.start.shape[1], -1)
            features = [midpoint, stats.delta, transported, area_feature, log_length]
            if self.timing:
                features.append(stats.timing)
            features = torch.cat(features, dim=-1)
        tokens = self.readout(features.to(self.readout[0].weight))
        if return_details:
            return tokens, {"statistics": stats, "transport": q, "features": features}
        return tokens, {}
