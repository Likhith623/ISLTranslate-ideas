import pytest
import torch
from pcat_slt.geometry import BodyFrame
from pcat_slt.operators import ProgressCoupledAreaTransport, polygonal_statistics, cayley


def refine(path):
    output = [path[:, 0]]
    for i in range(path.shape[1] - 1):
        for fraction in (0.13, 0.62, 1.0):
            output.append(path[:, i] + fraction * (path[:, i + 1] - path[:, i]))
    return torch.stack(output, 1)


def test_exact_signed_area_and_skew():
    z = torch.tensor([[[0., 0.], [1., 0.], [1., 1.]]], dtype=torch.float64)
    s = polygonal_statistics(z, torch.tensor([3]), 1, torch.ones(2, dtype=z.dtype))
    torch.testing.assert_close(s.area[0, 0], torch.tensor([[0., .5], [-.5, 0.]], dtype=z.dtype))
    torch.testing.assert_close(s.area + s.area.transpose(-1, -2), torch.zeros_like(s.area))
    torch.testing.assert_close(s.delta[0, 0], torch.ones(2, dtype=z.dtype))


@pytest.mark.parametrize("bins", [1, 3, 8])
def test_exact_clipped_subdivision_invariance(bins):
    z = torch.randn(2, 6, 4, dtype=torch.float64)
    operator = ProgressCoupledAreaTransport(4, bins, 12).double().eval()
    fine = refine(z)
    a, da = operator(z, torch.tensor([6, 6]), return_details=True)
    b, db = operator(fine, torch.tensor([fine.shape[1]] * 2), return_details=True)
    torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-11)
    for field in ("start", "delta", "area", "length"):
        torch.testing.assert_close(getattr(da["statistics"], field), getattr(db["statistics"], field),
                                   atol=2e-11, rtol=2e-11)


def test_timestamp_warp_and_duplicate_vertex_invariance():
    z = torch.randn(1, 5, 3, dtype=torch.float64)
    layer = ProgressCoupledAreaTransport(3, 5, 8).double().eval()
    lengths = torch.tensor([5])
    ts = torch.tensor([[0., .02, .1, 4., 8.]], dtype=z.dtype)
    a = layer(z, lengths, ts)
    b = layer(z, lengths, ts.square())
    repeated = z.repeat_interleave(2, dim=1)
    c = layer(repeated, torch.tensor([10]))
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a, c)


def test_independent_articulator_order_is_retained():
    z = torch.tensor([[[0., 0.], [1., 0.], [1., 1.]],
                      [[0., 0.], [0., 1.], [1., 1.]]], dtype=torch.float64)
    lengths = torch.tensor([3, 3])
    layer = ProgressCoupledAreaTransport(2, 1, 8).double().eval()
    _, details = layer(z, lengths, return_details=True)
    h = details["statistics"].area
    torch.testing.assert_close(h[0], -h[1])
    assert not torch.allclose(details["transport"][0], details["transport"][1])
    # Any feature based only on separate first-order paths loses this distinction.
    first = ProgressCoupledAreaTransport(2, 1, 8, mode="first_order").double().eval()
    _, control = first(z, lengths, return_details=True)
    torch.testing.assert_close(control["features"][0], control["features"][1])


def test_cayley_orthogonality_norm_and_time_reversal():
    raw = torch.randn(3, 5, 5, dtype=torch.float64)
    h = raw - raw.transpose(-1, -2)
    gain = torch.tensor(.7, dtype=h.dtype)
    q = cayley(h, gain)
    eye = torch.eye(5, dtype=h.dtype).expand(3, -1, -1)
    torch.testing.assert_close(q.transpose(-1, -2) @ q, eye, atol=1e-12, rtol=1e-12)
    x = torch.randn(3, 5, 1, dtype=h.dtype)
    torch.testing.assert_close((q @ x).norm(dim=-2), x.norm(dim=-2))
    torch.testing.assert_close(cayley(-h, gain), q.transpose(-1, -2))


def test_cayley_lipschitz_bound():
    a = torch.randn(5, 4, 4, dtype=torch.float64)
    b = a + 0.1 * torch.randn_like(a)
    a, b = a - a.transpose(-1, -2), b - b.transpose(-1, -2)
    gain = torch.tensor(.8, dtype=a.dtype)
    observed = torch.linalg.matrix_norm(cayley(a, gain) - cayley(b, gain))
    bound = 4 * gain.abs() * torch.linalg.matrix_norm(a - b)
    assert (observed <= bound + 1e-12).all()


def test_concatenation_area_identity():
    x = torch.randn(1, 8, 3, dtype=torch.float64)
    metric = torch.ones(3, dtype=x.dtype)
    def stats(path):
        return polygonal_statistics(path, torch.tensor([path.shape[1]]), 1, metric)
    full, left, right = stats(x), stats(x[:, :4]), stats(x[:, 3:])
    u, v = left.delta[:, 0], right.delta[:, 0]
    cross = .5 * (u[..., None] @ v[:, None] - v[..., None] @ u[:, None])
    torch.testing.assert_close(full.area[:, 0], left.area[:, 0] + right.area[:, 0] + cross)


def test_stationary_path_and_metric_gradients():
    layer = ProgressCoupledAreaTransport(3, 5, 8).double()
    z = torch.ones(2, 7, 3, dtype=torch.float64, requires_grad=True)
    result, d = layer(z, torch.tensor([7, 3]), return_details=True)
    assert torch.isfinite(result).all()
    torch.testing.assert_close(d["statistics"].area, torch.zeros_like(d["statistics"].area))
    result.square().sum().backward()
    assert torch.isfinite(z.grad).all()
    assert torch.isfinite(layer.raw_metric.grad).all()


def test_autograd_gradcheck_away_from_cell_boundaries():
    z = torch.randn(1, 5, 3, dtype=torch.float64, requires_grad=True)
    layer = ProgressCoupledAreaTransport(3, 3, 6).double()
    assert torch.autograd.gradcheck(lambda x: layer(x, torch.tensor([5])), (z,), atol=1e-4, rtol=1e-3)


def test_learned_metric_and_gain_receive_nonzero_gradients():
    layer = ProgressCoupledAreaTransport(4, 3, 8).double()
    z = torch.randn(2, 7, 4, dtype=torch.float64)
    out = layer(z, torch.tensor([7, 7]))
    (out * torch.randn_like(out)).sum().backward()
    assert layer.raw_metric.grad.abs().sum() > 1e-7
    assert layer.raw_gain.grad.abs() > 1e-7
    assert (layer.metric > 0).all()


def test_similarity_invariance_and_chirality(pose_batch):
    pose, confidence, _ = pose_batch
    pose, confidence = pose.double(), confidence.double()
    frame = BodyFrame()
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=pose.dtype))
    q[:, 0] *= torch.det(q)
    transformed = 3.7 * (pose @ q.T) + torch.tensor([5., -2., 4.], dtype=pose.dtype)
    torch.testing.assert_close(frame(pose, confidence), frame(transformed, confidence), atol=1e-11, rtol=1e-11)
    reflected = pose.clone()
    reflected[..., 2] *= -1
    assert not torch.allclose(frame(pose, confidence), frame(reflected, confidence))


def test_degenerate_anchor_rejected(pose_batch):
    pose, confidence, _ = pose_batch
    pose[:, :, 2] = 0
    with pytest.raises(ValueError, match="collinear"):
        BodyFrame()(pose, confidence)


def test_timing_sidechannel_intentionally_changes_under_speed():
    z = torch.tensor([[[0., 0.], [1., 0.], [1., 1.], [1., 1.]]])
    layer = ProgressCoupledAreaTransport(2, 3, 6, timing=True)
    ts = torch.tensor([[0., 1., 2., 5.]])
    _, a = layer(z, torch.tensor([4]), ts, True)
    _, b = layer(z, torch.tensor([4]), ts * 2, True)
    torch.testing.assert_close(a["statistics"].area, b["statistics"].area)
    assert not torch.allclose(a["statistics"].timing, b["statistics"].timing)
    # Elapsed fractions include stationary segments and conserve total time.
    torch.testing.assert_close(a["statistics"].timing[..., 1].sum(1), torch.ones(1))


def test_double_timestamps_with_float32_features():
    z = torch.randn(2, 5, 3)
    ts = torch.arange(5, dtype=torch.float64).expand(2, 5)
    layer = ProgressCoupledAreaTransport(3, 4, 8, timing=True)
    assert layer(z, torch.tensor([5, 5]), ts).dtype == torch.float32
