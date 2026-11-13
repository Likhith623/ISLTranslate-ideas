import pytest
import torch
from pcat_slt.config import ModelConfig, from_dict
from pcat_slt.model import SignTranslator
from pcat_slt.moving_frame import MovingBodyFrame
from pcat_slt.operators import (
    cayley,
    cayley_action,
    givens_transport,
    ProgressCoupledAreaTransport,
    pyramid_statistics,
)
from pcat_slt.runtime import resolve_device


@pytest.mark.parametrize("n", [1, 2, 5, 16])
def test_givens_exact_norm_reversal_and_gradient(n):
    raw = torch.randn(2, 3, n, n, dtype=torch.float64, requires_grad=True)
    h = raw - raw.transpose(-1, -2)
    eye = torch.eye(n, dtype=h.dtype).expand_as(h)
    q = givens_transport(h, torch.tensor(0.7), eye)
    torch.testing.assert_close(q.transpose(-1, -2) @ q, eye, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        givens_transport(-h, torch.tensor(0.7), eye), q.transpose(-1, -2), atol=2e-12, rtol=2e-12
    )
    (q * torch.randn_like(q)).sum().backward()
    assert torch.isfinite(raw.grad).all()


def test_givens_gradcheck():
    x = torch.randn(1, 1, 4, 4, dtype=torch.float64, requires_grad=True)
    value = torch.randn(1, 1, 4, 1, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda h: givens_transport(h - h.transpose(-1, -2), torch.tensor(0.5), value), (x,)
    )


def test_exact_cayley_action_matches_dense_oracle_and_gradients():
    raw = torch.randn(2, 3, 5, 5, dtype=torch.float64, requires_grad=True)
    h = raw - raw.transpose(-1, -2)
    value = torch.randn(2, 3, 5, 1, dtype=torch.float64, requires_grad=True)
    gain = torch.tensor(0.7, dtype=torch.float64)
    dense = cayley(h, gain) @ value
    action = cayley_action(h, gain, value)
    torch.testing.assert_close(action, dense, atol=1e-12, rtol=1e-12)
    probe = torch.randn_like(action)
    a = torch.autograd.grad((action * probe).sum(), (raw, value), retain_graph=True)
    b = torch.autograd.grad((dense * probe).sum(), (raw, value))
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-11)


def test_area_sketch_is_exact_projected_area_with_shared_clock():
    z = torch.randn(2, 11, 12, dtype=torch.float64)
    lengths = torch.tensor([11, 8])
    op = ProgressCoupledAreaTransport(12, 4, 16, clock="euclidean", sketch_dim=3, sketch_heads=2).double()
    s = op.statistics(z, lengths)
    full = pyramid_statistics(z, lengths, (4,), torch.ones(12, dtype=z.dtype))
    assert s.area.shape == (2, 8, 3, 3)
    for head, p in enumerate(op.projection):
        expected = p.T @ full.area @ p
        torch.testing.assert_close(s.area[:, head * 4 : (head + 1) * 4], expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(s.length, full.length)


def test_adaptive_counts_masks_and_refinement():
    z = torch.zeros(2, 5, 4)
    z[1, :, 0] = torch.arange(5) * 10
    op = ProgressCoupledAreaTransport(4, 8, 12, scales=[1, 8], transport="givens", adaptive_step=2)
    tokens, mask, _ = op.forward_with_mask(z, torch.tensor([5, 5]))
    assert tokens.shape == (2, 9, 12)
    assert (~mask).sum(1).tolist() == [3, 9]
    alone, amask, _ = op.forward_with_mask(z[:1], torch.tensor([5]))
    torch.testing.assert_close(tokens[:1, :3], alone)
    assert not amask.any()
    fine = torch.stack([z[:, 0], (z[:, 0] + z[:, 1]) / 2, z[:, 1], z[:, 2], z[:, 3], z[:, 4]], 1)
    b, m, _ = op.forward_with_mask(fine, torch.tensor([6, 6]))
    torch.testing.assert_close(tokens, b, atol=1e-5, rtol=1e-5)
    assert torch.equal(mask, m)


def test_full_adaptive_local_clock_model_batch_independence(pose_batch):
    p, c, lengths = pose_batch
    cfg = ModelConfig(
        rank=2,
        bins=8,
        progress_scales=[1, 8],
        width=32,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_multiplier=1,
        dropout=0.0,
        max_target_length=8,
        transport="givens",
        area_sketch_dim=3,
        area_sketch_heads=2,
        adaptive_step=2,
        local_clocks=True,
        local_bins=2,
    )
    model = SignTranslator(cfg).eval()
    ids = torch.tensor([[1, 40], [1, 40]])
    together = model(p, c, lengths, ids)
    alone = model(p[1:, :5], c[1:, :5], lengths[1:], ids[1:])
    torch.testing.assert_close(together[1:], alone, atol=1e-5, rtol=1e-5)
    together.sum().backward()
    assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)


def test_local_frame_tracks_translation_and_retains_global_motion(pose_batch):
    p, c, lengths = pose_batch
    p = p[:1].clone()
    c = c[:1]
    lengths = torch.tensor([7])
    p[:, 1:] = p[:, :1]
    p[:, :, 0] = torch.tensor([-0.5, 0.0, 0.0])
    p[:, :, 1] = torch.tensor([0.5, 0.0, 0.0])
    p[:, :, 2] = torch.tensor([0.0, 0.7, 0.0])
    model = SignTranslator(
        ModelConfig(
            width=288,
            heads=4,
            detail_memory=True,
            detail_position_dim=16,
            body_frame="dual",
            body_time_constant=0.001,
            encoder_layers=1,
            decoder_layers=1,
            dropout=0.0,
        )
    )
    shift = torch.arange(7).float()[None, :, None, None] * torch.tensor([0.2, 0.0, 0.0])
    p = p + shift
    f = model.canonical_features(p, c, lengths)
    ts = torch.arange(7).float()[None]
    local, motion = model.moving_frame(f, lengths, ts)
    torch.testing.assert_close(local[:, 0], local[:, -1], atol=1e-5, rtol=1e-5)
    assert motion[0, -1, 0] > 1
    memory, mask = model.encode_with_mask(p, c, lengths, ts)
    torch.testing.assert_close(memory[:, -7:, :264], f.flatten(2))
    assert not mask.any()


def test_moving_frame_valid_prefix_is_causal(pose_batch):
    p, c, lengths = pose_batch
    f = torch.cat((p, c[..., None]), -1)
    module = MovingBodyFrame([0, 1, 2])
    ts = torch.arange(7).float().expand(2, 7)
    a, _ = module(f, lengths, ts)
    f = f.clone()
    f[:, 4:] += 3
    b, _ = module(f, lengths, ts)
    torch.testing.assert_close(a[:, :4], b[:, :4])


def test_depth_ignore_is_not_affected_by_unreliable_z(pose_batch):
    p, c, lengths = pose_batch
    model = SignTranslator(
        ModelConfig(width=24, heads=4, depth_policy="ignore", dropout=0.0, encoder_layers=1, decoder_layers=1)
    ).eval()
    a = model.encode(p, c, lengths)
    p = p.clone()
    p[..., 2] = torch.randn_like(p[..., 2]) * 1000
    torch.testing.assert_close(a, model.encode(p, c, lengths))


def test_mps_config_and_unavailable_host_error():
    from_dict({"train": {"device": "mps"}})
    with pytest.raises(ValueError):
        from_dict({"train": {"device": "mps"}, "model": {"integration_precision": "float64"}})
    if not torch.backends.mps.is_available():
        with pytest.raises(RuntimeError, match="unavailable"):
            resolve_device("mps")


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS hardware unavailable")
def test_actual_mps_forward_backward(pose_batch):
    p, c, lengths = pose_batch
    cfg = ModelConfig(width=24, heads=4, transport="givens", dropout=0.0, encoder_layers=1, decoder_layers=1)
    model = SignTranslator(cfg).to("mps")
    out = model(p.to("mps"), c.to("mps"), lengths.to("mps"), torch.ones(2, 2, dtype=torch.long, device="mps"))
    out.sum().backward()
    assert torch.isfinite(out).all()
