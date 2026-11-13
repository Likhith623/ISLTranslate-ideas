"""Regression tests for limitations addressed by v0.2."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from pcat_slt.config import ModelConfig, DataConfig, from_dict, load_config
from pcat_slt.data import SignDataset, assert_disjoint, load_pose
from pcat_slt.geometry import BodyFrame
from pcat_slt.model import SignTranslator
from pcat_slt.operators import (ProgressCoupledAreaTransport, polygonal_statistics,
                                polygonal_statistics_reference)
from pcat_slt.synthetic import make_synthetic
from pcat_slt.tokenizer import ByteTokenizer, Utf8Constraint


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("cells", [1, 4, 17])
def test_prefix_integrator_matches_independent_clipping_oracle(temporal, cells):
    z = torch.randn(2, 11, 5, dtype=torch.float64)
    z[:, 3:6] = z[:, 2:3]
    lengths = torch.tensor([11, 8])
    z[1, 8:] = torch.nan
    metric = torch.rand(5, dtype=z.dtype) + .1
    ts = torch.rand(2, 11, dtype=z.dtype).cumsum(1)
    fast = polygonal_statistics(z, lengths, cells, metric, ts, temporal)
    oracle = polygonal_statistics_reference(z, lengths, cells, metric, ts, temporal)
    for field in ("start", "delta", "area", "length", "timing"):
        torch.testing.assert_close(getattr(fast, field), getattr(oracle, field), atol=3e-11, rtol=3e-11)


def test_prefix_gradients_match_oracle():
    z = torch.randn(1, 9, 3, dtype=torch.float64, requires_grad=True)
    metric = torch.rand(3, dtype=z.dtype, requires_grad=True) + .3
    length = torch.tensor([9])
    probe = torch.randn(1, 5, 3, 3, dtype=z.dtype)
    a = polygonal_statistics(z, length, 5, metric).area
    b = polygonal_statistics_reference(z, length, 5, metric).area
    ga = torch.autograd.grad((a * probe).sum(), (z, metric), retain_graph=True)
    gb = torch.autograd.grad((b * probe).sum(), (z, metric))
    for left, right in zip(ga, gb):
        torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_pyramid_augmented_subdivision_invariance():
    z = torch.randn(2, 7, 4, dtype=torch.float64)
    fine = [z[:, 0]]
    for i in range(6):
        fine.extend([z[:, i] + a * (z[:, i + 1] - z[:, i]) for a in (.17, .61, 1.)])
    fine = torch.stack(fine, 1)
    layer = ProgressCoupledAreaTransport(4, 8, 12, scales=[1, 2, 8], augment_progress=True).double()
    coarse, details = layer(z, torch.tensor([7, 7]), return_details=True)
    refined = layer(fine, torch.tensor([19, 19]))
    assert coarse.shape == (2, 11, 12)
    assert details["statistics"].area.shape[-2:] == (5, 5)
    torch.testing.assert_close(coarse, refined, atol=1e-10, rtol=1e-10)


def test_augmented_prefix_and_clipped_backends_match():
    z = torch.randn(2, 12, 4, dtype=torch.float64)
    z[:, 4:6] = z[:, 3:4]
    a = ProgressCoupledAreaTransport(4, 8, 12, scales=[1, 4, 8], augment_progress=True).double()
    b = ProgressCoupledAreaTransport(4, 8, 12, scales=[1, 4, 8], augment_progress=True,
                                    integration="clipped").double()
    b.load_state_dict(a.state_dict())
    left = a(z, torch.tensor([12, 9]))
    right = b(z, torch.tensor([12, 9]))
    torch.testing.assert_close(left, right, atol=1e-10, rtol=1e-10)


def test_float64_integration_supports_float32_network_backward():
    layer = ProgressCoupledAreaTransport(3, 4, 8, integration_precision="float64")
    z = torch.randn(2, 7, 3, requires_grad=True)
    tokens, details = layer(z, torch.tensor([7, 7]), return_details=True)
    assert details["statistics"].area.dtype == torch.float64
    assert tokens.dtype == torch.float32
    tokens.sum().backward()
    assert torch.isfinite(z.grad).all()


def test_progress_coordinate_separates_retracing_counterexample():
    z = torch.tensor([[[0.], [1.], [0.]], [[0.], [-1.], [0.]]], dtype=torch.float64)
    plain = ProgressCoupledAreaTransport(1, 1, 8).double()
    improved = ProgressCoupledAreaTransport(1, 1, 8, augment_progress=True).double()
    _, old = plain(z, torch.tensor([3, 3]), return_details=True)
    _, new = improved(z, torch.tensor([3, 3]), return_details=True)
    torch.testing.assert_close(old["features"][0], old["features"][1])
    assert not torch.allclose(new["features"][0], new["features"][1])


def test_coarse_scale_retains_order_across_fine_cell_boundaries():
    z = torch.tensor([[[0., 0.], [1., 0.], [1., 1.]]])
    layer = ProgressCoupledAreaTransport(2, 2, 8, scales=[1, 2])
    s = layer.statistics(z, torch.tensor([3]))
    assert s.area[0, 0, 0, 1].detach().item() == pytest.approx(.5)
    torch.testing.assert_close(s.area[0, 1:], torch.zeros_like(s.area[0, 1:]))


def test_metric_condition_bound_for_extreme_parameters():
    layer = ProgressCoupledAreaTransport(3, 4, 8, metric_condition_limit=16)
    with torch.no_grad():
        layer.raw_metric.copy_(torch.tensor([-1e6, 0., 1e6]))
    assert torch.isfinite(layer.metric).all()
    assert layer.metric.max() / layer.metric.min() <= 16
    assert layer.metric.min() >= 1 / 16


def test_body_reference_fallback_and_padding(pose_batch):
    pose, conf, lengths = pose_batch
    conf[:, :2, :3] = 0
    frame = BodyFrame()
    assert frame.reference_indices(pose, conf, lengths).tolist() == [2, 2]
    normalized = frame(pose, conf, lengths)
    assert torch.isfinite(normalized).all()
    with pytest.raises(ValueError, match="no reliable"):
        BodyFrame(policy="first")(pose, conf, lengths)
    conf[1, :5, :3] = 0
    with pytest.raises(ValueError, match="no reliable"):
        frame(pose, conf, lengths)


def test_loader_accepts_missing_initial_anchors(tmp_path):
    root = make_synthetic(tmp_path / "demo", samples=4)
    path = root / "train_00000.npz"
    with np.load(path) as f:
        arrays = {key: f[key].copy() for key in f.files}
    arrays["confidence"][:2, :3] = 0
    arrays["pose"][:2, :3] = np.nan
    np.savez(path, **arrays)
    sample = load_pose(path, ModelConfig(), 64)
    assert torch.isfinite(sample["pose"]).all()


def rich_config(**extra):
    return ModelConfig(rank=2, bins=4, progress_scales=[1, 2, 4], progress_augmentation=True,
                       detail_memory=True, width=288, heads=4, detail_position_dim=16,
                       encoder_layers=1, decoder_layers=1, ff_multiplier=1, dropout=0.,
                       max_target_length=16, **extra)


def test_detail_memory_preserves_all_canonical_coordinates(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(rich_config()).eval()
    memory, mask = model.encode_with_mask(pose, confidence, lengths)
    expected = model.canonical_features(pose, confidence, lengths).flatten(2)
    assert memory.shape == (2, 14, 288)
    torch.testing.assert_close(memory[:, 7:, :264], expected, atol=0, rtol=0)
    assert mask[1, 12:].all() and not mask[1, :12].any()


def test_rich_model_backward_and_padded_nan_isolation(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(rich_config(timing=True)).eval()
    ts = torch.arange(7).float().expand(2, 7).clone()
    ids = torch.tensor([[1, 50], [1, 50]])
    a = model(pose, confidence, lengths, ids, ts)
    pose[1, 5:], confidence[1, 5:], ts[1, 5:] = torch.nan, torch.nan, torch.nan
    b = model(pose, confidence, lengths, ids, ts)
    torch.testing.assert_close(a, b)
    (b * torch.randn_like(b)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_invariant_submemory_remains_available_with_detail(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(rich_config()).eval()
    invariant = model.encode_invariant(pose, confidence, lengths)
    complete, _ = model.encode_with_mask(pose, confidence, lengths)
    torch.testing.assert_close(invariant, complete[:, :7])


@pytest.mark.parametrize("text", ["Hello", "नमस्ते", "ಕನ್ನಡ", "🤟🧠", "\U0010ffff", "\u0800\ud7ff\ue000"])
def test_utf8_fsm_accepts_valid_unicode(text):
    tokens = ByteTokenizer().encode(text)[1:]
    constraint = Utf8Constraint(1, "cpu")
    for index, token in enumerate(tokens):
        logits = torch.zeros(1, 259)
        masked = constraint.mask(logits, len(tokens) - index)
        assert torch.isfinite(masked[0, token])
        tensor = torch.tensor([token])
        constraint.advance(tensor, tensor.eq(2))
    assert constraint.state.item() == 0


@pytest.mark.parametrize("invalid", [[0x80], [0xC0], [0xC1], [0xF5], [0xFF],
                                     [0xE0, 0x80], [0xED, 0xA0], [0xF0, 0x80], [0xF4, 0x90]])
def test_utf8_fsm_rejects_invalid_sequences(invalid):
    constraint = Utf8Constraint(1, "cpu")
    for index, byte in enumerate(invalid):
        token = byte + 3
        masked = constraint.mask(torch.zeros(1, 259), 10)
        if index == len(invalid) - 1:
            assert masked[0, token].isneginf()
        else:
            constraint.advance(torch.tensor([token]), torch.tensor([False]))


def test_random_utf8_decoding_never_exits_with_incomplete_codepoint():
    for budget in range(1, 10):
        constraint = Utf8Constraint(16, "cpu")
        outputs = [[] for _ in range(16)]
        finished = torch.zeros(16, dtype=torch.bool)
        for step in range(budget):
            logits = torch.randn(16, 259)
            logits[:, 2] -= 10  # exercise actual bytes instead of early EOS
            tokens = constraint.mask(logits, budget - step).argmax(-1)
            tokens = torch.where(finished, 0, tokens)
            finished |= tokens.eq(2)
            constraint.advance(tokens, finished)
            for row, token in zip(outputs, tokens.tolist()):
                row.append(token)
        for row in outputs:
            ByteTokenizer().decode(row, errors="strict")


def test_source_recording_leakage_rejected(tmp_path):
    root = make_synthetic(tmp_path / "data", samples=4)
    for split in ("train", "val"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row["source_video"] = "shared-recording"
        path.write_text("\n".join(json.dumps(row) for row in rows))
    datasets = {s: SignDataset(root / f"{s}.jsonl", ModelConfig(), DataConfig()) for s in ("train", "val")}
    with pytest.raises(ValueError, match="source/group leakage"):
        assert_disjoint(datasets)


@pytest.mark.parametrize("changes", [
    {"progress_scales": [4, 2]}, {"progress_scales": [1, 1, 32]},
    {"detail_memory": True}, {"detail_position_dim": 3}, {"metric_condition_limit": .5},
])
def test_invalid_enhanced_configuration_rejected(changes):
    with pytest.raises(ValueError):
        from_dict({"model": changes})


@pytest.mark.parametrize("name", ["base", "invariant"])
def test_recommended_full_configuration_forward_and_backward(name, pose_batch):
    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / f"{name}.yaml")
    model = SignTranslator(cfg.model).eval()
    pose, confidence, lengths = pose_batch
    ts = torch.arange(7).float().expand(2, 7) / 25
    logits = model(pose, confidence, lengths, torch.tensor([[1, 50], [1, 50]]), ts)
    assert logits.shape == (2, 2, 259)
    (logits * torch.randn_like(logits)).sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
