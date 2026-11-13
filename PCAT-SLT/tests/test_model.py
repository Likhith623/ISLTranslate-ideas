import pytest
import torch
from pcat_slt.config import ModelConfig
from pcat_slt.model import SignTranslator


def small_config(**kwargs):
    return ModelConfig(width=24, heads=4, rank=2, bins=4, encoder_layers=1,
                       decoder_layers=1, ff_multiplier=2, dropout=0., max_target_length=16, **kwargs)


@pytest.mark.parametrize("mode", ["pcat", "signature", "first_order", "no_transport", "temporal"])
def test_full_model_shape_and_backward(pose_batch, mode):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(small_config(operator=mode))
    ids = torch.tensor([[1, 100, 101], [1, 60, 0]])
    logits = model(pose, confidence, lengths, ids)
    assert logits.shape == (2, 3, 259)
    logits.square().mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.region_maps[0].weight.grad.abs().sum() > 0


def test_padded_garbage_has_no_influence(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(small_config()).eval()
    ids = torch.tensor([[1, 100], [1, 100]])
    a = model(pose, confidence, lengths, ids)
    pose[1, 5:] = torch.nan
    confidence[1, 5:] = torch.nan
    b = model(pose, confidence, lengths, ids)
    torch.testing.assert_close(a, b)


def test_decoder_causality(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(small_config()).eval()
    ids = torch.tensor([[1, 80, 81, 82], [1, 80, 81, 82]])
    a = model(pose, confidence, lengths, ids)
    ids[:, 2:] = 120
    b = model(pose, confidence, lengths, ids)
    torch.testing.assert_close(a[:, :2], b[:, :2])


def test_full_encoder_refinement_invariance(pose_batch):
    pose, confidence, _ = pose_batch
    pose, confidence = pose[:1].double(), confidence[:1].double()
    refined = [pose[:, 0]]
    for i in range(6):
        refined.extend([(pose[:, i] + pose[:, i + 1]) / 2, pose[:, i + 1]])
    refined = torch.stack(refined, 1)
    model = SignTranslator(small_config()).double().eval()
    a = model.encode(pose, confidence, torch.tensor([7]))
    b = model.encode(refined, torch.ones(1, 13, 66, dtype=torch.float64), torch.tensor([13]))
    torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)


def test_generation_eos_and_limits(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(small_config()).eval()
    with torch.no_grad():
        model.output.weight.zero_()
    # All logits tied: PAD/BOS suppressed; EOS (id=2) wins deterministically.
    ids = model.generate(pose, confidence, lengths)
    assert ids.tolist() == [[1, 2], [1, 2]]
    with pytest.raises(ValueError):
        model.generate(pose, confidence, lengths, max_new_tokens=17)
    model.train()
    with pytest.raises(RuntimeError):
        model.generate(pose, confidence, lengths)


def test_batch_independence(pose_batch):
    pose, confidence, lengths = pose_batch
    model = SignTranslator(small_config()).eval()
    together = model.encode(pose, confidence, lengths)
    alone = model.encode(pose[1:2, :5], confidence[1:2, :5], lengths[1:2])
    torch.testing.assert_close(together[1:2], alone, atol=3e-6, rtol=3e-6)
