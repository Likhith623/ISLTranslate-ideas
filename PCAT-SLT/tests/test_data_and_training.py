import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from pcat_slt.config import Config, ModelConfig, DataConfig, TrainConfig, from_dict
from pcat_slt.data import SignDataset, collate_signs, assert_disjoint, load_pose
from pcat_slt.metrics import translation_metrics
from pcat_slt.runtime import capture_rng, restore_rng, load_checkpoint, input_fingerprint
from pcat_slt.synthetic import make_synthetic
from pcat_slt.tokenizer import ByteTokenizer
from pcat_slt.train import train, token_loss
from pcat_slt.validate import validate


def demo_cfg(tmp_path, epochs=2):
    data = make_synthetic(tmp_path / "data", samples=7)
    cfg = Config(ModelConfig(width=16, heads=2, rank=1, bins=3, encoder_layers=1,
                             decoder_layers=1, ff_multiplier=2, dropout=.1, max_target_length=8),
                 DataConfig(train=str(data / "train.jsonl"), val=str(data / "val.jsonl"),
                            test=str(data / "test.jsonl"), batch_size=3, max_frames=64, max_bytes=6),
                 TrainConfig(epochs=epochs, lr=.003, accumulation=2, warmup_steps=0,
                             output_dir=str(tmp_path / "run"), device="cpu"))
    cfg.validate()
    return cfg


def test_utf8_roundtrip():
    tokenizer = ByteTokenizer()
    text = "Hello, नमस्ते ಕನ್ನಡ 🤟"
    assert tokenizer.decode(tokenizer.encode(text) + [0, 0]) == text
    assert tokenizer.decode([1, 3 + 255, 2]) == "�"


def test_dataset_collator_and_validation(tmp_path):
    cfg = demo_cfg(tmp_path)
    dataset = SignDataset(cfg.data.train, cfg.model, cfg.data)
    batch = collate_signs([dataset[0], dataset[1]])
    assert batch["pose"].shape[0] == 2
    assert batch["decoder_input"].shape == batch["labels"].shape == (2, 3)
    report = validate(cfg, signer_independent=True)
    assert report["train"]["samples"] == 7
    assert not any(report["signer_overlap"].values())


def test_missing_coordinates_imputed_and_bad_observations_rejected(tmp_path):
    cfg = demo_cfg(tmp_path)
    path = Path(cfg.data.train).parent / "train_00000.npz"
    with np.load(path) as f:
        arrays = {k: f[k].copy() for k in f.files}
    arrays["confidence"][2:4, 3] = 0
    arrays["pose"][2:4, 3] = np.nan
    np.savez(path, **arrays)
    sample = load_pose(path, cfg.model, 64)
    assert torch.isfinite(sample["pose"]).all()
    assert sample["confidence"][2:4, 3].eq(0).all()
    arrays["confidence"][2, 3] = 1
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="observed coordinates"):
        load_pose(path, cfg.model, 64)


def test_split_leakage_and_long_targets_rejected(tmp_path):
    cfg = demo_cfg(tmp_path)
    dataset = SignDataset(cfg.data.train, cfg.model, cfg.data)
    with pytest.raises(ValueError, match="duplicate"):
        assert_disjoint({"train": dataset, "val": dataset})
    cfg.data.max_bytes = 1
    with pytest.raises(ValueError, match="max_bytes"):
        SignDataset(cfg.data.train, cfg.model, cfg.data)


@pytest.mark.parametrize("mutation", [
    {"model": {"heads": 7}}, {"model": {"metric_floor": 0}},
    {"model": {"operator": "unknown"}}, {"data": {"max_frames": 1}},
    {"train": {"accumulation": 0}}, {"train": {"lr": float("nan")}},
    {"train": {"device": "unsupported"}}, {"typo": {}},
])
def test_invalid_configs_rejected(mutation):
    with pytest.raises((ValueError, TypeError)):
        from_dict(mutation)


def test_rng_roundtrip():
    state = capture_rng()
    expected = (random.random(), np.random.rand(), torch.rand(4))
    restore_rng(state)
    actual = (random.random(), np.random.rand(), torch.rand(4))
    assert expected[:2] == actual[:2]
    torch.testing.assert_close(expected[2], actual[2])


def test_input_fingerprint_detects_npz_changes(tmp_path):
    cfg = demo_cfg(tmp_path)
    dataset = SignDataset(cfg.data.train, cfg.model, cfg.data)
    before = input_fingerprint(dataset)
    path = Path(dataset.records[0]["pose"])
    with np.load(path) as f:
        arrays = {key: f[key].copy() for key in f.files}
    arrays["pose"][3, 4, 0] += .1
    np.savez_compressed(path, **arrays)
    assert input_fingerprint(dataset) != before


@pytest.mark.parametrize("rich", [False, True])
def test_epoch_resume_exactly_matches_uninterrupted_training(tmp_path, rich):
    cfg = demo_cfg(tmp_path)
    if rich:
        cfg.model = ModelConfig(width=288, heads=4, rank=1, bins=3, progress_scales=[1, 3],
                                progress_augmentation=True, detail_memory=True, detail_position_dim=16,
                                timing=True, integration_precision="float64", encoder_layers=1,
                                decoder_layers=1, ff_multiplier=1,
                                dropout=.1, max_target_length=8)
    full_path = train(cfg)
    _, _, full = load_checkpoint(full_path)
    cfg.train.output_dir = str(tmp_path / "resumed")
    partial = train(cfg, stop_after_epoch=1)
    train(cfg, resume=partial)
    _, _, resumed = load_checkpoint(partial)
    assert resumed["epoch"] == full["epoch"] == 2
    assert resumed["global_step"] == full["global_step"] == 4
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], atol=0, rtol=0)
    assert full["best_val_nll"] == resumed["best_val_nll"]
    # Epoch reports are actual losses, not constructed success criteria.
    history = [json.loads(line) for line in (Path(cfg.train.output_dir) / "history.jsonl").read_text().splitlines()]
    assert history[-1]["train_smoothed_ce"] < history[0]["train_smoothed_ce"]


def test_token_loss_accumulation_matches_concatenation():
    a = torch.randn(2, 4, 259, requires_grad=True)
    b = torch.randn(1, 4, 259, requires_grad=True)
    ya = torch.tensor([[40, 41, 2, 0], [42, 2, 0, 0]])
    yb = torch.tensor([[43, 44, 45, 2]])
    n = int(ya.ne(0).sum() + yb.ne(0).sum())
    expected = token_loss(torch.cat([a, b]), torch.cat([ya, yb])) / n
    split = (token_loss(a, ya) + token_loss(b, yb)) / n
    torch.testing.assert_close(expected, split)


def test_metrics_exact_and_wrong():
    ref = ["one two three four five", "six seven eight nine ten"]
    exact = translation_metrics(ref, ref)
    assert exact["bleu4"] == pytest.approx(100)
    assert exact["rouge_l_f1"] == exact["exact_match"] == 100
    wrong = translation_metrics(["x", "y"], ref)
    assert wrong["bleu4"] < exact["bleu4"]
    assert "tok:13a" in exact["signatures"]["bleu4"]
