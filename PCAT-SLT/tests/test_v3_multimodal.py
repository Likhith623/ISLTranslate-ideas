import json
from pathlib import Path
import pickle
import shutil
import numpy as np
import pytest
import torch
from pcat_slt.config import ModelConfig, Config, DataConfig, TrainConfig
from pcat_slt.data import SignDataset, collate_signs, load_rgb, model_inputs, make_collator
from pcat_slt.model import SignTranslator
from pcat_slt.corpora import import_corpus, read_metadata
from pcat_slt.runtime import atomic_save, load_checkpoint
from pcat_slt.synthetic import make_synthetic
from pcat_slt.text import codec_payload
from pcat_slt.train import train


def visual_cfg(**extra):
    return ModelConfig(
        width=32,
        heads=4,
        rank=2,
        bins=4,
        encoder_layers=1,
        decoder_layers=1,
        ff_multiplier=1,
        dropout=0.0,
        max_target_length=16,
        visual_backend="cnn",
        visual_dim=8,
        **extra,
    )


def test_actual_rgb_observation_changes_model_and_masked_pixels_do_not(pose_batch):
    p, c, lengths = pose_batch
    model = SignTranslator(visual_cfg()).eval()
    rgb = torch.rand(2, 7, 4, 3, 24, 24)
    mask = torch.ones(2, 7, 4, dtype=torch.bool)
    mask[:, 2, 2] = False
    ids = torch.tensor([[1, 50], [1, 50]])
    a = model(p, c, lengths, ids, rgb=rgb, rgb_mask=mask)
    masked = rgb.clone()
    masked[:, 2, 2] = torch.nan
    masked[1, 5:] = torch.nan
    b = model(p, c, lengths, ids, rgb=masked, rgb_mask=mask)
    torch.testing.assert_close(a, b)
    changed = rgb.clone()
    changed[:, :, 2] = 1 - rgb[:, :, 2]
    out = model(p, c, lengths, ids, rgb=changed, rgb_mask=mask)
    assert not torch.allclose(a, out, atol=1e-7, rtol=1e-7)
    (out * torch.randn_like(out)).sum().backward()
    assert model.vision.projection.weight.grad.abs().sum() > 0


def add_visual_data(root):
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        for row in rows:
            with np.load(root / row["pose"]) as f:
                ts = f["timestamps"]
            name = row["id"] + "_rgb.npz"
            np.savez_compressed(
                root / name,
                rgb=np.full((len(ts), 4, 3, 16, 16), 0.5, np.float32),
                rgb_mask=np.ones((len(ts), 4), bool),
                timestamps=ts,
            )
            row["rgb"] = name
        path.write_text("\n".join(json.dumps(r) for r in rows))


def test_rgb_loader_alignment_and_collation(tmp_path):
    root = make_synthetic(tmp_path / "data", samples=4)
    add_visual_data(root)
    cfg = visual_cfg()
    d = DataConfig(max_frames=64, max_bytes=14)
    ds = SignDataset(root / "train.jsonl", cfg, d)
    batch = collate_signs([ds[0], ds[1]])
    assert batch["rgb"].shape[0] == 2 and batch["rgb_mask"].dtype == torch.bool
    sample = ds[0]
    path = root / "train_00000_rgb.npz"
    with np.load(path) as f:
        payload = {k: f[k].copy() for k in f.files}
    payload["timestamps"][3] += 0.001
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="not aligned"):
        load_rgb(path, sample, cfg)
    payload["timestamps"][3] -= 0.001
    payload["timestamps"] += 0.5
    np.savez(path, **payload)
    with pytest.raises(ValueError, match="not aligned"):
        load_rgb(path, sample, cfg)


def test_visual_temporal_order_survives_zero_pose_progress(pose_batch):
    pose, confidence, _ = pose_batch
    pose = pose[:, :1].expand(-1, 7, -1, -1).clone()
    lengths = torch.tensor([7, 7])
    model = SignTranslator(visual_cfg(visual_grid=4, visual_memory_norm=True)).eval()
    rgb = torch.rand(2, 7, 4, 3, 24, 24)
    mask = torch.ones(2, 7, 4, dtype=torch.bool)
    ids = torch.ones(2, 2, dtype=torch.long)
    original = model(pose, confidence, lengths, ids, rgb=rgb, rgb_mask=mask)
    reversed_frames = model(pose, confidence, lengths, ids, rgb=rgb.flip(1), rgb_mask=mask)
    assert not torch.allclose(original, reversed_frames, atol=1e-7, rtol=1e-7)


def test_unanchored_rgb_fallback_is_explicit_and_requires_observations(pose_batch):
    pose, confidence, lengths = pose_batch
    pose, confidence = torch.zeros_like(pose), torch.zeros_like(confidence)
    cfg = visual_cfg(allow_unanchored_rgb=True, visual_memory_norm=True)
    model = SignTranslator(cfg).eval()
    rgb = torch.rand(2, 7, 4, 3, 24, 24)
    mask = torch.ones(2, 7, 4, dtype=torch.bool)
    ids = torch.ones(2, 2, dtype=torch.long)
    output = model(pose, confidence, lengths, ids, rgb=rgb, rgb_mask=mask)
    assert torch.isfinite(output).all()
    assert model.canonical_features(pose, confidence, lengths).eq(0).all()
    with pytest.raises(ValueError, match="neither anchored"):
        model(pose, confidence, lengths, ids, rgb=rgb, rgb_mask=torch.zeros_like(mask))


def test_unanchored_dual_frame_does_not_expose_fabricated_motion(pose_batch):
    pose, confidence, lengths = pose_batch
    cfg = ModelConfig(
        width=288,
        heads=4,
        rank=2,
        bins=4,
        detail_memory=True,
        detail_position_dim=16,
        body_frame="dual",
        visual_backend="cnn",
        visual_dim=8,
        allow_unanchored_rgb=True,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )
    model = SignTranslator(cfg).eval()
    _, mask = model.encode_with_mask(
        torch.zeros_like(pose),
        torch.zeros_like(confidence),
        lengths,
        torch.arange(7).float().expand(2, 7),
        torch.rand(2, 7, 4, 3, 24, 24),
        torch.ones(2, 7, 4, dtype=torch.bool),
    )
    assert mask[:, 4 + 28 : 4 + 28 + 7].all()


def test_large_absolute_times_are_rebased_before_float32_cast(pose_batch):
    pose, confidence, lengths = pose_batch
    cfg = ModelConfig(
        width=288,
        heads=4,
        rank=2,
        bins=4,
        detail_memory=True,
        detail_position_dim=16,
        body_frame="dual",
        timing=True,
        visual_backend="cnn",
        visual_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.0,
    )
    model = SignTranslator(cfg).eval()
    ts = torch.arange(7, dtype=torch.float64).expand(2, 7)
    rgb, mask = torch.rand(2, 7, 4, 3, 24, 24), torch.ones(2, 7, 4, dtype=torch.bool)
    a = model(pose, confidence, lengths, torch.ones(2, 2, dtype=torch.long), ts, rgb, mask)
    b = model(pose, confidence, lengths, torch.ones(2, 2, dtype=torch.long), ts + 1_700_000_000, rgb, mask)
    torch.testing.assert_close(a, b)
    with pytest.raises(ValueError, match="timestamps in seconds"):
        model(pose, confidence, lengths, torch.ones(2, 2, dtype=torch.long), rgb=rgb, rgb_mask=mask)


@pytest.fixture
def tiny_pretrained_assets(tmp_path):
    pytest.importorskip("transformers")
    from transformers import (
        T5Config,
        T5ForConditionalGeneration,
        PreTrainedTokenizerFast,
        Dinov2Config,
        Dinov2Model,
    )
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing

    path = tmp_path / "tiny_t5"
    path.mkdir()
    backend = Tokenizer(WordLevel({"<pad>": 0, "</s>": 1, "<unk>": 2, "hf": 3, "fh": 4}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>"
    )
    tokenizer.save_pretrained(path)
    T5ForConditionalGeneration(
        T5Config(
            vocab_size=5,
            d_model=16,
            d_ff=32,
            num_layers=1,
            num_decoder_layers=1,
            num_heads=2,
            d_kv=8,
            dropout_rate=0.0,
            decoder_start_token_id=0,
            pad_token_id=0,
            eos_token_id=1,
        )
    ).save_pretrained(path)
    vision = tmp_path / "tiny_dino"
    Dinov2Model(
        Dinov2Config(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            image_size=28,
            patch_size=14,
            mlp_ratio=2,
        )
    ).save_pretrained(vision)
    return path, vision


def test_hf_and_dino_weight_loading_backward_generation_and_portable_checkpoint(
    tmp_path, tiny_pretrained_assets, monkeypatch
):
    text, vision = tiny_pretrained_assets
    cfg = Config(
        visual_cfg(text_backend="hf", text_model=str(text), visual_model=str(vision)),
        DataConfig(max_frames=64, max_bytes=14),
        TrainConfig(),
    )
    cfg.model.visual_backend = "dinov2"
    model = SignTranslator(cfg.model).eval()
    root = make_synthetic(tmp_path / "data", samples=4)
    add_visual_data(root)
    # DINOv2 supports interpolation from 16x16 crops to its patch grid, too.
    ds = SignDataset(root / "train.jsonl", cfg.model, cfg.data, model.codec)
    batch = make_collator(model.codec)([ds[0], ds[1]])
    logits = model(**model_inputs(batch))
    assert logits.shape[-1] == 5
    logits.sum().backward()
    assert model.pretrained_decoder.projection.weight.grad.abs().sum() > 0
    ids = model.generate(**model_inputs(batch, decoder=False), max_new_tokens=4)
    assert ids.shape[0] == 2
    checkpoint = tmp_path / "model.pt"
    atomic_save(
        {
            "format_version": 3,
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "text_codec": codec_payload(model.codec),
        },
        checkpoint,
    )
    shutil.move(text, text.with_name("removed_text_source"))
    shutil.move(vision, vision.with_name("removed_vision_source"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    restored, _, _ = load_checkpoint(checkpoint)
    restored.eval()
    torch.testing.assert_close(logits, restored(**model_inputs(batch)))
    assert restored.codec.decode(restored.codec.encode("hf")) == "hf"


def test_real_corpus_importers_preserve_targets_and_splits(tmp_path):
    poses = tmp_path / "poses"
    poses.mkdir()
    np.savez(
        poses / "clip.npz",
        pose=np.zeros((2, 66, 3), np.float32),
        confidence=np.ones((2, 66), np.float32),
        timestamps=np.array([0.0, 1.0]),
    )
    # Reading/array validation is intentionally the separate validator's task.
    annotations = tmp_path / "phoenix.csv"
    annotations.write_text("name|translation|speaker\nclip|German target|S1\n")
    counts = import_corpus("phoenix2014t", annotations, poses, tmp_path / "p_out", split="dev")
    assert counts["val"] == 1
    record = json.loads((tmp_path / "p_out/val.jsonl").read_text())
    assert record["text"] == "German target" and record["signer"] == "S1"
    isl = tmp_path / "isl.csv"
    isl.write_text("uid,text\nclip,English target\n")
    split = tmp_path / "splits.json"
    split.write_text('{"clip":"test"}')
    assert import_corpus("isltranslate", isl, poses, tmp_path / "i_out", split_map=split)["test"] == 1
    csl = tmp_path / "csl.pkl"
    csl.write_bytes(pickle.dumps({"info": [{"name": "clip", "text": ["你", "好"], "split": "train"}]}))
    assert import_corpus("csldaily", csl, poses, tmp_path / "c_out")["train"] == 1
    assert json.loads((tmp_path / "c_out/train.jsonl").read_text())["text"] == "你好"
    with pytest.raises(ValueError, match="word-level"):
        import_corpus("cislr", isl, poses, tmp_path / "bad")


def test_pickle_metadata_rejects_objects(tmp_path):
    p = tmp_path / "metadata.pkl"
    p.write_bytes(pickle.dumps(Path("/tmp")))
    with pytest.raises(pickle.UnpicklingError):
        read_metadata(p, ",")


def test_pretrained_training_and_offline_epoch_resume(tmp_path, tiny_pretrained_assets, monkeypatch):
    text, _ = tiny_pretrained_assets
    root = make_synthetic(tmp_path / "data", samples=4)
    model_cfg = ModelConfig(
        width=16,
        heads=2,
        rank=1,
        bins=2,
        encoder_layers=1,
        decoder_layers=1,
        dropout=0.1,
        max_target_length=8,
        text_backend="hf",
        text_model=str(text),
    )
    data = DataConfig(
        train=str(root / "train.jsonl"),
        val=str(root / "val.jsonl"),
        test=str(root / "test.jsonl"),
        batch_size=2,
        max_frames=64,
        max_bytes=6,
    )
    cfg = Config(model_cfg, data, TrainConfig(epochs=2, warmup_steps=0, output_dir=str(tmp_path / "run")))
    full_path = train(cfg)
    _, _, full = load_checkpoint(full_path)
    cfg.train.output_dir = str(tmp_path / "resumed")
    path = train(cfg, stop_after_epoch=1)
    shutil.move(text, text.with_name("removed_pretrained"))
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    train(cfg, resume=path)
    model, _, state = load_checkpoint(path)
    assert state["epoch"] == 2 and model.codec.pad_id == 0 and model.codec.eos_id == 1
    assert not model.pretrained_decoder.model.get_encoder().block
    for name in full["model"]:
        torch.testing.assert_close(full["model"][name], state["model"][name], atol=0, rtol=0)
    assert full["best_val_nll"] == state["best_val_nll"]


def test_rgb_preparation_uses_named_frames_and_preserves_alignment(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    import importlib.util

    module_path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rgb.py"
    spec = importlib.util.spec_from_file_location("prepare_rgb_test", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frames = tmp_path / "frames"
    frames.mkdir()
    Image.new("RGB", (32, 16), color=(255, 0, 0)).save(frames / "frame2.png")
    Image.new("RGB", (32, 16), color=(0, 0, 255)).save(frames / "frame10.png")
    boxes = tmp_path / "boxes.npz"
    np.savez(
        boxes,
        boxes=np.tile([0.0, 0.0, 1.0, 1.0], (2, 4, 1)),
        rgb_mask=np.ones((2, 4), bool),
        timestamps=np.array([0.0, 0.04]),
        frame_names=np.array(["frame10.png", "frame2.png"]),
    )
    output = tmp_path / "rgb.npz"
    module.prepare(frames, boxes, output, 16)
    with np.load(output) as f:
        assert f["rgb"][0, 0, 2].sum() > f["rgb"][0, 0, 0].sum()
        assert f["frame_names"].tolist() == ["frame10.png", "frame2.png"]


def test_multimodal_ablation_configs_are_all_valid_without_downloads(tmp_path):
    import subprocess
    import sys
    from pcat_slt.config import load_config

    root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(root / "scripts/run_ablations.py"),
            "--config",
            str(root / "configs/vpcat.yaml"),
            "--output",
            str(tmp_path / "ablations"),
            "--seeds",
            "17",
        ],
        check=True,
        capture_output=True,
    )
    configs = list((tmp_path / "ablations/configs").glob("*.yaml"))
    assert len(configs) == 18
    for path in configs:
        load_config(path)
