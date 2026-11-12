from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isl_mega.checkpoint import load_checkpoint, save_checkpoint
from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, validate_split_disjointness, write_grouped_splits
from isl_mega.decoder import T5VisualDecoder
from isl_mega.losses import compute_ctc_batch
from isl_mega.model import ISLMEGA
from isl_mega.pose import estimate_torso_scale
from isl_mega.temporal import HierarchicalTemporalEncoder


REPO_ROOT = Path(__file__).parents[1]


def _load_script_module(rel_path: str, module_name: str):
    script = REPO_ROOT / rel_path
    spec = importlib.util.spec_from_file_location(module_name, script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_synthetic_dataset(root: Path) -> tuple[Path, dict[str, Path]]:
    manifests: dict[str, Path] = {}
    for split, items in (
        ("train", [("tr1", "src_tr1", "sig1"), ("tr2", "src_tr2", "sig2")]),
        ("valid", [("va1", "src_va1", "sig3")]),
        ("test", [("te1", "src_te1", "sig4")]),
    ):
        rows = []
        for uid, src_id, sig_id in items:
            t = 3
            ts = np.arange(t, dtype=np.float64) / 25.0
            frames = np.zeros((t, 16, 16, 3), dtype=np.uint8)
            pose_xy = np.full((t, 66, 2), 0.5, dtype=np.float32)
            pose_valid = np.ones((t, 66), dtype=bool)
            npz_path = root / f"{uid}.npz"
            np.savez_compressed(
                npz_path,
                frames=frames,
                video_timestamps=ts,
                pose_xy=pose_xy,
                pose_timestamps=ts,
                pose_valid=pose_valid,
            )
            rows.append(
                {
                    "uid": uid,
                    "npz": str(npz_path),
                    "translation": f"sample {uid}",
                    "source_id": src_id,
                    "signer_id": sig_id,
                }
            )
        mpath = root / f"{split}.jsonl"
        mpath.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        manifests[split] = mpath

    cfg_path = root / "cfg.yaml"
    cfg_path.write_text(
        f"vision:\n  image_size: 32\n  crop_size: 16\n  patch_size: 4\n  dim: 16\n  depth: 1\n  heads: 2\n"
        f"pose:\n  hidden_dim: 16\n  layers: 1\n  heads: 2\n"
        f"fusion:\n  dim: 16\n  num_heads: 2\n  layers: 1\n"
        f"temporal:\n  dim: 16\n  local_kernels: [3, 5]\n  local_dilations: [1, 1]\n  global_layers: 1\n  heads: 2\n  ffn_dim: 32\n"
        f"data:\n  train_manifest: {json.dumps(str(manifests['train']))}\n"
        f"  valid_manifest: {json.dumps(str(manifests['valid']))}\n"
        f"  test_manifest: {json.dumps(str(manifests['test']))}\n"
        f"  image_size: 32\n  crop_size: 16\n  max_target_tokens: 32\n"
        f"decoder:\n  max_target_tokens: 32\n",
        encoding="utf-8",
    )
    return cfg_path, manifests


def test_temporal_encoder_has_no_unused_trainable_parameters():
    for cfg_name, expected_adaptive in (("smoke.yaml", False), ("smoke_v2.yaml", True)):
        cfg = load_config(REPO_ROOT / "configs" / cfg_name)
        enc = HierarchicalTemporalEncoder(cfg.temporal).train()
        assert bool(cfg.temporal.adaptive_consolidation) is expected_adaptive
        if expected_adaptive:
            assert enc.detail is None
            assert enc.adaptive_detail is not None
        else:
            assert enc.detail is not None
            assert enc.adaptive_detail is None
            assert enc.salience_net is None
            assert enc.log_slot_sigma is None
            assert enc.slot_spread_proj is None

        x = torch.randn(2, 6, cfg.temporal.dim, requires_grad=True)
        lengths = torch.tensor([6, 3])
        times = torch.arange(6).float()[None].repeat(2, 1) / 25.0
        out, _, aux = enc(x, lengths, times)
        loss = out.square().mean() + aux["retrieved_global"].square().mean()
        loss.backward()
        missing_grads = [
            name for name, param in enc.named_parameters()
            if param.requires_grad and param.grad is None
        ]
        assert not missing_grads, f"Unused trainable parameters in {cfg_name}: {missing_grads}"
        assert all(torch.isfinite(p.grad).all() for p in enc.parameters() if p.requires_grad)


def test_fp16_sqrt_zero_delta_and_pose_coords_no_nan():
    # Equal shoulder coordinates produce squared difference 0; float16 1e-8 underflows without .float() guard.
    xy = torch.full((1, 3, 66, 2), 0.5, dtype=torch.float16, requires_grad=True)
    valid = torch.ones((1, 3, 66), dtype=torch.bool)
    scale = estimate_torso_scale(xy, valid)
    assert torch.isfinite(scale).all()
    scale.sum().backward()
    assert xy.grad is not None and torch.isfinite(xy.grad).all()

    if torch.cuda.is_available():
        cfg = load_config(REPO_ROOT / "configs" / "smoke_v2.yaml").temporal
        enc = HierarchicalTemporalEncoder(cfg).cuda().train()
        # Repeated identical frames force delta == 0 while gradients must remain finite under fp16 autocast.
        x = torch.ones((1, 4, cfg.dim), device="cuda", requires_grad=True)
        lengths = torch.tensor([4], device="cuda")
        times = torch.arange(4, device="cuda").float()[None] / 25.0
        with torch.autocast("cuda", dtype=torch.float16):
            y, _, _ = enc(x, lengths, times)
            loss = y.sum()
        loss.backward()
        assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in enc.parameters())


def test_compute_ctc_batch_respects_span_constraint_and_label_known():
    logits = torch.randn(1, 10, 40, requires_grad=True)
    lengths = torch.tensor([10])
    avail = torch.ones(1, 10, dtype=torch.bool)

    # Localized span covers only 2 frames for a 4-char target -> must skip rather than supervise whole clip.
    short_span = torch.tensor([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    loss_short = compute_ctc_batch(
        logits, lengths, ["abcd"], avail, span_targets=short_span, span_known=torch.tensor([True])
    )
    assert loss_short.item() == 0.0

    # Explicit empty annotated span must likewise skip even if full clip has >= 4 frames.
    empty_span = torch.zeros(1, 10, dtype=torch.bool)
    loss_empty = compute_ctc_batch(
        logits, lengths, ["abcd"], avail, span_targets=empty_span, span_known=torch.tensor([True])
    )
    assert loss_empty.item() == 0.0

    # Unknown label annotation must be skipped when label_known=False.
    loss_unknown = compute_ctc_batch(
        logits, lengths, ["abcd"], avail, label_known=torch.tensor([False])
    )
    assert loss_unknown.item() == 0.0

    loss_known = compute_ctc_batch(
        logits, lengths, ["abcd"], avail, label_known=torch.tensor([True])
    )
    assert loss_known.item() > 0.0


def test_checkpoint_save_load_restores_scaler_and_converts_cuda_rng_to_cpu(tmp_path, monkeypatch):
    model = torch.nn.Linear(4, 2)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    class FakeScaler:
        def __init__(self, scale: float):
            self.scale = scale

        def is_enabled(self) -> bool:
            return True

        def state_dict(self) -> dict:
            return {"scale": self.scale}

        def load_state_dict(self, state: dict) -> None:
            self.scale = float(state["scale"])

    ckpt = tmp_path / "ckpt.pt"
    save_checkpoint(ckpt, model, optimizer=opt, epoch=1, step=5, scaler=FakeScaler(1024.0))

    # Inject mock GPU-located cuda_rng tensors to verify load_checkpoint converts them via .cpu().
    raw = torch.load(ckpt, map_location="cpu", weights_only=False)

    class FakeCudaByteTensor:
        def __init__(self, tensor: torch.Tensor):
            self._tensor = tensor

        def cpu(self) -> torch.Tensor:
            return self._tensor

    rng_cpu = torch.get_rng_state()
    raw["cuda_rng"] = [FakeCudaByteTensor(rng_cpu), FakeCudaByteTensor(rng_cpu)]
    monkeypatch.setattr(torch, "load", lambda *a, **kw: raw)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    captured = {}

    def fake_set_rng_state(state, device=None):
        assert isinstance(state, torch.Tensor) and state.device.type == "cpu"
        captured["state"] = state
        captured["device"] = device

    monkeypatch.setattr(torch.cuda, "set_rng_state", fake_set_rng_state)
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda states: (_ for _ in ()).throw(AssertionError("count mismatch")))

    restored_scaler = FakeScaler(1.0)
    load_checkpoint(ckpt, model, optimizer=opt, map_location="cpu", scaler=restored_scaler)
    assert restored_scaler.scale == 1024.0
    assert captured["device"] == 0
    assert torch.equal(captured["state"], rng_cpu)


def test_train_epoch_passes_label_known_and_allows_amp_scaler_overflow():
    train_mod = _load_script_module("scripts/train.py", "isl_mega_train_fix_test")
    cfg = load_config(REPO_ROOT / "configs" / "smoke_v2.yaml")

    class DummyEncoder(torch.nn.Module):
        def __init__(self, dim=64, vocab=40):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab, dim)

        def forward(self, input_ids=None, attention_mask=None, return_dict=True, inputs_embeds=None):
            return SimpleNamespace(last_hidden_state=inputs_embeds if inputs_embeds is not None else self.embedding(input_ids))

    class DummyT5(torch.nn.Module):
        def __init__(self, dim=64, vocab=40):
            super().__init__()
            self.config = SimpleNamespace(d_model=dim, pad_token_id=0)
            self.encoder = DummyEncoder(dim, vocab)
            self.proj = torch.nn.Linear(dim, dim)
            self.output = torch.nn.Linear(dim, vocab)

        def prepare_decoder_input_ids_from_labels(self, labels):
            shifted = torch.zeros_like(labels)
            shifted[:, 1:] = labels[:, :-1].clamp_min(0)
            return shifted

        def forward(self, inputs_embeds, attention_mask, labels=None, decoder_input_ids=None,
                    decoder_attention_mask=None, use_cache=False, return_dict=True):
            if decoder_input_ids is None:
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
            pooled = inputs_embeds.mean(dim=1, keepdim=True)
            hidden = self.encoder.embedding(decoder_input_ids) + self.proj(pooled)
            logits = self.output(hidden)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100)
            return SimpleNamespace(loss=loss, logits=logits, encoder_last_hidden_state=inputs_embeds)

    class MockTokenizer:
        def __call__(self, texts, padding=True, truncation=False, return_tensors="pt"):
            ids = torch.tensor([[1, 2, 3, 4] for _ in texts], dtype=torch.long)
            return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    class OverflowSimulatingScaler:
        def __init__(self):
            self.unscaled = False
            self.stepped = False
            self.updated = False

        def is_enabled(self) -> bool:
            return True

        def scale(self, loss: torch.Tensor) -> torch.Tensor:
            return loss * 65536.0

        def unscale_(self, optimizer) -> None:
            self.unscaled = True
            # Inject an inf grad on one parameter to simulate normal fp16 overflow.
            for group in optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is not None:
                        p.grad.fill_(float("inf"))
                        break
                break

        def step(self, optimizer) -> None:
            self.stepped = True

        def update(self) -> None:
            self.updated = True

    model = ISLMEGA(cfg, decoder=T5VisualDecoder(DummyT5(dim=cfg.temporal.dim), cfg.temporal.dim))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = OverflowSimulatingScaler()
    batch = {
        "uids": ["u1"],
        "source_ids": ["s1"],
        "signer_ids": ["g1"],
        "frames": torch.rand(1, 4, 3, 64, 64),
        "timestamps": torch.arange(4).float()[None] / 25.0,
        "lengths": torch.tensor([4]),
        "frame_mask": torch.ones(1, 4, dtype=torch.bool),
        "pose_xy": torch.full((1, 4, 66, 2), 0.5),
        "pose_confidence": torch.ones(1, 4, 66),
        "pose_valid": torch.ones(1, 4, 66, dtype=torch.bool),
        "confidence_known": torch.ones(1, 4, 66, dtype=torch.bool),
        "boxes": torch.tensor([0.1, 0.1, 0.9, 0.9]).view(1, 1, 1, 4).expand(1, 4, 4, 4).clone(),
        "box_valid": torch.ones(1, 4, 4, dtype=torch.bool),
        "box_scores": torch.ones(1, 4, 4),
        "translations": ["hello world"],
        "fingerspelling_texts": ["ab"],
        "fingerspelling_spans": torch.zeros(1, 4, dtype=torch.bool),
        "fingerspelling_span_known": torch.tensor([False]),
        "fingerspelling_label_known": torch.tensor([False]),
    }
    stats = train_mod.run_epoch(
        model, [batch], MockTokenizer(), cfg, torch.device("cpu"), optimizer=opt, scaler=scaler
    )
    assert stats["ctc"] == 0.0
    assert scaler.unscaled and scaler.stepped and scaler.updated


def test_evaluate_build_split_records_handles_valid_and_train_without_self_collision(tmp_path):
    eval_mod = _load_script_module("scripts/evaluate.py", "isl_mega_eval_fix_test")
    cfg_path, manifests = _write_synthetic_dataset(tmp_path)
    cfg = load_config(cfg_path)

    for split in ("valid", "train", "test"):
        ds = ISLVideoDataset(manifests[split], cfg.data.crop_size, cfg.data.image_size)
        split_records = eval_mod.build_eval_split_records(manifests[split], cfg, [vars(r) for r in ds.records])
        validate_split_disjointness(split_records, "source_id")
        validate_split_disjointness(split_records, "signer_id", require_signer_disjoint=True)

    # Custom manifest with leaked source_id from train must still be rejected.
    leaked = tmp_path / "leaked.jsonl"
    leaked.write_text(json.dumps({
        "uid": "ext1", "npz": str(tmp_path / "va1.npz"), "translation": "leak", "source_id": "src_tr1", "signer_id": "sig9"
    }) + "\n", encoding="utf-8")
    ds_leaked = ISLVideoDataset(leaked, cfg.data.crop_size, cfg.data.image_size)
    bad_records = eval_mod.build_eval_split_records(leaked, cfg, [vars(r) for r in ds_leaked.records])
    with pytest.raises(ValueError, match="source_id leakage"):
        validate_split_disjointness(bad_records, "source_id")


def test_validate_data_uses_canonical_body_slice_42_to_48(tmp_path, monkeypatch):
    val_mod = _load_script_module("scripts/validate_data.py", "isl_mega_validate_fix_test")
    t = 4
    ts = np.arange(t, dtype=np.float64) / 25.0
    pose_valid = np.zeros((t, 66), dtype=bool)
    pose_valid[:, 0:42] = True  # Both hands valid; upper body (42:48) and face (48:66) missing.
    npz_path = tmp_path / "hands_only.npz"
    np.savez_compressed(
        npz_path,
        frames=np.zeros((t, 16, 16, 3), dtype=np.uint8),
        video_timestamps=ts,
        pose_xy=np.full((t, 66, 2), 0.5, dtype=np.float32),
        pose_timestamps=ts,
        pose_valid=pose_valid,
    )
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({
        "uid": "u1", "npz": str(npz_path), "translation": "hi", "source_id": "s1"
    }) + "\n", encoding="utf-8")
    out_json = tmp_path / "report.json"
    monkeypatch.setattr("sys.argv", ["validate_data.py", str(manifest), "--out", str(out_json)])
    val_mod.main()
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["mean_region_pose_coverage"]["left_hand"] == pytest.approx(1.0)
    assert report["mean_region_pose_coverage"]["right_hand"] == pytest.approx(1.0)
    assert report["mean_region_pose_coverage"]["body"] == pytest.approx(0.0)
    assert report["mean_region_pose_coverage"]["face"] == pytest.approx(0.0)


def test_pretrain_visual_reports_crop_weighted_epoch_loss(tmp_path, monkeypatch, capsys):
    from isl_mega.pretraining import SignAwareDistillation

    pretrain_mod = _load_script_module("scripts/pretrain_visual.py", "isl_mega_pretrain_fix_test")
    monkeypatch.setattr(
        pretrain_mod,
        "SignAwareDistillation",
        lambda vision_cfg: SignAwareDistillation(vision_cfg, prototypes=32),
    )
    rows = []
    for uid, t, src in (("u1", 2, "s1"), ("u2", 3, "s2")):
        ts = np.arange(t, dtype=np.float64) / 25.0
        npz_path = tmp_path / f"{uid}.npz"
        np.savez_compressed(
            npz_path,
            frames=np.zeros((t, 16, 16, 3), dtype=np.uint8),
            video_timestamps=ts,
            pose_xy=np.full((t, 66, 2), 0.5, dtype=np.float32),
            pose_timestamps=ts,
        )
        rows.append({"uid": uid, "npz": str(npz_path), "source_id": src})
    manifest = tmp_path / "ssl.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    cfg_path, _ = _write_synthetic_dataset(tmp_path)
    out_ckpt = tmp_path / "ssl.pt"
    monkeypatch.setattr(
        "sys.argv",
        ["pretrain_visual.py", "--config", str(cfg_path), "--manifest", str(manifest),
         "--out", str(out_ckpt), "--epochs", "1", "--batch-size", "1",
         "--allow-external-manifest"],
    )
    pretrain_mod.main()
    captured = capsys.readouterr().out
    assert "crops=" in captured
    assert out_ckpt.exists()


def test_dataset_rejects_invalid_valid_annotations_and_non_boolean_provenance(tmp_path):
    ts = np.arange(2, dtype=np.float64) / 25.0
    npz_path = tmp_path / "bad.npz"
    pose_xy = np.full((2, 66, 2), 0.5, dtype=np.float32)
    pose_xy[0, 0, 0] = 1.5
    np.savez_compressed(
        npz_path,
        frames=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        video_timestamps=ts,
        pose_xy=pose_xy,
        pose_timestamps=ts,
        pose_valid=np.ones((2, 66), dtype=bool),
    )
    manifest = tmp_path / "bad.jsonl"
    manifest.write_text(json.dumps({
        "uid": "bad", "npz": str(npz_path), "translation": "x", "source_id": "s",
        "fingerspelling_label_known": "false",
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be boolean"):
        ISLVideoDataset(manifest, crop_size=8, image_size=8)

    manifest.write_text(json.dumps({
        "uid": "bad", "npz": str(npz_path), "translation": "x", "source_id": "s",
        "fingerspelling_label_known": False,
    }) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        ISLVideoDataset(manifest, crop_size=8, image_size=8)[0]


def test_grouped_split_manifests_use_relocatable_npz_paths(tmp_path):
    rows = []
    for index in range(3):
        npz_path = tmp_path / f"sample_{index}.npz"
        np.savez_compressed(
            npz_path,
            frames=np.zeros((1, 8, 8, 3), dtype=np.uint8),
            video_timestamps=np.array([0.0]),
            pose_xy=np.full((1, 66, 2), 0.5, dtype=np.float32),
            pose_timestamps=np.array([0.0]),
        )
        rows.append({"uid": f"u{index}", "npz": str(npz_path), "source_id": f"source_{index}"})
    output = tmp_path / "manifests"
    write_grouped_splits(rows, output, seed=7)
    for split in ("train", "valid", "test"):
        row = json.loads((output / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()[0])
        assert not Path(row["npz"]).is_absolute()
        assert (output / row["npz"]).exists()
