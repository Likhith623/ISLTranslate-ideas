from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest
import torch

from isl_mega.config import load_config
from isl_mega.data import (align_pose_to_video, derive_rois, ISLVideoDataset,
                           collate_examples, validate_split_disjointness, write_grouped_splits,
                           letterbox_frames, transform_points_to_letterbox,
                           transform_boxes_to_letterbox)
from isl_mega.vision import crop_regions, PatchVisionTransformer
from isl_mega.pose import PoseExpert
from isl_mega.fingerspelling import encode_characters, greedy_ctc_decode
from isl_mega.losses import compute_ctc_batch
from isl_mega.temporal import HierarchicalTemporalEncoder
from isl_mega.fusion import ReliabilityAwareFusion, STREAM_NAMES
from isl_mega.planning import LatentPlanGrounder
from isl_mega.model import build_model
from isl_mega.pretraining import SignAwareDistillation


def tiny_config():
    return load_config(Path(__file__).parents[1] / "configs" / "smoke.yaml")


def inputs(batch=2, time=7, image=64):
    torch.manual_seed(4)
    frames = torch.rand(batch, time, 3, image, image)
    xy = torch.rand(batch, time, 66, 2) * 0.8 + 0.1
    conf = torch.rand(batch, time, 66)
    valid = torch.rand(batch, time, 66) > 0.08
    times = torch.arange(time).float()[None].repeat(batch, 1) / 25
    lengths = torch.tensor([time, max(1, time - 2)])[:batch]
    boxes = torch.tensor([0.1, 0.1, 0.9, 0.9]).view(1, 1, 1, 4).expand(batch, time, 4, 4).clone()
    box_valid = torch.ones(batch, time, 4, dtype=torch.bool)
    scores = torch.ones(batch, time, 4)
    return dict(frames=frames, timestamps=times, lengths=lengths, pose_xy=xy,
                pose_confidence=conf, pose_valid=valid, confidence_known=torch.ones_like(valid),
                boxes=boxes, box_valid=box_valid, box_scores=scores)



def test_metric_results_record_signatures_and_software_version(monkeypatch):
    import sys
    from types import SimpleNamespace
    from isl_mega.metrics import translation_metrics

    class Score:
        def __init__(self, score):
            self.score = score
        def get_signature(self):
            return SimpleNamespace(format=lambda: "mock-signature")

    fake = SimpleNamespace(
        __version__="mock-1.0",
        corpus_bleu=lambda hypotheses, refs: Score(12.0),
        corpus_chrf=lambda hypotheses, refs, word_order=2: Score(33.0),
        corpus_ter=lambda hypotheses, refs: Score(55.0),
    )
    monkeypatch.setitem(sys.modules, "sacrebleu", fake)
    result = translation_metrics(["hello"], ["hello"])
    assert result["num_examples"] == 1
    assert result["sacrebleu_version"] == "mock-1.0"
    assert result["bleu_signature"] == "mock-signature"
    assert result["chrf_signature"] == "mock-signature"
    assert result["ter_signature"] == "mock-signature"

def test_smoke_and_research_configs_validate():
    tiny_config()
    research = load_config(Path(__file__).parents[1] / "configs" / "research.yaml")
    assert research.temporal.dim == 1024
    assert research.decoder.model_name == "google/flan-t5-xl"


def test_pose_alignment_is_temporal_and_gap_safe():
    pt = np.array([0.0, 0.1, 0.2])
    vt = np.array([0.05, 0.15, 0.5])
    xy = np.zeros((3, 66, 2), np.float32)
    xy[0, :, 0] = 0.0; xy[1, :, 0] = 1.0; xy[2, :, 0] = 2.0
    conf = np.ones((3, 66), np.float32)
    valid = np.ones((3, 66), bool)
    out, _, mask = align_pose_to_video(vt, pt, xy, conf, valid, max_gap_seconds=0.11)
    assert out[0, 0, 0] == pytest.approx(0.5)
    assert out[1, 0, 0] == pytest.approx(1.5)
    assert not mask[2].any()


def test_timestamp_validation_rejects_nonmonotonic():
    with pytest.raises(ValueError):
        align_pose_to_video(np.array([0.1, 0.0]), np.array([0.0, 0.1]),
                            np.zeros((2, 66, 2)), np.ones((2, 66)),
                            np.ones((2, 66), bool))


def test_derive_rois_handles_missing_face_and_body_fallback():
    xy = np.random.rand(3, 66, 2).astype(np.float32)
    valid = np.ones((3, 66), bool)
    valid[:, 48:66] = False
    valid[:, 42:48] = False
    boxes, mask = derive_rois(xy, valid)
    assert not mask[:, 0].any()
    assert mask[:, 3].all()  # Full-frame fallback explicitly available for body stream.
    assert boxes[:, 3].tolist() == [[0.0, 0.0, 1.0, 1.0]] * 3


def test_crop_regions_shape_and_missing_mask():
    x = torch.rand(2, 3, 3, 20, 20)
    boxes = torch.tensor([0.1, 0.2, 0.8, 0.9]).view(1, 1, 1, 4).expand(2, 3, 4, 4)
    valid = torch.ones(2, 3, 4, dtype=torch.bool); valid[:, :, 0] = False
    crops = crop_regions(x, boxes, valid, 16)
    assert crops.shape == (2, 3, 4, 3, 16, 16)
    assert crops[:, :, 0].abs().sum() == 0
    assert crops[:, :, 1:].abs().sum() > 0


def test_pose_expert_shapes_and_all_missing_are_finite():
    cfg = tiny_config()
    expert = PoseExpert(cfg.pose)
    b, t = 2, 5
    xy = torch.rand(b, t, 66, 2)
    conf = torch.rand(b, t, 66)
    valid = torch.ones(b, t, 66, dtype=torch.bool)
    known = torch.ones_like(valid)
    times = torch.arange(t).float()[None].repeat(b, 1) / 30
    mask = torch.ones(b, t, dtype=torch.bool)
    out = expert(xy, conf, valid, known, times, mask)
    assert set(out["features"]) == {"left_hand", "right_hand", "body", "face"}
    assert out["features"]["left_hand"].shape == (b, t, cfg.pose.hidden_dim)
    missing = expert(xy, conf, torch.zeros_like(valid), known, times, mask)
    assert all(torch.isfinite(v).all() for v in missing["features"].values())
    assert not any(v.any() for v in missing["available"].values())



def test_body_relative_coordinates_use_shoulder_midpoint():
    from torch import nn
    cfg = tiny_config()
    expert = PoseExpert(cfg.pose)

    class Capture(nn.Module):
        def forward(self, features, xy, valid):
            self.features = features.detach().clone()
            return torch.zeros(features.shape[0], cfg.pose.hidden_dim), valid.any(-1)

    capture = Capture()
    expert.body = capture
    xy = torch.zeros(1, 1, 66, 2)
    xy[0, 0, 42] = torch.tensor([0.2, 0.3])  # left shoulder
    xy[0, 0, 43] = torch.tensor([0.6, 0.3])  # right shoulder
    xy[0, 0, 44] = torch.tensor([0.1, 0.8])  # left elbow should not move anchor
    valid = torch.zeros(1, 1, 66, dtype=torch.bool)
    valid[0, 0, 42:45] = True
    conf = valid.float()
    known = torch.ones_like(valid)
    times = torch.zeros(1, 1)
    mask = torch.ones(1, 1, dtype=torch.bool)
    expert(xy, conf, valid, known, times, mask)
    # Body relative coordinate channels are indices 2:4; shoulder midpoint=(0.4,0.3).
    got = capture.features[0, 0, :, 2:4]
    assert got[0].tolist() == pytest.approx([-0.2, 0.0])
    assert got[1].tolist() == pytest.approx([0.2, 0.0])
    assert got[2].tolist() == pytest.approx([-0.3, 0.5])

def test_fingerspelling_codec_and_ctc_loss():
    assert encode_characters("Ab-1") == [1, 2, 37, 28]
    logits = torch.randn(2, 8, 40, requires_grad=True)
    avail = torch.ones(2, 8, dtype=torch.bool)
    loss = compute_ctc_batch(logits, torch.tensor([8, 8]), ["ab", "c"], avail)
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert len(greedy_ctc_decode(logits.detach(), torch.tensor([8, 8]))) == 2


def test_ctc_skips_segments_without_enough_available_frames():
    logits = torch.randn(1, 5, 40)
    available = torch.tensor([[True, False, False, False, False]])
    loss = compute_ctc_batch(logits, torch.tensor([5]), ["this is too long"], available)
    assert loss.item() == 0.0


def test_temporal_encoder_retains_all_valid_frames():
    cfg = tiny_config()
    enc = HierarchicalTemporalEncoder(cfg.temporal).eval()
    x = torch.randn(2, 7, cfg.temporal.dim)
    lengths = torch.tensor([7, 5])
    times = torch.arange(7).float()[None].repeat(2, 1) / 30
    y, mask, aux = enc(x, lengths, times)
    assert y.shape == x.shape
    assert mask.sum(1).tolist() == [7, 5]
    assert aux["global"].shape[1] == 4


def test_temporal_padding_invariance():
    cfg = tiny_config()
    enc = HierarchicalTemporalEncoder(cfg.temporal).eval()
    torch.manual_seed(8)
    x = torch.randn(1, 5, cfg.temporal.dim)
    t = torch.arange(5).float()[None] / 30
    y1, _, _ = enc(x, torch.tensor([5]), t)
    xp = torch.cat([x, torch.zeros(1, 2, cfg.temporal.dim)], dim=1)
    tp = torch.cat([t, torch.zeros(1, 2)], dim=1)
    y2, _, _ = enc(xp, torch.tensor([5]), tp)
    assert torch.allclose(y1, y2[:, :5], atol=1e-5, rtol=1e-5)


def test_fusion_all_missing_is_finite_and_weights_zero():
    cfg = tiny_config()
    dims = {name: 32 for name in STREAM_NAMES}
    fusion = ReliabilityAwareFusion(dims, cfg.fusion).eval()
    b, t = 2, 3
    streams = {n: torch.randn(b, t, 32) for n in STREAM_NAMES}
    avail = {n: torch.zeros(b, t, dtype=torch.bool) for n in STREAM_NAMES}
    quality = {n: torch.zeros(b, t, 5) for n in STREAM_NAMES}
    out, weights, _, _ = fusion(streams, avail, quality)
    assert torch.isfinite(out).all()
    assert torch.isfinite(weights).all()
    assert weights.sum().item() == 0.0


def test_fusion_learns_small_time_lag_attention():
    cfg = tiny_config()
    fusion = ReliabilityAwareFusion({n: 32 for n in STREAM_NAMES}, cfg.fusion)
    b, t = 1, 4
    streams = {n: torch.randn(b, t, 32) for n in STREAM_NAMES}
    avail = {n: torch.ones(b, t, dtype=torch.bool) for n in STREAM_NAMES}
    quality = {n: torch.ones(b, t, 5) for n in STREAM_NAMES}
    out, weights, _, _ = fusion(streams, avail, quality)
    assert out.shape == (b, t, cfg.fusion.dim)
    assert torch.allclose(weights.sum(-1), torch.ones(b, t), atol=1e-5)



def test_source_aware_cross_modal_alignment_masks_same_source_negatives():
    from isl_mega.losses import compute_cross_modal_alignment_loss

    torch.manual_seed(25)
    interactions = torch.randn(2, 4, 10, 16, requires_grad=True)
    available = torch.ones(2, 4, 10, dtype=torch.bool)
    loss = compute_cross_modal_alignment_loss(interactions, available, ["source-a", "source-b"], 0.1)
    assert torch.isfinite(loss) and loss.item() > 0
    loss.backward()
    assert interactions.grad is not None and torch.isfinite(interactions.grad).all()

    # With one source only, other frames are not used as false negatives.
    same_source = compute_cross_modal_alignment_loss(interactions.detach(), available,
                                                     ["source-a", "source-a"], 0.1)
    assert same_source.item() == 0.0

def test_planner_returns_evidence_attention():
    cfg = tiny_config()
    planner = LatentPlanGrounder(cfg.temporal.dim, 4, 1, cfg.temporal.heads, 0.0)
    memory = torch.randn(2, 6, cfg.temporal.dim)
    mask = torch.tensor([[1]*6, [1,1,1,0,0,0]], dtype=torch.bool)
    plan, attn = planner(memory, mask)
    assert plan.shape == (2, 4, cfg.temporal.dim)
    assert attn.shape == (2, 4, 6)
    assert attn[1, :, 3:].abs().sum() == 0


def test_full_model_forward_backward_and_frame_retention():
    cfg = tiny_config()
    model = build_model(cfg, with_decoder=False)
    x = inputs()
    out = model(**x)
    assert out["memory"].shape == (2, 7, cfg.temporal.dim)
    assert out["memory_mask"].sum(1).tolist() == [7, 5]
    assert out["plan_tokens"].shape == (2, cfg.temporal.planner_queries, cfg.temporal.dim)
    assert out["grounding_attention"].shape == (2, cfg.temporal.planner_queries, 7)
    assert out["ctc_logits"].shape[:2] == (2, 7)
    loss = out["memory"].square().mean() + out["ctc_logits"].square().mean()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_full_model_all_modalities_missing_is_finite():
    cfg = tiny_config()
    model = build_model(cfg, with_decoder=False).eval()
    x = inputs(batch=1, time=5)
    x["box_valid"].zero_(); x["pose_valid"].zero_()
    with torch.no_grad():
        out = model(**x)
    assert torch.isfinite(out["memory"]).all()
    assert torch.isfinite(out["plan_tokens"]).all()


def test_full_model_padding_invariance():
    cfg = tiny_config()
    model = build_model(cfg, with_decoder=False).eval()
    base = inputs(batch=1, time=5)
    with torch.no_grad():
        a = model(**base)["memory"]
    padded = inputs(batch=1, time=7)
    for key in ("frames", "timestamps", "pose_xy", "pose_confidence", "pose_valid", "confidence_known", "boxes", "box_valid", "box_scores"):
        padded[key][:, :5] = base[key]
    padded["frames"][:, 5:] = 0
    padded["timestamps"][:, 5:] = 0
    padded["pose_xy"][:, 5:] = 0
    padded["pose_confidence"][:, 5:] = 0
    padded["pose_valid"][:, 5:] = False
    padded["confidence_known"][:, 5:] = False
    padded["boxes"][:, 5:] = 0
    padded["box_valid"][:, 5:] = False
    padded["box_scores"][:, 5:] = 0
    padded["lengths"] = torch.tensor([5])
    with torch.no_grad():
        b = model(**padded)["memory"]
    assert torch.allclose(a, b[:, :5], atol=2e-5, rtol=2e-5)


def test_dataset_npz_manifest_and_collate(tmp_path):
    rows = []
    for i, t in enumerate([5, 7]):
        ts = 100.0 + np.arange(t, dtype=np.float32) / 25
        frames = np.zeros((t, 32, 32, 3), np.uint8)
        xy = np.random.default_rng(i).uniform(0.2, 0.8, (t, 66, 2)).astype(np.float32)
        np.savez_compressed(tmp_path / f"{i}.npz", frames=frames, video_timestamps=ts,
                            pose_xy=xy, pose_timestamps=ts)
        rows.append({"uid": str(i), "npz": f"{i}.npz", "translation": "hello", "source_id": f"s{i}"})
    manifest = tmp_path / "all.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows))
    ds = ISLVideoDataset(manifest, crop_size=32, image_size=64)
    batch = collate_examples([ds[0], ds[1]])
    assert batch["frames"].shape == (2, 7, 3, 64, 64)
    assert batch["frame_mask"].sum(1).tolist() == [5, 7]
    assert batch["pose_xy"].shape == (2, 7, 66, 2)
    assert batch["timestamps"][0, 0].item() == pytest.approx(0.0)



def test_split_validation_rejects_duplicate_uids_across_different_sources():
    splits = {
        "train": [{"uid": "same-clip", "source_id": "source-a"}],
        "valid": [{"uid": "same-clip", "source_id": "source-b"}],
    }
    with pytest.raises(ValueError, match="Duplicate uid"):
        validate_split_disjointness(splits)

def test_grouped_splits_are_source_disjoint(tmp_path):
    records = [{"uid": str(i), "source_id": f"src{i}", "npz": "x", "translation": "y"}
               for i in range(30)]
    splits = write_grouped_splits(records, tmp_path, seed=5)
    validate_split_disjointness(splits, "source_id")
    with pytest.raises(ValueError):
        validate_split_disjointness({"train": [records[0]], "test": [records[0]]}, "source_id")


def test_signaware_pretraining_forward_ema_and_backprop():
    cfg = tiny_config()
    ssl = SignAwareDistillation(cfg.vision, prototypes=64)
    ssl.train()
    global_frames = torch.rand(2, 3, 64, 64)
    local = torch.rand(2, 3, 3, 32, 32)
    valid = torch.tensor([[1,1,1], [1,0,1]], dtype=torch.bool)
    loss, center = ssl(global_frames, local, valid)
    loss.backward()
    ssl.update_teacher(0.9)
    ssl.update_center(center)
    assert torch.isfinite(loss)
    assert torch.isfinite(ssl.center).all()
    assert not ssl.teacher.training


def test_temporal_encoder_respects_nondefault_coarse_stride():
    cfg = tiny_config().temporal
    cfg.coarse_stride = 3
    enc = HierarchicalTemporalEncoder(cfg).eval()
    x = torch.randn(2, 8, cfg.dim)
    lengths = torch.tensor([8, 5])
    times = torch.arange(8).float()[None].repeat(2, 1) / 25
    y, mask, aux = enc(x, lengths, times)
    assert y.shape == x.shape
    assert mask.sum(1).tolist() == [8, 5]
    assert aux["global"].shape[1] == 3


def test_motion_expert_is_3d_and_keeps_all_frames_and_padding_invariant():
    from isl_mega.motion import MotionExpert
    cfg = tiny_config().motion
    expert = MotionExpert(cfg).eval()
    torch.manual_seed(101)
    crops = torch.rand(1, 5, 1, 3, 32, 32)
    mask = torch.ones(1, 5, dtype=torch.bool)
    a = expert(crops, mask)
    padded = torch.cat([crops, torch.zeros(1, 2, 1, 3, 32, 32)], dim=1)
    pad_mask = torch.tensor([[True, True, True, True, True, False, False]])
    b = expert(padded, pad_mask)
    assert a.shape == (1, 5, cfg.hidden_dim)
    assert torch.allclose(a, b[:, :5], atol=1e-5, rtol=1e-5)


def test_pose_confidence_threshold_respects_confidence_known_mask():
    cfg = tiny_config().pose
    cfg.min_confidence = 0.8
    expert = PoseExpert(cfg).eval()
    xy = torch.rand(1, 2, 66, 2)
    confidence = torch.full((1, 2, 66), 0.2)
    valid = torch.ones(1, 2, 66, dtype=torch.bool)
    times = torch.tensor([[0.0, 0.04]])
    frame_mask = torch.ones(1, 2, dtype=torch.bool)
    known = torch.ones_like(valid)
    known_result = expert(xy, confidence, valid, known, times, frame_mask)
    assert not known_result["available"]["left_hand"].any()
    unknown_result = expert(xy, confidence, valid, torch.zeros_like(known), times, frame_mask)
    assert unknown_result["available"]["left_hand"].all()


def test_roi_quality_is_per_region_and_handles_missing_boxes():
    from isl_mega.vision import roi_quality
    crops = torch.zeros(1, 2, 4, 3, 12, 12)
    # A checkerboard-like crop has more edge energy than a constant crop.
    pattern = (torch.arange(12)[None, :] % 2).float().expand(12, 12)
    crops[0, 0, 1] = pattern[None].expand(3, -1, -1)
    boxes = torch.tensor([0.0, 0.0, 1.0, 1.0]).view(1, 1, 1, 4).expand(1, 2, 4, 4)
    valid = torch.ones(1, 2, 4, dtype=torch.bool)
    valid[:, :, 0] = False
    q = roi_quality(crops, boxes, valid)
    assert q.shape == (1, 2, 4, 5)
    assert q[0, 0, 1, 2] > q[0, 0, 2, 2]
    assert q[:, :, 0, 0].sum() == 0


def test_config_parser_rejects_unknown_keys(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("vision:\n  dim: 64\n  typo_dim: 128\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown VisionConfig"):
        load_config(bad)


def test_signer_disjoint_validation_requires_complete_signer_ids():
    splits = {
        "train": [{"uid": "u1", "source_id": "s1", "signer_id": "a"}],
        "valid": [{"uid": "u2", "source_id": "s2"}],
    }
    with pytest.raises(ValueError, match="missing signer_id"):
        validate_split_disjointness(splits, require_signer_disjoint=True)


def test_fingerspelling_codec_rejects_silent_character_loss():
    with pytest.raises(ValueError, match="Unsupported"):
        encode_characters("नाम")


def test_ctc_batch_skips_sequences_that_cannot_separate_repeated_characters():
    logits = torch.randn(1, 2, 40, requires_grad=True)
    availability = torch.ones(1, 2, dtype=torch.bool)
    loss = compute_ctc_batch(logits, torch.tensor([2]), ["aa"], availability)
    assert loss.item() == 0.0
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()


def test_epoch_scheduler_is_stepped_per_epoch():
    import importlib.util
    script = Path(__file__).parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("isl_mega_train_script", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    param = torch.nn.Parameter(torch.ones(()))
    optimizer = torch.optim.SGD([param], lr=1.0)
    scheduler = module.build_epoch_scheduler(optimizer, epochs=4, warmup_ratio=0.25)
    values = [optimizer.param_groups[0]["lr"]]
    for _ in range(4):
        optimizer.step()
        scheduler.step()
        values.append(optimizer.param_groups[0]["lr"])
    assert values[0] == pytest.approx(1.0)
    assert values[-1] == pytest.approx(0.0)


def test_decoder_adapter_with_mock_t5_and_label_smoothing():
    from types import SimpleNamespace
    from isl_mega.decoder import T5VisualDecoder

    class DummyEncoder(torch.nn.Module):
        def __init__(self, dim=16, vocab=40):
            super().__init__()
            self.embedding = torch.nn.Embedding(vocab, dim)

        def forward(self, input_ids=None, attention_mask=None, return_dict=True, inputs_embeds=None):
            hidden = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
            return SimpleNamespace(last_hidden_state=hidden)

    class DummyT5(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(d_model=16, pad_token_id=0)
            self.encoder = DummyEncoder()
            self.output = torch.nn.Linear(16, 40)
            self.last_decoder_mask = None

        def prepare_decoder_input_ids_from_labels(self, labels):
            shifted = torch.zeros_like(labels)
            shifted[:, 1:] = labels[:, :-1].clamp_min(0)
            return shifted

        def forward(self, inputs_embeds, attention_mask, labels=None, decoder_input_ids=None,
                    decoder_attention_mask=None, use_cache=False, return_dict=True):
            if decoder_input_ids is None:
                decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
            self.last_decoder_mask = decoder_attention_mask
            hidden = self.encoder.embedding(decoder_input_ids)
            logits = self.output(hidden)
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
                )
            return SimpleNamespace(loss=loss, logits=logits,
                                   encoder_last_hidden_state=inputs_embeds)

        def generate(self, encoder_outputs, attention_mask, **kwargs):
            hidden = encoder_outputs.last_hidden_state
            return torch.ones(hidden.shape[0], 3, dtype=torch.long, device=hidden.device)

    wrapper = T5VisualDecoder(DummyT5(), visual_dim=8)
    visual = torch.randn(2, 5, 8)
    visual_mask = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 0, 0, 0]], dtype=torch.bool)
    plan = torch.randn(2, 2, 8)
    labels = torch.tensor([[1, 2, 3, -100], [4, 5, -100, -100]])
    standard = wrapper(visual, visual_mask, plan, labels)
    smoothed = wrapper(visual, visual_mask, plan, labels, label_smoothing=0.1, semantic_ot=True)
    assert torch.isfinite(standard["loss"])
    assert torch.isfinite(smoothed["loss"])
    assert torch.isfinite(smoothed["semantic_ot_loss"])
    # T5's start token commonly equals pad_token_id but must be visible.
    assert wrapper.model.last_decoder_mask[:, 0].tolist() == [1, 1]
    assert wrapper.model.last_decoder_mask[0].tolist() == [1, 1, 1, 1]
    assert wrapper.model.last_decoder_mask[1].tolist() == [1, 1, 1, 0]
    assert wrapper.generate(visual, visual_mask, plan).shape == (2, 3)


def test_letterboxing_preserves_aspect_and_transforms_pose_and_boxes():
    # 80x160 source -> 32x64 content in a 64x64 canvas, vertically centered.
    frames = torch.ones(2, 3, 80, 160)
    out, affine = letterbox_frames(frames, 64)
    assert out.shape == (2, 3, 64, 64)
    assert torch.all(out[:, :, :16] == 0)
    assert torch.all(out[:, :, 16:48] == 1)
    assert torch.all(out[:, :, 48:] == 0)
    point = np.array([[[0.5, 0.5]]], dtype=np.float32)
    transformed = transform_points_to_letterbox(point, affine)
    assert transformed[0, 0, 0] == pytest.approx(0.5)
    assert transformed[0, 0, 1] == pytest.approx(0.5)
    box = np.array([[[0.25, 0.25, 0.75, 0.75]]], dtype=np.float32)
    mapped = transform_boxes_to_letterbox(box, affine)
    assert mapped[0, 0].tolist() == pytest.approx([0.25, 0.375, 0.75, 0.625])


def test_unlabeled_dataset_manifest_is_accepted_for_ssl_pretraining(tmp_path):
    t = 4
    ts = np.arange(t, dtype=np.float64) / 25.0
    frames = np.zeros((t, 80, 160, 3), np.uint8)
    xy = np.full((t, 66, 2), 0.5, np.float32)
    np.savez_compressed(tmp_path / "unlabeled.npz", frames=frames, video_timestamps=ts,
                        pose_xy=xy, pose_timestamps=ts)
    manifest = tmp_path / "unlabeled.jsonl"
    manifest.write_text(json.dumps({"uid": "u1", "npz": "unlabeled.npz",
                                    "source_id": "source-1"}) + "\n", encoding="utf-8")
    ds = ISLVideoDataset(manifest, crop_size=32, image_size=64, require_translation=False)
    sample = ds[0]
    assert sample["translation"] == ""
    assert sample["frames"].shape == (t, 3, 64, 64)
    # A centered point remains centered after aspect-preserving letterboxing.
    assert sample["pose_xy"][0, 0].tolist() == pytest.approx([0.5, 0.5])


@pytest.mark.parametrize("frame_dtype,values", [
    (np.uint16, 255),
    (np.float32, 255.0),
])
def test_dataset_rejects_unsupported_or_out_of_range_frame_encoding(tmp_path, frame_dtype, values):
    t = 2
    ts = np.arange(t, dtype=np.float64) / 25.0
    frames = np.full((t, 16, 16, 3), values, dtype=frame_dtype)
    xy = np.full((t, 66, 2), 0.5, np.float32)
    np.savez_compressed(tmp_path / "bad_frames.npz", frames=frames, video_timestamps=ts,
                        pose_xy=xy, pose_timestamps=ts)
    manifest = tmp_path / "bad_frames.jsonl"
    manifest.write_text(json.dumps({"uid": "bad", "npz": "bad_frames.npz",
                                    "translation": "test", "source_id": "s"}) + "\n")
    with pytest.raises(ValueError, match="frame dtype|Floating-point frames"):
        ISLVideoDataset(manifest, crop_size=32, image_size=64)[0]



def test_dataset_scales_uint8_using_dtype_even_for_dark_frames(tmp_path):
    t = 2
    ts = np.arange(t, dtype=np.float64) / 25.0
    frames = np.ones((t, 16, 16, 3), dtype=np.uint8)
    xy = np.full((t, 66, 2), 0.5, np.float32)
    np.savez_compressed(tmp_path / "dark.npz", frames=frames, video_timestamps=ts,
                        pose_xy=xy, pose_timestamps=ts)
    manifest = tmp_path / "dark.jsonl"
    manifest.write_text(json.dumps({"uid": "dark", "npz": "dark.npz",
                                    "translation": "dark frame", "source_id": "dark-source"}) + "\n")
    sample = ISLVideoDataset(manifest, crop_size=32, image_size=64)[0]
    # Letterbox is square here, so every RGB output pixel is 1/255.
    assert sample["frames"].mean().item() == pytest.approx(1.0 / 255.0, abs=1e-6)

def test_dataset_rejects_null_or_empty_required_manifest_values(tmp_path):
    manifest = tmp_path / "invalid.jsonl"
    manifest.write_text(json.dumps({"uid": "x", "npz": "x.npz",
                                    "translation": "ok", "source_id": None}) + "\n")
    with pytest.raises(ValueError, match="source_id.*nonempty string"):
        ISLVideoDataset(manifest)

def test_dataset_requires_translation_for_supervised_training(tmp_path):
    t = 2
    ts = np.arange(t, dtype=np.float64) / 25.0
    frames = np.zeros((t, 16, 16, 3), np.uint8)
    xy = np.full((t, 66, 2), 0.5, np.float32)
    np.savez_compressed(tmp_path / "sample.npz", frames=frames, video_timestamps=ts,
                        pose_xy=xy, pose_timestamps=ts)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"uid": "x", "npz": "sample.npz", "source_id": "s"}) + "\n")
    with pytest.raises(ValueError, match="translation"):
        ISLVideoDataset(manifest)



def test_fingerspelling_expert_supports_single_hand_and_rejects_missing_both():
    from isl_mega.fingerspelling import FingerspellingExpert

    torch.manual_seed(11)
    expert = FingerspellingExpert(visual_dim=8, pose_dim=6, hidden_dim=16, layers=1, dropout=0.0).eval()
    b, t = 2, 5
    vr = torch.randn(b, t, 8)
    pr = torch.randn(b, t, 6)
    false = torch.zeros(b, t, dtype=torch.bool)
    left = torch.tensor([[1, 1, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    right = torch.tensor([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]], dtype=torch.bool)
    out = expert(vr, vr, pr, pr, torch.tensor([5, 5]), left, right)
    assert out["available"][0].tolist() == [True, True, False, False, False]
    assert not out["available"][1].any()
    assert out["features"][1].abs().sum() == 0
    assert torch.isfinite(out["logits"]).all()


def test_temporal_detail_centres_are_per_sample_for_partial_stride_groups():
    cfg = tiny_config().temporal
    cfg.coarse_stride = 3
    cfg.detail_window = 1
    enc = HierarchicalTemporalEncoder(cfg).eval()
    torch.manual_seed(19)
    # First sample ends in a full group; second ends in a partial group.
    x = torch.randn(2, 5, cfg.dim)
    times = torch.arange(5).float()[None].repeat(2, 1) / 25
    lengths = torch.tensor([5, 4])
    with torch.no_grad():
        _, _, aux = enc(x, lengths, times)
    # With stride=3, the second sample's final coarse group must retrieve frame 3,
    # rather than the nominal padded-batch centre frame 4.
    retrieved = enc.detail(aux["global"], aux["fine"], lengths)
    centres = torch.tensor([0, 3])
    assert retrieved.shape == (2, 2, cfg.dim)
    assert torch.isfinite(retrieved).all()
    # Padding the batch to a longer max length must not move either sample's centres.
    xp = torch.cat([x, torch.zeros(2, 1, cfg.dim)], dim=1)
    tp = torch.cat([times, torch.zeros(2, 1)], dim=1)
    with torch.no_grad():
        yp, _, _ = enc(xp, lengths, tp)
        y, _, _ = enc(x, lengths, times)
    assert torch.allclose(y, yp[:, :5], atol=2e-5, rtol=2e-5)

def test_temporal_partial_stride_group_uses_last_valid_position():
    cfg = tiny_config().temporal
    cfg.coarse_stride = 3
    enc = HierarchicalTemporalEncoder(cfg).eval()
    x = torch.randn(2, 5, cfg.dim)
    lengths = torch.tensor([5, 4])
    times = torch.arange(5).float()[None].repeat(2, 1) / 25
    with torch.no_grad():
        y, mask, aux = enc(x, lengths, times)
    assert y.shape == x.shape
    assert mask.sum(1).tolist() == [5, 4]
    assert aux["global"].shape[1] == 2
    assert torch.isfinite(aux["global"]).all()


def test_sign_ssl_checkpoint_loads_into_all_specialist_appearance_backbones(tmp_path):
    from isl_mega.pretraining import load_ssl_backbone
    cfg = tiny_config()
    ssl = SignAwareDistillation(cfg.vision, prototypes=32)
    ckpt = tmp_path / "ssl.pt"
    torch.save({"student_backbone": ssl.student.state_dict()}, ckpt)
    model = build_model(cfg, with_decoder=False)
    results = load_ssl_backbone(model, str(ckpt), strict=True)
    assert len(results) == 3
    source = ssl.student.state_dict()
    for target in (model.appearance.hand, model.appearance.face, model.appearance.body):
        for name, value in target.state_dict().items():
            assert torch.equal(value.cpu(), source[name].cpu())


def test_dataset_marks_full_frame_body_fallback_as_low_quality_proxy(tmp_path):
    t = 3
    ts = np.arange(t, dtype=np.float64) / 25.0
    frames = np.zeros((t, 32, 32, 3), np.uint8)
    xy = np.zeros((t, 66, 2), np.float32)
    valid = np.zeros((t, 66), bool)
    np.savez_compressed(tmp_path / "missing.npz", frames=frames, video_timestamps=ts,
                        pose_xy=xy, pose_timestamps=ts, pose_valid=valid)
    manifest = tmp_path / "missing.jsonl"
    manifest.write_text(json.dumps({"uid": "m", "npz": "missing.npz", "translation": "missing observations", 
                                    "source_id": "source-m"}) + "\n", encoding="utf-8")
    sample = ISLVideoDataset(manifest, crop_size=32, image_size=64)[0]
    assert sample["box_valid"][:, 3].all()
    assert torch.allclose(sample["boxes"][:, 3], torch.tensor([0., 0., 1., 1.]).expand(t, -1))
    assert torch.allclose(sample["box_scores"][:, 3], torch.full((t,), 0.25))
