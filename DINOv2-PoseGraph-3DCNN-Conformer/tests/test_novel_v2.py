from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, collate_examples
from isl_mega.decoder import T5VisualDecoder
from isl_mega.fingerspelling import FingerspellingExpert
from isl_mega.fusion import ReliabilityAwareFusion, STREAM_NAMES
from isl_mega.losses import (
    compute_ctc_batch,
    compute_fingerspelling_spotting_loss,
    compute_fusion_entropy_loss,
    compute_quality_calibration_loss,
)
from isl_mega.model import ISLMEGA, build_model
from isl_mega.planning import LatentPlanGrounder
from isl_mega.pose import PoseExpert, estimate_torso_scale
from isl_mega.temporal import HierarchicalTemporalEncoder


def v2_smoke_config():
    return load_config(Path(__file__).parents[1] / "configs" / "smoke_v2.yaml")


def test_quality_calibration_prefers_observed_reliability_and_masks_missing_streams():
    quality = torch.tensor([[[[1.0, 1.0], [0.1, 0.1], [0.0, 0.0]]]])
    available = torch.tensor([[[True, True, False]]])
    target = torch.tensor([[[0.9091, 0.0909, 0.0]]])
    uniform = torch.tensor([[[0.5, 0.5, 0.0]]])
    assert compute_quality_calibration_loss(target, quality, available).item() < compute_quality_calibration_loss(
        uniform, quality, available
    ).item()


def make_batch_inputs(batch: int = 2, time: int = 8, image: int = 64):
    torch.manual_seed(17)
    frames = torch.rand(batch, time, 3, image, image)
    xy = torch.rand(batch, time, 66, 2) * 0.6 + 0.2
    conf = torch.rand(batch, time, 66)
    valid = torch.rand(batch, time, 66) > 0.05
    times = torch.arange(time).float()[None].repeat(batch, 1) / 25.0
    lengths = torch.tensor([time, max(1, time - 3)])[:batch]
    boxes = torch.tensor([0.1, 0.1, 0.9, 0.9]).view(1, 1, 1, 4).expand(batch, time, 4, 4).clone()
    box_valid = torch.ones(batch, time, 4, dtype=torch.bool)
    scores = torch.ones(batch, time, 4)
    return dict(
        frames=frames,
        timestamps=times,
        lengths=lengths,
        pose_xy=xy,
        pose_confidence=conf,
        pose_valid=valid,
        confidence_known=torch.ones_like(valid),
        boxes=boxes,
        box_valid=box_valid,
        box_scores=scores,
    )


class MockT5Encoder(torch.nn.Module):
    def __init__(self, dim: int = 16, vocab: int = 40):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab, dim)

    def forward(self, input_ids=None, attention_mask=None, return_dict=True, inputs_embeds=None):
        hidden = inputs_embeds if inputs_embeds is not None else self.embedding(input_ids)
        return SimpleNamespace(last_hidden_state=hidden)


class MockT5(torch.nn.Module):
    def __init__(self, dim: int = 16, vocab: int = 40):
        super().__init__()
        self.config = SimpleNamespace(d_model=dim, pad_token_id=0)
        self.encoder = MockT5Encoder(dim, vocab)
        self.cross_proj = torch.nn.Linear(dim, dim)
        self.output = torch.nn.Linear(dim, vocab)

    def prepare_decoder_input_ids_from_labels(self, labels):
        shifted = torch.zeros_like(labels)
        shifted[:, 1:] = labels[:, :-1].clamp_min(0)
        return shifted

    def forward(
        self,
        inputs_embeds,
        attention_mask,
        labels=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        use_cache=False,
        return_dict=True,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = self.prepare_decoder_input_ids_from_labels(labels)
        # Pool visual encoder state weighted by mask so visual edits change decoder logits.
        mask = attention_mask.float()[..., None]
        pooled = (inputs_embeds * mask).sum(dim=1, keepdim=True) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        hidden = self.encoder.embedding(decoder_input_ids) + self.cross_proj(pooled)
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
            )
        return SimpleNamespace(loss=loss, logits=logits, encoder_last_hidden_state=inputs_embeds)

    def generate(self, encoder_outputs, attention_mask, **kwargs):
        hidden = encoder_outputs.last_hidden_state
        return torch.ones((hidden.shape[0], 4), dtype=torch.long, device=hidden.device)


def test_v2_and_ablation_configs_validate_and_reject_bad_flags(tmp_path):
    v2 = load_config(Path(__file__).parents[1] / "configs" / "research.yaml")
    v1 = load_config(Path(__file__).parents[1] / "configs" / "ablation_v1_baseline.yaml")
    v2_smoke = v2_smoke_config()
    assert v2.pose.cross_articulator_graph and v2.pose.normalize_by_shoulder_span and v2.pose.share_hand_encoder
    assert v2.fusion.continuous_time_lag and v2.fusion.reliability_uncertainty_gate
    assert v2.fingerspelling.span_routing and v2.temporal.adaptive_consolidation
    assert v2.temporal.diverse_ordered_planning and v2.temporal.dynamic_plan_routing
    assert v2.decoder.plan_semantic_ot and v2.decoder.use_counterfactual_grounding
    assert not v1.pose.cross_articulator_graph and not v1.temporal.adaptive_consolidation
    assert v2_smoke.train.counterfactual_weight > 0

    bad = tmp_path / "bad_v2.yaml"
    bad.write_text("temporal:\n  planner_queries: 2\n  min_plan_queries: 4\n", encoding="utf-8")
    with pytest.raises(ValueError, match="min_plan_queries"):
        load_config(bad)


def test_pose_v2_torso_normalization_shared_hands_and_cross_graph():
    cfg = v2_smoke_config().pose
    expert = PoseExpert(cfg).eval()
    assert expert.left is expert.right

    b, t = 1, 4
    xy = torch.rand(b, t, 66, 2) * 0.4 + 0.3
    xy[:, :, 42] = torch.tensor([0.3, 0.5])
    xy[:, :, 43] = torch.tensor([0.7, 0.5])
    valid = torch.ones(b, t, 66, dtype=torch.bool)
    scale = estimate_torso_scale(xy, valid)
    assert torch.allclose(scale, torch.full((b, t), 0.4), atol=1e-5)

    # Shoulder-missing fallback still yields finite bounded scale.
    no_shoulders = valid.clone()
    no_shoulders[:, :, 42:44] = False
    assert torch.isfinite(estimate_torso_scale(xy, no_shoulders)).all()

    conf = torch.ones(b, t, 66)
    known = torch.ones(b, t, 66, dtype=torch.bool)
    times = torch.arange(t).float()[None] / 25.0
    frame_mask = torch.ones(b, t, dtype=torch.bool)

    out_a = expert(xy, conf, valid, known, times, frame_mask)
    # Moving only the face landmark constellation alters cross-articulator hand features.
    xy_shifted_face = xy.clone()
    xy_shifted_face[:, :, 48:66] += torch.tensor([0.12, -0.10])
    out_b = expert(xy_shifted_face, conf, valid, known, times, frame_mask)
    assert not torch.allclose(out_a["features"]["left_hand"], out_b["features"]["left_hand"])

    # Invalid face/body nodes must not leak messages into valid hand nodes.
    valid_face_masked = valid.clone()
    valid_face_masked[:, :, 48:66] = False  # Face completely invalid.
    out_masked_a = expert(xy, conf, valid_face_masked, known, times, frame_mask)
    xy_shifted_invalid_face = xy.clone()
    xy_shifted_invalid_face[:, :, 48:66] += 5.0  # Dramatic coordinate shift on invalid nodes.
    out_masked_b = expert(xy_shifted_invalid_face, conf, valid_face_masked, known, times, frame_mask)
    assert torch.allclose(out_masked_a["features"]["left_hand"], out_masked_b["features"]["left_hand"], atol=1e-5)

    # First-step gradient verification: cross-articulator graph block parameters must receive nonzero gradients.
    train_expert = PoseExpert(cfg).train()
    out_train = train_expert(xy, conf, valid, known, times, frame_mask)
    loss = out_train["features"]["left_hand"].sum()
    loss.backward()
    assert train_expert.cross_graph.gate.grad is not None and train_expert.cross_graph.gate.grad.abs().item() > 0.0
    first_block = train_expert.cross_graph.blocks[0]
    has_nonzero_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0.0 for p in first_block.parameters())
    assert has_nonzero_grad, "Cross-articulator block parameters must receive nonzero first-step gradients"

    missing = expert(xy, conf, torch.zeros_like(valid), known, times, frame_mask)
    assert all(torch.isfinite(v).all() for v in missing["features"].values())
    assert not any(v.any() for v in missing["available"].values())


def test_fusion_v2_continuous_time_lag_respects_physical_window_and_entropy():
    cfg = v2_smoke_config().fusion
    dims = {n: 16 for n in STREAM_NAMES}
    fusion = ReliabilityAwareFusion(dims, cfg).eval()
    b, t = 1, 5
    streams = {n: torch.randn(b, t, 16) for n in STREAM_NAMES}
    avail = {n: torch.ones(b, t, dtype=torch.bool) for n in STREAM_NAMES}
    quality = {n: torch.ones(b, t, 5) for n in STREAM_NAMES}

    # Widely spaced timestamps (> lag_window_seconds=0.20s) filter neighbours out cleanly.
    sparse_times = torch.tensor([[0.0, 0.5, 1.0, 1.5, 2.0]])
    fused, weights, _, _, aux = fusion.forward_with_aux(streams, avail, quality, timestamps=sparse_times)
    assert torch.isfinite(fused).all()
    assert torch.allclose(aux["expected_lag_seconds"], torch.zeros_like(aux["expected_lag_seconds"]), atol=1e-5)

    dense_times = torch.tensor([[0.0, 0.04, 0.08, 0.12, 0.16]])
    fused_d, weights_d, _, _, aux_d = fusion.forward_with_aux(streams, avail, quality, timestamps=dense_times)
    assert fused_d.shape == (b, t, cfg.dim)
    ent_loss = compute_fusion_entropy_loss(weights_d, aux_d["stream_mask"])
    assert 0.0 <= float(ent_loss) <= 1.0


def test_adaptive_temporal_consolidation_is_padding_invariant():
    cfg = v2_smoke_config().temporal
    enc = HierarchicalTemporalEncoder(cfg).eval()
    torch.manual_seed(31)
    x = torch.randn(1, 5, cfg.dim)
    t = torch.arange(5).float()[None] / 25.0
    y_short, mask_s, aux_s = enc(x, torch.tensor([5]), t)

    xp = torch.cat([x, torch.randn(1, 4, cfg.dim)], dim=1)
    tp = torch.cat([t, torch.zeros(1, 4)], dim=1)
    y_padded, mask_p, aux_p = enc(xp, torch.tensor([5]), tp)

    assert y_short.shape == (1, 5, cfg.dim)
    assert mask_s.sum().item() == 5 and mask_p.sum().item() == 5
    assert torch.allclose(y_short, y_padded[:, :5], atol=1e-5, rtol=1e-5)
    assert (aux_p["event_salience"][:, 5:] == 0).all()
    # Valid assignment rows over valid frames must sum to 1.0.
    valid_slots = aux_s["global_mask"][0]
    assert torch.allclose(aux_s["event_assignment"][0, valid_slots].sum(-1), torch.ones(valid_slots.sum()))


def test_fingerspelling_span_routing_and_spotting_loss(tmp_path):
    expert = FingerspellingExpert(8, 6, hidden_dim=16, layers=1, dropout=0.0, span_routing=True)
    b, t = 2, 7
    vr = torch.randn(b, t, 8)
    pr = torch.randn(b, t, 6)
    avail = torch.tensor([[1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    lengths = torch.tensor([5, 7])
    out = expert(vr, vr, pr, pr, lengths, avail, avail)
    assert out["spot_logits"].shape == (b, t)
    assert (out["spot_prob"][0, 5:] == 0).all()

    span_targets = torch.tensor([[0, 1, 1, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    span_known = torch.tensor([True, False])
    loss_spot = compute_fingerspelling_spotting_loss(
        out["spot_logits"], out["available"], lengths, ["abc", ""], span_targets=span_targets, span_known=span_known
    )
    loss_ctc = compute_ctc_batch(
        out["logits"], lengths, ["abc", ""], out["available"], span_targets=span_targets, span_known=span_known
    )
    (loss_spot + loss_ctc).backward()
    assert torch.isfinite(loss_spot) and torch.isfinite(loss_ctc)

    # Verify JSONL span parsing in dataset/collate.
    ts = np.arange(5, dtype=np.float64) / 25.0
    np.savez_compressed(
        tmp_path / "fs.npz",
        frames=np.zeros((5, 16, 16, 3), np.uint8),
        video_timestamps=ts,
        pose_xy=np.full((5, 66, 2), 0.5, np.float32),
        pose_timestamps=ts,
    )
    manifest = tmp_path / "m.jsonl"
    manifest.write_text(json.dumps({
        "uid": "u1", "npz": "fs.npz", "translation": "hi", "source_id": "s1",
        "fingerspelling_text": "hi", "fingerspelling_spans": [[0.04, 0.12]],
    }) + "\n", encoding="utf-8")
    ds = ISLVideoDataset(manifest, crop_size=16, image_size=32)
    batch = collate_examples([ds[0]])
    assert batch["fingerspelling_span_known"].tolist() == [True]
    assert batch["fingerspelling_spans"][0].tolist() == [False, True, True, True, False]


def test_planner_v2_dynamic_mask_diversity_and_decoder_counterfactual_plan_ot():
    planner = LatentPlanGrounder(
        dim=16, num_queries=6, layers=1, heads=4, dropout=0.0,
        diverse_ordered=True, dynamic_routing=True, min_queries=2, target_stride=4,
    ).eval()
    mem = torch.randn(2, 9, 16)
    mask = torch.tensor([[1] * 9, [1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
    times = torch.arange(9).float()[None].repeat(2, 1) / 25.0
    plan, attn, aux = planner.forward_with_aux(mem, mask, timestamps=times)
    assert aux["plan_mask"][0].sum().item() == 3
    assert aux["plan_mask"][1].sum().item() == 2
    assert (plan[1, 2:].abs().sum() == 0) and (attn[1, 2:].abs().sum() == 0)
    assert torch.isfinite(aux["diversity_loss"]) and torch.isfinite(aux["order_loss"])

    # Identical attention rows trigger higher diversity penalty than orthogonal attention rows.
    eye_attn = torch.eye(4, 8)[None]
    dup_attn = torch.zeros(1, 4, 8)
    dup_attn[:, :, 0] = 1.0
    pm = torch.ones(1, 4, dtype=torch.bool)
    mm = torch.ones(1, 8, dtype=torch.bool)
    eye_plan = torch.randn(1, 4, 16)
    div_orth, _ = planner._grounding_losses(eye_plan, eye_attn, pm, mm, None)
    div_dup, _ = planner._grounding_losses(eye_plan, dup_attn, pm, mm, None)
    assert div_dup > div_orth

    wrapper = T5VisualDecoder(MockT5(dim=16, vocab=40), visual_dim=8)
    for batch_sz in (2, 1):
        visual = torch.randn(batch_sz, 5, 8, requires_grad=True)
        visual_mask = torch.ones(batch_sz, 5, dtype=torch.bool)
        plan_in = torch.randn(batch_sz, 3, 8, requires_grad=True)
        plan_mask = torch.tensor([[1, 1, 0]] * batch_sz, dtype=torch.bool)
        labels = torch.tensor([[1, 2, 3, -100]] * batch_sz)
        out = wrapper(
            visual, visual_mask, plan_in, labels,
            semantic_ot=True, plan_mask=plan_mask, plan_semantic_ot=True,
            counterfactual_grounding=True, counterfactual_margin=0.5, counterfactual_kl_margin=0.2,
        )
        assert torch.isfinite(out["plan_semantic_ot_loss"])
        assert torch.isfinite(out["counterfactual_loss"])
        total = out["loss"] + out["semantic_ot_loss"] + out["counterfactual_loss"]
        total.backward()
        assert visual.grad is not None and torch.isfinite(visual.grad).all()
        assert plan_in.grad is not None and torch.isfinite(plan_in.grad).all()


def test_full_v2_model_end_to_end_backward_and_padding_invariance():
    cfg = v2_smoke_config()
    model = build_model(cfg, with_decoder=False).eval()
    base = make_batch_inputs(batch=1, time=6)
    with torch.no_grad():
        a = model(**base)["memory"]
    padded = make_batch_inputs(batch=1, time=10)
    for k in (
        "frames", "timestamps", "pose_xy", "pose_confidence", "pose_valid",
        "confidence_known", "boxes", "box_valid", "box_scores",
    ):
        padded[k][:, :6] = base[k]
    padded["frames"][:, 6:] = 0
    padded["timestamps"][:, 6:] = 0
    padded["pose_valid"][:, 6:] = False
    padded["box_valid"][:, 6:] = False
    padded["lengths"] = torch.tensor([6])
    with torch.no_grad():
        b = model(**padded)["memory"]
    assert torch.allclose(a, b[:, :6], atol=2e-5, rtol=2e-5)

    model.train()
    batch = make_batch_inputs(batch=2, time=8)
    out = model(**batch)
    ent_loss = compute_fusion_entropy_loss(out["modality_weights"], out["stream_available"], out["memory_mask"])
    total = (
        out["memory"].square().mean()
        + out["plan_diversity_loss"]
        + out["plan_order_loss"]
        + ent_loss
    )
    total.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_single_frame_adaptive_temporal_and_full_model_backward_finite():
    cfg = v2_smoke_config()
    model = ISLMEGA(cfg, decoder=T5VisualDecoder(MockT5(dim=cfg.temporal.dim, vocab=40), cfg.temporal.dim))
    model.train()
    batch = make_batch_inputs(batch=1, time=1)
    batch["lengths"] = torch.tensor([1])
    labels = torch.tensor([[1, 2, 3, -100]])
    out = model(labels=labels, semantic_ot=True, counterfactual_grounding=True, **batch)
    loss = (
        out["decoder_loss"]
        + out["decoder_semantic_ot_loss"]
        + out["decoder_counterfactual_loss"]
        + out["plan_diversity_loss"]
        + out["plan_order_loss"]
    )
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_wrist_missing_hand_falls_back_to_centroid_preserving_finger_geometry():
    cfg = v2_smoke_config().pose
    cfg.cross_articulator_graph = False
    expert = PoseExpert(cfg).eval()
    captured = {}

    class CaptureRight(torch.nn.Module):
        def forward(self, features, xy, valid, scale=None, extra_emb=None):
            captured["features"] = features.detach().clone()
            return torch.zeros((features.shape[0], features.shape[1], cfg.hidden_dim), dtype=features.dtype), valid.any(-1)

        def forward_nodes(self, *args, **kwargs):
            return self.forward(*args, **kwargs)

    expert.right = CaptureRight()
    b, t = 1, 2
    xy = torch.full((b, t, 66, 2), 0.5)
    xy[:, :, 21:42, 0] = 0.6
    xy[:, :, 22:42, 1] = torch.linspace(0.3, 0.7, 20)
    valid = torch.ones((b, t, 66), dtype=torch.bool)
    valid[:, :, 21] = False  # Right wrist missing, all 20 right-hand finger landmarks valid.
    conf = valid.float()
    known = torch.ones_like(valid)
    times = torch.tensor([[0.0, 0.04]])
    frame_mask = torch.ones((b, t), dtype=torch.bool)
    out = expert(xy, conf, valid, known, times, frame_mask)
    assert out["available"]["right_hand"].all()
    # Relative y coordinates of fingers (channel 3) must remain preserved via hand centroid fallback.
    assert captured["features"][0, 0, 1:, 3].abs().max().item() > 0.05


def test_config_validation_and_span_error_handling(tmp_path):
    bad_kernels = tmp_path / "bad_kernels.yaml"
    bad_kernels.write_text("temporal:\n  dim: 64\n  local_kernels: [3, 5, 7]\n  local_dilations: [1, 1, 1]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="divisible"):
        load_config(bad_kernels)

    bad_nan = tmp_path / "bad_nan.yaml"
    bad_nan.write_text("train:\n  visual_lr: .nan\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid optimizer settings"):
        load_config(bad_nan)

    bad_manifest = tmp_path / "bad_spans.jsonl"
    bad_manifest.write_text(
        json.dumps({"uid": "u", "npz": "x.npz", "translation": "ok", "source_id": "s", "fingerspelling_spans": [[0.5, 0.1]]}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="start <= end"):
        ISLVideoDataset(bad_manifest, crop_size=16, image_size=32)


def test_integrated_v2_decoder_duplicate_label_fallback_generate_and_train_epoch(tmp_path):
    import importlib.util
    cfg = v2_smoke_config()
    mock_t5 = MockT5(dim=cfg.temporal.dim, vocab=40)
    model = ISLMEGA(cfg, decoder=T5VisualDecoder(mock_t5, cfg.temporal.dim))
    batch = make_batch_inputs(batch=2, time=8)
    # Identical labels across batch items force per-sample zero-counterfactual fallback instead of self-contrast.
    same_labels = torch.tensor([[1, 2, 3, -100], [1, 2, 3, -100]])
    out = model(labels=same_labels, semantic_ot=True, counterfactual_grounding=True, **batch)
    assert torch.isfinite(out["decoder_counterfactual_loss"])
    assert torch.isfinite(out["decoder_counterfactual_nll_gap"])
    (out["decoder_loss"] + out["decoder_counterfactual_loss"]).backward()
    model.eval()
    gen = model.generate(num_beams=1, max_new_tokens=4, **batch)
    assert gen.shape == (2, 4)

    # Exercise scripts/train.py:run_epoch end-to-end with MockTokenizer + span annotations.
    script = Path(__file__).parents[1] / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("isl_mega_train_v2_test", script)
    train_mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(train_mod)

    class MockTokenizer:
        def __call__(self, texts, padding=True, truncation=False, return_tensors="pt"):
            rows = [[((i + j) % 10) + 1 for j in range(4)] for i, _ in enumerate(texts)]
            ids = torch.tensor(rows, dtype=torch.long)
            return SimpleNamespace(input_ids=ids, attention_mask=torch.ones_like(ids))

    model2 = ISLMEGA(cfg, decoder=T5VisualDecoder(MockT5(dim=cfg.temporal.dim, vocab=40), cfg.temporal.dim))
    opt = torch.optim.AdamW(model2.parameters(), lr=1e-4)
    collate_batch = {
        "uids": ["u1", "u2"],
        "source_ids": ["src1", "src2"],
        "signer_ids": ["sig1", "sig2"],
        **batch,
        "frame_mask": torch.arange(8)[None] < batch["lengths"][:, None],
        "translations": ["hello world", "good morning"],
        "fingerspelling_texts": ["ab", ""],
        "fingerspelling_spans": torch.tensor([
            [0, 1, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0],
        ], dtype=torch.bool),
        "fingerspelling_span_known": torch.tensor([True, False]),
    }
    stats = train_mod.run_epoch(model2, [collate_batch], MockTokenizer(), cfg, torch.device("cpu"), optimizer=opt)
    assert stats["batches"] == 1
    assert np.isfinite(stats["loss"]) and np.isfinite(stats["cf_nll_gap"])


def test_compact_v3_config_and_motion_disabled_forward_backward():
    v3 = load_config(Path(__file__).parents[1] / "configs" / "research_v3_compact.yaml")
    assert not v3.motion.enabled
    assert not v3.vision.separate_hand_face_body
    assert v3.vision.freeze_backbone
    assert v3.decoder.model_name == "google/flan-t5-base"
    assert v3.train.ot_weight == 0.0 and v3.train.fusion_entropy_weight == 0.0

    # Verify that motion.enabled=False cleanly runs through ISLMEGA without instantiating MotionExpert
    cfg = v2_smoke_config()
    cfg.motion.enabled = False
    model = ISLMEGA(cfg)
    assert model.motion is None
    batch = make_batch_inputs(batch=2, time=4)
    enc = model.encode(**batch)
    assert enc["memory"].shape == (2, 4, cfg.temporal.dim)
    loss = enc["memory"].sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.temporal.parameters() if p.requires_grad)


def test_joint_source_signer_connected_component_splitting_and_diagnostics(tmp_path):
    from isl_mega.data import write_grouped_splits, validate_split_disjointness
    records = [
        {"uid": "c1", "source_id": "v1", "signer_id": "s1"},
        {"uid": "c2", "source_id": "v1", "signer_id": "s1"},
        {"uid": "c3", "source_id": "v2", "signer_id": "s1"},  # Connects v1 and v2 via s1
        {"uid": "c4", "source_id": "v3", "signer_id": "s2"},
        {"uid": "c5", "source_id": "v3", "signer_id": "s2"},
        {"uid": "c6", "source_id": "v4", "signer_id": "s3"},
        {"uid": "c7", "source_id": "v5", "signer_id": "s4"},
    ]
    splits = write_grouped_splits(
        records, tmp_path / "splits", ratios=(0.5, 0.25, 0.25), seed=42, group_key="joint_source_signer"
    )
    validate_split_disjointness(splits, "source_id")
    validate_split_disjointness(splits, "signer_id")
    diag_file = tmp_path / "splits" / "split_diagnostics.json"
    assert diag_file.exists()
    diag = json.loads(diag_file.read_text(encoding="utf-8"))
    assert diag["group_key"] == "joint_source_signer"
    assert diag["total_records"] == 7
    assert diag["total_components"] >= 3




