"""Regression tests for the RA-MAGC package.

These tests use synthetic arrays only. They verify implementation contracts, not
ISL translation quality or generalization on a real dataset.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from ramagc.config import (
    ConformerConfig,
    ModelConfig,
    MultiScaleTemporalConfig,
    RegionFusionConfig,
)
from ramagc.dataset import (
    RAMAGCCollator,
    StructuredPoseSample,
    compute_torso_reference,
    compute_velocity,
    extract_structured_sample,
)
from ramagc.fusion import ReliabilityGate
from ramagc.graph_layers import EdgeAwareGraphAttention
from ramagc.model import RAMAGCTranslator, RAMAGCVisualEncoder
from ramagc.temporal import (
    ConformerBlock,
    FineDetailRetrieval,
    MultiScaleTemporalBlock,
    PairwiseAggregator,
    length_mask,
)


def create_synthetic_raw_data(T: int = 20, include_face: bool = True, time_offset: float = 0.0):
    times = time_offset + np.arange(T, dtype=np.float64) * 0.04
    segments = np.zeros(T, dtype=np.int32)
    body = np.random.default_rng(1).normal(size=(T, 6, 2)).astype(np.float32)
    body[:, 0, :] = [-0.2, 0.0]
    body[:, 1, :] = [0.2, 0.0]
    body_valid = np.ones((T, 6), dtype=bool)
    left = np.random.default_rng(2).normal(size=(T, 21, 2)).astype(np.float32) * 0.1 + [-0.3, -0.3]
    right = np.random.default_rng(3).normal(size=(T, 21, 2)).astype(np.float32) * 0.1 + [0.3, -0.3]
    raw_coords = {"body": body, "left_hand": left, "right_hand": right}
    raw_valid = {
        "body": body_valid,
        "left_hand": np.ones((T, 21), dtype=bool),
        "right_hand": np.ones((T, 21), dtype=bool),
    }
    if include_face:
        face = np.random.default_rng(4).normal(size=(T, 18, 2)).astype(np.float32) * 0.05
        face[:, 0, :] = [-0.05, 0.2]
        face[:, 4, :] = [0.05, 0.2]
        raw_coords["face"] = face
        raw_valid["face"] = np.ones((T, 18), dtype=bool)
    return times, segments, raw_coords, raw_valid


def make_sample(T: int = 12, include_face: bool = False, time_offset: float = 0.0):
    times, segments, coords, valid = create_synthetic_raw_data(T, include_face, time_offset)
    return extract_structured_sample(
        uid=f"sample-{T}-{include_face}",
        text="a synthetic sentence",
        times=times,
        segments=segments,
        raw_coords=coords,
        raw_valid=valid,
        include_face=include_face,
    )


def _encoder_inputs(batch: dict):
    return {
        key: batch[key]
        for key in (
            "left_hand_nodes", "left_hand_edges", "left_hand_adj", "left_hand_valid",
            "right_hand_nodes", "right_hand_edges", "right_hand_adj", "right_hand_valid",
            "body_nodes", "body_edges", "body_adj", "body_valid",
            "timestamps", "lengths", "face_nodes", "face_valid",
        )
    }


def test_feature_dimensions_and_clip_relative_times():
    sample = make_sample(T=15, include_face=True, time_offset=1_700_000_000.0)
    assert sample.left_hand_nodes.shape == (15, 21, 13)
    assert sample.right_hand_nodes.shape == (15, 21, 13)
    assert sample.body_nodes.shape == (15, 6, 10)
    assert sample.face_nodes is not None and sample.face_nodes.shape == (15, 18, 13)
    assert np.isclose(sample.times[0], 0.0)
    assert np.allclose(np.diff(sample.times), 0.04, atol=1e-6)
    for array in (sample.left_hand_nodes, sample.right_hand_nodes, sample.body_nodes, sample.face_nodes):
        assert np.isfinite(array).all()


def test_torso_center_interpolates_missing_shoulder_frames_in_timestamp_space():
    T = 3
    body = np.zeros((T, 6, 2), dtype=np.float32)
    body[:, 0, 0] = [-0.2, 0.0, 0.4]
    body[:, 1, 0] = [0.2, 0.4, 0.8]
    body[:, 0, 1] = body[:, 1, 1] = 0.0
    valid = np.ones((T, 6), dtype=bool)
    valid[1, 0:2] = False
    times = np.array([0.0, 1.0, 3.0])
    centers, scale = compute_torso_reference(body, valid, times=times)
    # Endpoints' midpoint x values are 0 and 0.6; interpolate at t=1 of 3.
    assert np.allclose(centers[:, 0], [0.0, 0.2, 0.6])
    assert np.allclose(centers[:, 1], 0.0)
    assert np.isclose(scale, 0.4)


def test_torso_reference_rejects_missing_all_shoulder_pairs():
    body = np.zeros((5, 6, 2), dtype=np.float32)
    valid = np.zeros((5, 6), dtype=bool)
    with pytest.raises(ValueError, match="no valid left/right shoulder pair"):
        compute_torso_reference(body, valid)


def test_velocity_uses_actual_elapsed_seconds():
    pos = np.array([[[0.0, 0.0]], [[0.2, 0.0]], [[0.5, 0.0]]], dtype=np.float32)
    valid = np.ones((3, 1), dtype=bool)
    vel, vel_valid = compute_velocity(pos, valid, np.array([0.0, 0.1, 0.25]))
    assert np.allclose(vel[1, 0], [2.0, 0.0])
    assert np.allclose(vel[2, 0], [2.0, 0.0])
    assert not vel_valid[0, 0] and vel_valid[1:, 0].all()


def test_confidence_availability_is_not_inferred_from_visibility():
    times, seg, coords, valid = create_synthetic_raw_data(8, include_face=False)
    no_conf = extract_structured_sample("no-conf", "x", times, seg, coords, valid)
    assert np.all(no_conf.left_hand_nodes[..., 6:8] == 0)
    assert np.all(no_conf.body_nodes[..., 4:6] == 0)
    conf = {
        "left_hand": np.full((8, 21), 0.85, dtype=np.float32),
        "body": np.full((8, 6), 0.95, dtype=np.float32),
    }
    yes_conf = extract_structured_sample("yes-conf", "x", times, seg, coords, valid, raw_conf=conf)
    assert np.allclose(yes_conf.left_hand_nodes[..., 6], 0.85)
    assert np.all(yes_conf.left_hand_nodes[..., 7] == 1.0)
    assert np.allclose(yes_conf.body_nodes[..., 4], 0.95)


def test_confidence_out_of_range_is_rejected():
    times, seg, coords, valid = create_synthetic_raw_data(5, include_face=False)
    conf = {"left_hand": np.full((5, 21), 1.5, dtype=np.float32)}
    with pytest.raises(ValueError, match=r"within \[0,1\]"):
        extract_structured_sample("bad-conf", "x", times, seg, coords, valid, raw_conf=conf)


def test_invalid_timestamp_rank_and_segment_are_rejected():
    times, segments, coords, valid = create_synthetic_raw_data(5, include_face=False)
    with pytest.raises(ValueError, match="one-dimensional"):
        extract_structured_sample("rank", "x", times[:, None], segments, coords, valid)
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_structured_sample("duplicate", "x", np.array([0, .04, .04, .12, .16]), segments, coords, valid)
    with pytest.raises(ValueError, match="multiple segment IDs"):
        extract_structured_sample("cut", "x", times, np.array([0, 0, 1, 1, 1]), coords, valid)


def test_face_feature_extraction_100_frames():
    sample = make_sample(T=100, include_face=True)
    assert sample.face_nodes is not None
    assert sample.face_nodes.shape == (100, 18, 13)
    assert np.isfinite(sample.face_nodes).all()


def test_pairwise_aggregation_odd_even_and_absolute_timestamp_offset_invariance():
    torch.manual_seed(0)
    aggregator = PairwiseAggregator(256, 256).eval()
    features = torch.randn(2, 10, 256)
    lengths = torch.tensor([9, 10])
    timestamps = torch.arange(10, dtype=torch.float64).repeat(2, 1) * 0.04
    with torch.no_grad():
        coarse_a, coarse_lens, elapsed_a = aggregator(features, lengths, timestamps)
        coarse_b, _, elapsed_b = aggregator(features, lengths, timestamps + 1_700_000_000.0)
    assert coarse_a.shape == (2, 5, 256)
    assert torch.equal(coarse_lens, torch.tensor([5, 5]))
    assert torch.allclose(coarse_a, coarse_b, atol=1e-6)
    assert torch.allclose(elapsed_a, elapsed_b, atol=1e-6)
    mask = length_mask(coarse_lens, 5, features.device)
    assert torch.all(coarse_a[~mask] == 0.0)


def test_pairwise_aggregator_rejects_nonmonotonic_valid_times():
    agg = PairwiseAggregator()
    features = torch.randn(1, 4, 256)
    with pytest.raises(ValueError, match="strictly increasing"):
        agg(features, torch.tensor([4]), torch.tensor([[0.0, 0.1, 0.08, 0.2]]))


def test_multiscale_configured_kernel_and_dilation_values_are_used():
    block = MultiScaleTemporalBlock(
        in_channels=256,
        out_channels=256,
        branch_dim=64,
        kernel_sizes=(3, 7, 5, 9),
        dilations=(1, 2, 3, 1),
    )
    assert [branch["depthwise"].kernel_size[0] for branch in block.branches] == [3, 7, 5, 9]
    assert [branch["depthwise"].dilation[0] for branch in block.branches] == [1, 2, 3, 1]
    x = torch.randn(2, 11, 256)
    mask = torch.tensor([[1] * 11, [1] * 7 + [0] * 4], dtype=torch.bool)
    assert block(x, mask).shape == (2, 11, 256)
    assert torch.count_nonzero(block(x, mask)[1, 7:]) == 0


def test_graph_attention_uses_query_i_key_j_and_edge_ij():
    layer = EdgeAwareGraphAttention(node_dim=4, edge_dim=4, num_heads=1, dropout=0.0)
    q = torch.tensor([[[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]]])
    k = torch.tensor([[[[0.0, 2.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]])
    edge = torch.zeros((1, 1, 2, 2, 4))
    edge[0, 0, 0, 1, 0] = 2.0
    edge[0, 0, 1, 0, 1] = 4.0
    actual = layer.attention_scores(q, k, edge)
    expected = torch.zeros((1, 1, 2, 2))
    for i in range(2):
        for j in range(2):
            expected[0, 0, i, j] = torch.dot(q[0, 0, i], k[0, 0, j] + edge[0, 0, i, j]) / 2.0
    assert torch.allclose(actual, expected)
    assert not torch.allclose(actual, actual.transpose(-1, -2))


def test_graph_attention_forward_is_finite_and_respects_mask_shape():
    layer = EdgeAwareGraphAttention(16, 4, num_heads=4, dropout=0.0)
    nodes = torch.randn(3, 5, 16)
    edges = torch.randn(3, 5, 5, 4)
    adj = torch.eye(5, dtype=torch.bool).unsqueeze(0).expand(3, -1, -1).clone()
    out = layer(nodes, edges, adj)
    assert out.shape == nodes.shape
    assert torch.isfinite(out).all()


def test_conformer_padding_invariance():
    config = ConformerConfig()
    block = ConformerBlock(config).eval()
    with torch.no_grad():
        block.conv_in_norm.bias.fill_(0.5)
        block.conv_mid_norm.bias.fill_(-0.3)
        block.attn_norm.bias.fill_(0.2)
    short = torch.randn(1, 4, config.d_model)
    mask_short = torch.ones(1, 4, dtype=torch.bool)
    bias_short = torch.zeros(1, config.num_heads, 4, 4)
    padded = torch.cat([short, torch.zeros(1, 2, config.d_model)], dim=1)
    mask_padded = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)
    bias_padded = torch.zeros(1, config.num_heads, 6, 6)
    with torch.no_grad():
        out_short = block(short, mask_short, bias_short)
        out_padded = block(padded, mask_padded, bias_padded)
    assert torch.allclose(out_short, out_padded[:, :4], atol=1e-6, rtol=0)
    assert torch.count_nonzero(out_padded[:, 4:]) == 0


def test_fine_detail_retrieval_local_attention_and_padding_invariance():
    module = FineDetailRetrieval(d_model=32, num_heads=4, local_window_radius=2).eval()
    fine = torch.randn(2, 11, 32)
    coarse = torch.randn(2, 6, 32)
    coarse_mask = torch.tensor([[1] * 6, [1] * 4 + [0] * 2], dtype=torch.bool)
    fine_mask = torch.tensor([[1] * 11, [1] * 8 + [0] * 3], dtype=torch.bool)
    with torch.no_grad():
        out = module(coarse, fine, coarse_mask, fine_mask)
        out_short = module(coarse[:1, :6], fine[:1, :11], coarse_mask[:1, :6], fine_mask[:1, :11])
    assert out.shape == coarse.shape
    assert torch.isfinite(out).all()
    assert torch.allclose(out[0], out_short[0], atol=1e-6, rtol=0)
    assert torch.count_nonzero(out[1, 4:]) == 0


def test_reliability_gate_uses_explicit_missing_observation_embedding():
    cfg = RegionFusionConfig()
    gate = ReliabilityGate(cfg)
    tokens = torch.randn(1, 2, 3, cfg.region_dim)
    interacted = torch.randn_like(tokens)
    quality = torch.zeros(1, 2, 3, 6)
    valid = torch.tensor([[[False, False, False], [True, False, False]]])
    out = gate(tokens, interacted, quality, valid)
    assert torch.allclose(out[0, 0], gate.missing_observation_embed[0, 0])
    assert torch.isfinite(out).all()


def test_visual_encoder_padding_invariance_and_output_shapes():
    torch.manual_seed(5)
    encoder = RAMAGCVisualEncoder(ModelConfig()).eval()
    short = make_sample(8, include_face=False)
    long = make_sample(14, include_face=False)
    collator = RAMAGCCollator()
    batch_single = collator([short])
    batch_mixed = collator([short, long])
    with torch.no_grad():
        emb_single, mask_single = encoder(**_encoder_inputs(batch_single))
        emb_mixed, mask_mixed = encoder(**_encoder_inputs(batch_mixed))
    short_coarse = (8 + 1) // 2
    assert emb_single.shape == (1, short_coarse, 512)
    assert torch.allclose(emb_single[0], emb_mixed[0, :short_coarse], atol=1e-4, rtol=0)
    assert mask_single.all() and mask_mixed[0, :short_coarse].all()


def test_face_enabled_model_keeps_face_stream_even_when_all_face_data_missing():
    config = ModelConfig()
    config.face.enabled = True
    encoder = RAMAGCVisualEncoder(config).eval()
    batch = RAMAGCCollator()([make_sample(8, include_face=False)])
    with torch.no_grad():
        embeddings, mask = encoder(**_encoder_inputs(batch))
    assert embeddings.shape == (1, 4, 512)
    assert mask.all()
    assert torch.isfinite(embeddings).all()


def test_batch_collator_returns_decoder_attention_mask():
    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.tensor([[10, 11, 0], [12, 0, 0]]),
                "attention_mask": torch.tensor([[1, 1, 1], [1, 0, 0]]),
            }

    batch = RAMAGCCollator(tokenizer=DummyTokenizer())([make_sample(6), make_sample(7)])
    assert torch.equal(batch["decoder_attention_mask"], torch.tensor([[1, 1, 1], [1, 0, 0]]))
    assert torch.equal(batch["labels"], torch.tensor([[10, 11, 0], [12, -100, -100]]))


def test_translator_passes_decoder_attention_mask_to_t5():
    class DummyVisual(nn.Module):
        output_dim = 8

        def forward(self, **kwargs):
            x = torch.randn(2, 3, 8)
            return x, torch.ones(2, 3, dtype=torch.bool)

    class DummyT5(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(d_model=8)
            self.last_decoder_mask = None

        def forward(self, **kwargs):
            self.last_decoder_mask = kwargs["decoder_attention_mask"]
            return SimpleNamespace(loss=torch.tensor(0.2), logits=torch.zeros(2, 3, 20))

    t5 = DummyT5()
    translator = RAMAGCTranslator(DummyVisual(), t5)
    target_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    translator(labels=torch.tensor([[4, 5, 6], [7, -100, -100]]), decoder_attention_mask=target_mask)
    assert torch.equal(t5.last_decoder_mask, target_mask)


def test_finite_backward_gradients():
    torch.manual_seed(7)
    encoder = RAMAGCVisualEncoder(ModelConfig()).train()
    batch = RAMAGCCollator()([make_sample(6, include_face=False)])
    embeddings, mask = encoder(**_encoder_inputs(batch))
    loss = embeddings[mask].square().mean()
    loss.backward()
    checked = 0
    for parameter in encoder.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
            checked += 1
    assert checked > 0
    assert loss.item() > 0


def test_configuration_rejects_incompatible_dimensions():
    with pytest.raises(ValueError, match="odd"):
        MultiScaleTemporalConfig(kernel_sizes=(3, 4, 5, 9))
    with pytest.raises(ValueError, match="equal conformer.d_model"):
        ModelConfig(multi_scale=MultiScaleTemporalConfig(out_channels=256), conformer=ConformerConfig(d_model=128))
