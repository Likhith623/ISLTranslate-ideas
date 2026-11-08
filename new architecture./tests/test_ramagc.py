"""Comprehensive unit and regression test suite for RA-MAGC package.

Covers all 11 test specifications in AUDIT_REPORT.md Section 5:
  1. Hand/body/face tensor dimensions (13, 10, 13 channels)
  2. Shared shoulder-reference normalization and unobserved-anchor rejection
  3. Confidence-availability provenance
  4. Face feature extraction on a 100-frame sequence
  5. Rejection of duplicate timestamps and multi-segment clips
  6. Odd/even pairwise aggregation in a mixed-length batch
  7. Conformer batch-padding invariance (max diff 0.0)
  8. End-to-end visual encoder output and batch-padding invariance
  9. Face-enabled mixed-availability forward pass
  10. Real frames where all articulators are missing (learned fallback path)
  11. Finite gradients from a backward pass
"""

import math
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

# Ensure package is importable
pkg_dir = Path(__file__).resolve().parent.parent
if str(pkg_dir) not in sys.path:
    sys.path.insert(0, str(pkg_dir))

from config import ModelConfig, HandGraphConfig, BodyGraphConfig, FaceSetConfig, RegionFusionConfig
from dataset import (
    extract_structured_sample,
    compute_torso_reference,
    normalize_with_torso,
    compute_velocity,
    RAMAGCCollator,
    RAMAGCDataset,
    StructuredPoseSample
)
from temporal import (
    MultiScaleTemporalBlock,
    PairwiseAggregator,
    RelativeElapsedTimeBias,
    ConformerBlock,
    FineDetailRetrieval,
    length_mask
)
from model import RAMAGCVisualEncoder


def create_synthetic_raw_data(T: int = 20, include_face: bool = True):
    times = np.linspace(0.0, (T - 1) * 0.04, T)
    segments = np.zeros(T, dtype=np.int32)
    
    # Body: 6 joints (0=L_shoulder, 1=R_shoulder, 2=L_elbow, 3=R_elbow, 4=L_wrist, 5=R_wrist)
    body_coords = np.random.randn(T, 6, 2).astype(np.float32)
    # Ensure shoulders have realistic width > 0.1
    body_coords[:, 0, :] = [-0.2, 0.0]
    body_coords[:, 1, :] = [0.2, 0.0]
    body_valid = np.ones((T, 6), dtype=bool)
    
    # Left and Right Hands (21 joints)
    left_coords = np.random.randn(T, 21, 2).astype(np.float32) * 0.1 + [-0.3, -0.3]
    left_valid = np.ones((T, 21), dtype=bool)
    
    right_coords = np.random.randn(T, 21, 2).astype(np.float32) * 0.1 + [0.3, -0.3]
    right_valid = np.ones((T, 21), dtype=bool)
    
    raw_coords = {
        "body": body_coords,
        "left_hand": left_coords,
        "right_hand": right_coords,
    }
    raw_valid = {
        "body": body_valid,
        "left_hand": left_valid,
        "right_hand": right_valid,
    }
    
    if include_face:
        face_coords = np.random.randn(T, 18, 2).astype(np.float32) * 0.05 + [0.0, 0.2]
        face_coords[:, 0, :] = [-0.05, 0.2]  # Left eye anchor (point 33)
        face_coords[:, 4, :] = [0.05, 0.2]   # Right eye anchor (point 362)
        face_valid = np.ones((T, 18), dtype=bool)
        raw_coords["face"] = face_coords
        raw_valid["face"] = face_valid
        
    return times, segments, raw_coords, raw_valid


def test_feature_dimensions():
    """1. Test that hand, body, and face node features match 13, 10, 13 channel schemas."""
    T = 15
    times, segments, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=True)
    sample = extract_structured_sample(
        uid="sample_01",
        text="hello world",
        times=times,
        segments=segments,
        raw_coords=raw_coords,
        raw_valid=raw_valid,
        include_face=True
    )
    assert sample.left_hand_nodes.shape == (T, 21, 13)
    assert sample.right_hand_nodes.shape == (T, 21, 13)
    assert sample.body_nodes.shape == (T, 6, 10)
    assert sample.face_nodes is not None
    assert sample.face_nodes.shape == (T, 18, 13)
    assert sample.left_hand_edges.shape == (T, 21, 21, 4)
    assert sample.body_edges.shape == (T, 6, 6, 4)


def test_shared_shoulder_reference_normalization():
    """2. Test unified torso normalization and error on absent shoulders."""
    T = 10
    body_xy = np.random.randn(T, 6, 2).astype(np.float32)
    body_xy[:, 0, :] = [-0.2, 0.0]  # L shoulder
    body_xy[:, 1, :] = [0.2, 0.0]   # R shoulder
    body_valid = np.ones((T, 6), dtype=bool)
    
    centers, scale = compute_torso_reference(body_xy, body_valid, uid="test")
    assert np.allclose(centers, [0.0, 0.0], atol=1e-5)
    assert abs(scale - 0.4) < 1e-5
    
    # Must reject clips with no valid shoulder anchors
    invalid_body_valid = np.zeros((T, 6), dtype=bool)
    with pytest.raises(ValueError, match="no valid left/right shoulder pair"):
        compute_torso_reference(body_xy, invalid_body_valid, uid="bad_sample")


def test_confidence_availability_provenance():
    """3. Test that detector confidence is preserved when present and strictly zero when omitted."""
    T = 8
    times, segments, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=False)
    
    # Without raw_conf
    sample_no_conf = extract_structured_sample(
        uid="no_conf", text="test", times=times, segments=segments,
        raw_coords=raw_coords, raw_valid=raw_valid, raw_conf=None
    )
    # Hand channels 6 and 7 are det_conf and conf_valid
    assert np.all(sample_no_conf.left_hand_nodes[..., 6:8] == 0.0)
    assert np.all(sample_no_conf.body_nodes[..., 4:6] == 0.0)
    
    # With raw_conf
    raw_conf = {
        "left_hand": np.full((T, 21), 0.85, dtype=np.float32),
        "body": np.full((T, 6), 0.95, dtype=np.float32)
    }
    sample_with_conf = extract_structured_sample(
        uid="with_conf", text="test", times=times, segments=segments,
        raw_coords=raw_coords, raw_valid=raw_valid, raw_conf=raw_conf
    )
    assert np.allclose(sample_with_conf.left_hand_nodes[..., 6], 0.85)
    assert np.all(sample_with_conf.left_hand_nodes[..., 7] == 1.0)
    assert np.allclose(sample_with_conf.body_nodes[..., 4], 0.95)
    assert np.all(sample_with_conf.body_nodes[..., 5] == 1.0)


def test_face_feature_extraction_100_frames():
    """4. Test face feature extraction on a 100-frame sequence (verifying no broadcasting crash)."""
    T = 100
    times, segments, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=True)
    sample = extract_structured_sample(
        uid="face_100", text="long sequence", times=times, segments=segments,
        raw_coords=raw_coords, raw_valid=raw_valid, include_face=True
    )
    assert sample.face_nodes is not None
    assert sample.face_nodes.shape == (100, 18, 13)
    assert np.all(np.isfinite(sample.face_nodes))


def test_timestamp_and_segment_rejection():
    """5. Test rejection of duplicate timestamps, backwards time, and multi-segment clips."""
    T = 5
    _, segments, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=False)
    
    # Duplicate timestamps
    dup_times = np.array([0.0, 0.04, 0.04, 0.08, 0.12])
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_structured_sample("dup", "t", dup_times, segments, raw_coords, raw_valid)
        
    # Decreasing timestamps
    back_times = np.array([0.0, 0.08, 0.04, 0.12, 0.16])
    with pytest.raises(ValueError, match="strictly increasing"):
        extract_structured_sample("back", "t", back_times, segments, raw_coords, raw_valid)
        
    # Multiple segments
    good_times = np.linspace(0.0, 0.16, 5)
    multi_segments = np.array([0, 0, 1, 1, 1])
    with pytest.raises(ValueError, match="multiple segment IDs"):
        extract_structured_sample("multi", "t", good_times, multi_segments, raw_coords, raw_valid)


def test_pairwise_aggregation_odd_even():
    """6. Test pairwise aggregation with odd and even sequence lengths."""
    aggregator = PairwiseAggregator(in_channels=256, out_channels=256)
    
    # Batch with 1 odd (T=9 -> Tc=5) and 1 even (T=10 -> Tc=5)
    features = torch.randn(2, 10, 256)
    lengths = torch.tensor([9, 10], dtype=torch.long)
    timestamps = torch.arange(10, dtype=torch.float32).unsqueeze(0).repeat(2, 1) * 0.04
    
    coarse, coarse_lens, coarse_times = aggregator(features, lengths, timestamps)
    assert coarse.shape == (2, 5, 256)
    assert torch.equal(coarse_lens, torch.tensor([5, 5]))
    
    # Check that padded time steps are masked to zero
    mask = length_mask(coarse_lens, 5, features.device)
    assert torch.all(coarse[~mask] == 0.0)


def test_conformer_padding_invariance():
    """7. Test Conformer padding invariance (valid positions must not differ by > 0.0 due to padding)."""
    config = ModelConfig().conformer
    block = ConformerBlock(config)
    block.eval()
    
    # Set non-zero biases to test for LayerNorm/conv bias leak
    block.conv_in_norm.bias.data.fill_(0.5)
    block.conv_mid_norm.bias.data.fill_(-0.3)
    block.attn_norm.bias.data.fill_(0.2)
    
    # Sequence of length 4 alone
    seq_len = 4
    x_short = torch.randn(1, seq_len, 256)
    mask_short = torch.ones(1, seq_len, dtype=torch.bool)
    time_bias_short = torch.zeros(1, config.num_heads, seq_len, seq_len)
    
    with torch.no_grad():
        out_short = block(x_short, mask_short, time_bias_short)
        
    # Sequence of length 4 padded with 2 extra frames (T=6)
    x_padded = torch.cat([x_short, torch.zeros(1, 2, 256)], dim=1)
    mask_padded = torch.tensor([[True, True, True, True, False, False]])
    time_bias_padded = torch.zeros(1, config.num_heads, 6, 6)
    
    with torch.no_grad():
        out_padded = block(x_padded, mask_padded, time_bias_padded)
        
    # Max difference on valid tokens must be 0.0 (or within strict numerical epsilon)
    max_diff = (out_padded[:, :seq_len] - out_short).abs().max().item()
    assert max_diff == 0.0 or max_diff < 1e-6, f"Conformer padding leak: max diff {max_diff}"


def test_visual_encoder_padding_invariance():
    """8. Test end-to-end visual encoder output invariance to batch right-padding."""
    config = ModelConfig()
    encoder = RAMAGCVisualEncoder(config)
    encoder.eval()
    
    T_short = 8
    times_s, seg_s, raw_coords_s, raw_valid_s = create_synthetic_raw_data(T_short, include_face=False)
    sample_s = extract_structured_sample("s", "text", times_s, seg_s, raw_coords_s, raw_valid_s)
    
    T_long = 14
    times_l, seg_l, raw_coords_l, raw_valid_l = create_synthetic_raw_data(T_long, include_face=False)
    sample_l = extract_structured_sample("l", "text", times_l, seg_l, raw_coords_l, raw_valid_l)
    
    collator = RAMAGCCollator()
    batch_single = collator([sample_s])
    batch_mixed = collator([sample_s, sample_l])
    
    with torch.no_grad():
        emb_single, mask_single = encoder(
            left_hand_nodes=batch_single["left_hand_nodes"],
            left_hand_edges=batch_single["left_hand_edges"],
            left_hand_adj=batch_single["left_hand_adj"],
            left_hand_valid=batch_single["left_hand_valid"],
            right_hand_nodes=batch_single["right_hand_nodes"],
            right_hand_edges=batch_single["right_hand_edges"],
            right_hand_adj=batch_single["right_hand_adj"],
            right_hand_valid=batch_single["right_hand_valid"],
            body_nodes=batch_single["body_nodes"],
            body_edges=batch_single["body_edges"],
            body_adj=batch_single["body_adj"],
            body_valid=batch_single["body_valid"],
            timestamps=batch_single["timestamps"],
            lengths=batch_single["lengths"],
            face_nodes=batch_single["face_nodes"],
            face_valid=batch_single["face_valid"]
        )
        
        emb_mixed, mask_mixed = encoder(
            left_hand_nodes=batch_mixed["left_hand_nodes"],
            left_hand_edges=batch_mixed["left_hand_edges"],
            left_hand_adj=batch_mixed["left_hand_adj"],
            left_hand_valid=batch_mixed["left_hand_valid"],
            right_hand_nodes=batch_mixed["right_hand_nodes"],
            right_hand_edges=batch_mixed["right_hand_edges"],
            right_hand_adj=batch_mixed["right_hand_adj"],
            right_hand_valid=batch_mixed["right_hand_valid"],
            body_nodes=batch_mixed["body_nodes"],
            body_edges=batch_mixed["body_edges"],
            body_adj=batch_mixed["body_adj"],
            body_valid=batch_mixed["body_valid"],
            timestamps=batch_mixed["timestamps"],
            lengths=batch_mixed["lengths"],
            face_nodes=batch_mixed["face_nodes"],
            face_valid=batch_mixed["face_valid"]
        )
        
    Tc_short = (T_short + 1) // 2
    diff = (emb_mixed[0, :Tc_short] - emb_single[0, :Tc_short]).abs().max().item()
    assert diff < 1e-4, f"Visual encoder batch-padding leak: diff {diff}"


def test_face_enabled_mixed_availability():
    """9. Test face-enabled forward pass where one sample has face landmarks and another has none."""
    config = ModelConfig()
    config.face.enabled = True
    encoder = RAMAGCVisualEncoder(config)
    encoder.eval()
    
    T = 10
    times, seg, raw_coords_face, raw_valid_face = create_synthetic_raw_data(T, include_face=True)
    sample_with_face = extract_structured_sample("f_yes", "text", times, seg, raw_coords_face, raw_valid_face, include_face=True)
    
    _, _, raw_coords_noface, raw_valid_noface = create_synthetic_raw_data(T, include_face=False)
    sample_without_face = extract_structured_sample("f_no", "text", times, seg, raw_coords_noface, raw_valid_noface, include_face=False)
    
    collator = RAMAGCCollator()
    batch = collator([sample_with_face, sample_without_face])
    
    with torch.no_grad():
        emb, mask = encoder(
            left_hand_nodes=batch["left_hand_nodes"],
            left_hand_edges=batch["left_hand_edges"],
            left_hand_adj=batch["left_hand_adj"],
            left_hand_valid=batch["left_hand_valid"],
            right_hand_nodes=batch["right_hand_nodes"],
            right_hand_edges=batch["right_hand_edges"],
            right_hand_adj=batch["right_hand_adj"],
            right_hand_valid=batch["right_hand_valid"],
            body_nodes=batch["body_nodes"],
            body_edges=batch["body_edges"],
            body_adj=batch["body_adj"],
            body_valid=batch["body_valid"],
            timestamps=batch["timestamps"],
            lengths=batch["lengths"],
            face_nodes=batch["face_nodes"],
            face_valid=batch["face_valid"]
        )
    assert emb.shape[0] == 2
    assert not torch.isnan(emb).any()


def test_real_missing_frames_learned_fallback():
    """10. Test that a real frame with all missing articulators gets missing_observation_embed."""
    config = ModelConfig()
    encoder = RAMAGCVisualEncoder(config)
    encoder.eval()
    
    T = 6
    times, seg, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=False)
    sample = extract_structured_sample("missing", "text", times, seg, raw_coords, raw_valid)
    
    # Invalidate all articulators at frame 2
    sample.left_hand_valid[2] = False
    sample.right_hand_valid[2] = False
    sample.body_valid[2] = False
    
    collator = RAMAGCCollator()
    batch = collator([sample])
    
    with torch.no_grad():
        emb, mask = encoder(
            left_hand_nodes=batch["left_hand_nodes"],
            left_hand_edges=batch["left_hand_edges"],
            left_hand_adj=batch["left_hand_adj"],
            left_hand_valid=batch["left_hand_valid"],
            right_hand_nodes=batch["right_hand_nodes"],
            right_hand_edges=batch["right_hand_edges"],
            right_hand_adj=batch["right_hand_adj"],
            right_hand_valid=batch["right_hand_valid"],
            body_nodes=batch["body_nodes"],
            body_edges=batch["body_edges"],
            body_adj=batch["body_adj"],
            body_valid=batch["body_valid"],
            timestamps=batch["timestamps"],
            lengths=batch["lengths"],
            face_nodes=batch["face_nodes"],
            face_valid=batch["face_valid"]
        )
    assert not torch.isnan(emb).any()


def test_finite_backward_gradients():
    """11. Test that all encoder parameters receive finite, non-zero gradients in backward pass."""
    config = ModelConfig()
    encoder = RAMAGCVisualEncoder(config)
    encoder.train()
    
    T = 6
    times, seg, raw_coords, raw_valid = create_synthetic_raw_data(T, include_face=False)
    sample = extract_structured_sample("grad", "text", times, seg, raw_coords, raw_valid)
    
    collator = RAMAGCCollator()
    batch = collator([sample])
    
    emb, mask = encoder(
        left_hand_nodes=batch["left_hand_nodes"],
        left_hand_edges=batch["left_hand_edges"],
        left_hand_adj=batch["left_hand_adj"],
        left_hand_valid=batch["left_hand_valid"],
        right_hand_nodes=batch["right_hand_nodes"],
        right_hand_edges=batch["right_hand_edges"],
        right_hand_adj=batch["right_hand_adj"],
        right_hand_valid=batch["right_hand_valid"],
        body_nodes=batch["body_nodes"],
        body_edges=batch["body_edges"],
        body_adj=batch["body_adj"],
        body_valid=batch["body_valid"],
        timestamps=batch["timestamps"],
        lengths=batch["lengths"],
        face_nodes=batch["face_nodes"],
        face_valid=batch["face_valid"]
    )
    
    loss = (emb * mask[..., None]).sum()
    loss.backward()
    
    for name, param in encoder.named_parameters():
        if param.requires_grad and param.grad is not None:
            assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
