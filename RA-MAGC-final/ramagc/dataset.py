"""Structured pose feature extraction and variable-length batching for RA-MAGC.

This module consumes already-decoded anatomical coordinates. It intentionally does
not parse ISLTranslate ``.pose`` files, construct data splits, or infer signer IDs.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import DataConfig


UPPER_JOINTS = (
    "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW",
    "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST",
)

HAND_JOINTS = (
    "WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP",
    "INDEX_FINGER_MCP", "INDEX_FINGER_PIP", "INDEX_FINGER_DIP", "INDEX_FINGER_TIP",
    "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP", "MIDDLE_FINGER_TIP",
    "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
    "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP",
)

FACE_POINTS = (
    "33", "133", "159", "145", "362", "263", "386", "374", "70",
    "63", "105", "336", "296", "334", "61", "291", "13", "14",
)

HAND_EDGES = tuple(
    (start, end)
    for finger in (
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 8),
        (0, 9, 10, 11, 12),
        (0, 13, 14, 15, 16),
        (0, 17, 18, 19, 20),
    )
    for start, end in zip(finger, finger[1:])
)

BODY_EDGES = ((0, 1), (0, 2), (2, 4), (1, 3), (3, 5))


def _validate_times(times: np.ndarray, uid: str = "sample") -> np.ndarray:
    times = np.asarray(times, dtype=np.float64)
    if times.ndim != 1:
        raise ValueError(f"{uid}: timestamps must be a one-dimensional array in seconds")
    if times.size == 0:
        raise ValueError(f"{uid}: sequence must contain at least one frame")
    if not np.isfinite(times).all():
        raise ValueError(f"{uid}: timestamps contain non-finite values (NaN or Inf)")
    if times.size > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{uid}: timestamps must be strictly increasing")
    return times


def _validate_mask(mask: Any, expected_shape: tuple[int, int], name: str, uid: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != expected_shape:
        raise ValueError(f"{uid}: {name} must have shape {expected_shape}, got {arr.shape}")
    if arr.dtype != np.bool_:
        if not np.issubdtype(arr.dtype, np.number):
            raise ValueError(f"{uid}: {name} must be boolean or numeric 0/1")
        if not np.isfinite(arr).all() or not np.isin(arr, (0, 1)).all():
            raise ValueError(f"{uid}: {name} must contain only finite 0/1 values")
    return arr.astype(bool, copy=False)


def _validate_region(
    raw_coords: Mapping[str, Any],
    raw_valid: Mapping[str, Any],
    key: str,
    frames: int,
    min_nodes: int,
    uid: str,
) -> tuple[np.ndarray, np.ndarray]:
    if key not in raw_coords or key not in raw_valid:
        raise ValueError(f"{uid}: required region '{key}' is missing from raw_coords/raw_valid")
    coords = np.asarray(raw_coords[key])
    if coords.ndim != 3 or coords.shape[0] != frames or coords.shape[1] < min_nodes or coords.shape[2] < 2:
        raise ValueError(
            f"{uid}: {key} coordinates must have shape [T,{min_nodes} or more,2 or more]; "
            f"got {coords.shape} for T={frames}"
        )
    coords = np.asarray(coords[:, :min_nodes, :2], dtype=np.float32)
    valid = _validate_mask(raw_valid[key], (frames, min_nodes), f"{key} validity mask", uid)
    return coords, valid


def compute_torso_reference(
    body_xy: np.ndarray,
    body_valid: np.ndarray,
    uid: str = "sample",
    epsilon: float = 1e-4,
    times: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Return a temporally interpolated shoulder-centre track and robust scale.

    Valid left/right shoulder pairs define the centre and shoulder width. Missing
    centre frames are linearly interpolated in frame time (with endpoint extension),
    preventing abrupt coordinate jumps when a shoulder disappears for one frame.
    The clip is rejected if no valid shoulder pair exists anywhere.
    """
    body_xy = np.asarray(body_xy, dtype=np.float32)
    body_valid = np.asarray(body_valid, dtype=bool)
    if body_xy.ndim != 3 or body_xy.shape[1:] != (6, 2):
        raise ValueError(f"{uid}: body_xy must have shape [T,6,2], got {body_xy.shape}")
    if body_valid.shape != body_xy.shape[:2]:
        raise ValueError(f"{uid}: body_valid must have shape [T,6], got {body_valid.shape}")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    shoulder_ok = body_valid[:, 0] & body_valid[:, 1]
    diff = body_xy[:, 1] - body_xy[:, 0]
    widths = np.linalg.norm(diff, axis=-1)
    good = shoulder_ok & np.isfinite(widths) & (widths > epsilon)
    indices = np.flatnonzero(good)
    if indices.size == 0:
        raise ValueError(f"{uid}: clip has no valid left/right shoulder pair for torso normalization")

    midpoints = (body_xy[indices, 0] + body_xy[indices, 1]) * 0.5
    interpolation_times = (
        np.arange(body_xy.shape[0], dtype=np.float64)
        if times is None else _validate_times(times, uid)
    )
    if interpolation_times.shape[0] != body_xy.shape[0]:
        raise ValueError(f"{uid}: times length must match body_xy frames")
    centers = np.stack(
        [np.interp(interpolation_times, interpolation_times[indices], midpoints[:, axis]) for axis in range(2)],
        axis=-1,
    ).astype(np.float32)
    scale = max(float(np.median(widths[indices])), epsilon)
    return centers, scale


def normalize_with_torso(
    coords: np.ndarray,
    valid: np.ndarray,
    centers: np.ndarray,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a shared torso coordinate frame to a region ``[T,N,2]``."""
    coords = np.asarray(coords, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    centers = np.asarray(centers, dtype=np.float32)
    if coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError(f"coords must have shape [T,N,2], got {coords.shape}")
    if valid.shape != coords.shape[:2] or centers.shape != (coords.shape[0], 2):
        raise ValueError("valid/centers shapes do not match coords")
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    usable = valid & np.isfinite(coords).all(axis=-1)
    normalized = (coords - centers[:, None, :]) / np.float32(scale)
    normalized = np.where(usable[..., None], normalized, 0.0).astype(np.float32)
    return normalized, usable


def compute_velocity(
    positions: np.ndarray,
    valid: np.ndarray,
    times: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute finite-difference velocity in normalized coordinate units/second.

    A velocity is valid only if the same joint is valid in both adjacent frames.
    No artificial minimum ``dt`` is substituted: timestamps are validated instead.
    """
    positions = np.asarray(positions, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    times = _validate_times(times)
    if positions.ndim != 3 or positions.shape[-1] != 2:
        raise ValueError(f"positions must have shape [T,N,2], got {positions.shape}")
    if valid.shape != positions.shape[:2] or times.shape[0] != positions.shape[0]:
        raise ValueError("positions, valid and times must have matching time dimensions")

    velocity = np.zeros_like(positions, dtype=np.float32)
    velocity_valid = np.zeros_like(valid, dtype=bool)
    if positions.shape[0] > 1:
        dt = np.diff(times).astype(np.float32)
        pair_valid = valid[1:] & valid[:-1]
        diffs = np.diff(positions, axis=0) / dt[:, None, None]
        velocity[1:] = np.where(pair_valid[..., None], diffs, 0.0)
        velocity_valid[1:] = pair_valid
    return velocity, velocity_valid


def build_edge_tensors(
    positions: np.ndarray,
    valid_mask: np.ndarray,
    num_nodes: int,
    edges: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Build per-frame edge attributes ``[dx,dy,length,valid]`` and adjacency."""
    positions = np.asarray(positions, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if positions.ndim != 3 or positions.shape[1:] != (num_nodes, 2):
        raise ValueError(f"positions must have shape [T,{num_nodes},2]")
    if valid_mask.shape != positions.shape[:2]:
        raise ValueError("valid_mask must have shape [T,num_nodes]")
    for i, j in edges:
        if not (0 <= i < num_nodes and 0 <= j < num_nodes) or i == j:
            raise ValueError(f"Invalid graph edge {(i, j)} for {num_nodes} nodes")

    frames = positions.shape[0]
    edge_features = np.zeros((frames, num_nodes, num_nodes, 4), dtype=np.float32)
    adjacency = np.zeros((frames, num_nodes, num_nodes), dtype=bool)
    node_indices = np.arange(num_nodes)
    adjacency[:, node_indices, node_indices] = valid_mask
    edge_features[:, node_indices, node_indices, 3] = valid_mask.astype(np.float32)

    for i, j in edges:
        edge_valid = valid_mask[:, i] & valid_mask[:, j]
        delta = positions[:, j] - positions[:, i]
        length = np.linalg.norm(delta, axis=-1)
        attr = np.stack([delta[:, 0], delta[:, 1], length, edge_valid.astype(np.float32)], axis=-1)
        reverse = np.stack([-delta[:, 0], -delta[:, 1], length, edge_valid.astype(np.float32)], axis=-1)
        edge_features[:, i, j] = np.where(edge_valid[:, None], attr, 0.0)
        edge_features[:, j, i] = np.where(edge_valid[:, None], reverse, 0.0)
        adjacency[:, i, j] = edge_valid
        adjacency[:, j, i] = edge_valid
    return edge_features, adjacency


@dataclass
class StructuredPoseSample:
    """One continuous signing example with precomputed anatomical feature tensors."""

    uid: str
    text: str
    times: np.ndarray
    left_hand_nodes: np.ndarray
    left_hand_edges: np.ndarray
    left_hand_adj: np.ndarray
    left_hand_valid: np.ndarray
    right_hand_nodes: np.ndarray
    right_hand_edges: np.ndarray
    right_hand_adj: np.ndarray
    right_hand_valid: np.ndarray
    body_nodes: np.ndarray
    body_edges: np.ndarray
    body_adj: np.ndarray
    body_valid: np.ndarray
    face_nodes: np.ndarray | None = None
    face_valid: np.ndarray | None = None


def extract_structured_sample(
    uid: str,
    text: str,
    times: np.ndarray,
    segments: np.ndarray | None,
    raw_coords: Mapping[str, Any],
    raw_valid: Mapping[str, Any],
    raw_conf: Mapping[str, Any] | None = None,
    include_face: bool = False,
) -> StructuredPoseSample:
    """Construct the 13-D hand/face and 10-D body node feature tensors."""
    times = _validate_times(times, uid)
    # Convert absolute source timestamps to clip-relative elapsed seconds before
    # storing float32 batch timestamps; this avoids epoch-time precision loss.
    times = times - times[0]
    frames = len(times)
    if not isinstance(raw_coords, Mapping) or not isinstance(raw_valid, Mapping):
        raise TypeError(f"{uid}: raw_coords and raw_valid must be mappings")
    if segments is not None:
        segments = np.asarray(segments)
        if segments.shape != (frames,):
            raise ValueError(f"{uid}: segments must have shape [{frames}]")
        if np.unique(segments).size > 1:
            raise ValueError(f"{uid}: sample contains multiple segment IDs; split it into continuous clips")
    if raw_conf is not None and not isinstance(raw_conf, Mapping):
        raise TypeError(f"{uid}: raw_conf must be a mapping or None")

    body_xy, body_observed = _validate_region(raw_coords, raw_valid, "body", frames, 6, uid)
    left_xy, left_observed = _validate_region(raw_coords, raw_valid, "left_hand", frames, 21, uid)
    right_xy, right_observed = _validate_region(raw_coords, raw_valid, "right_hand", frames, 21, uid)

    def confidence_for(key: str, nodes: int) -> tuple[np.ndarray, np.ndarray]:
        if raw_conf is None or key not in raw_conf or raw_conf[key] is None:
            return np.zeros((frames, nodes), np.float32), np.zeros((frames, nodes), np.float32)
        conf = np.asarray(raw_conf[key], dtype=np.float32)
        if conf.shape[0:1] != (frames,) or conf.ndim != 2 or conf.shape[1] < nodes:
            raise ValueError(f"{uid}: confidence for {key} must have shape [T,{nodes} or more]")
        conf = conf[:, :nodes]
        finite = np.isfinite(conf)
        out_of_range = finite & ((conf < 0.0) | (conf > 1.0))
        if out_of_range.any():
            raise ValueError(f"{uid}: detector confidence for {key} must be within [0,1]")
        clean = np.where(finite, conf, 0.0).astype(np.float32)
        return clean, finite.astype(np.float32)

    # A linearly interpolated centre track prevents fallback-induced jumps when a
    # shoulder pair is temporarily unavailable; the robust scale is clip-global.
    centers, scale = compute_torso_reference(body_xy, body_observed, uid=uid, times=times)
    body_pos, body_pos_valid = normalize_with_torso(body_xy, body_observed, centers, scale)
    body_vel, body_vel_valid = compute_velocity(body_pos, body_pos_valid, times)
    body_conf, body_conf_valid = confidence_for("body", 6)
    body_nodes = np.concatenate(
        [
            body_pos,
            body_vel,
            body_conf[..., None],
            body_conf_valid[..., None],
            body_observed[..., None].astype(np.float32),
            np.zeros((frames, 6, 1), np.float32),
            body_pos_valid[..., None].astype(np.float32),
            body_vel_valid[..., None].astype(np.float32),
        ],
        axis=-1,
    )
    body_edges, body_adj = build_edge_tensors(body_pos, body_pos_valid, 6, BODY_EDGES)

    def build_hand(key: str, coords: np.ndarray, observed: np.ndarray):
        pos, pos_valid = normalize_with_torso(coords, observed, centers, scale)
        vel, vel_valid = compute_velocity(pos, pos_valid, times)
        conf, conf_valid = confidence_for(key, 21)
        wrist_pos = pos[:, 0:1, :]
        wrist_valid = pos_valid[:, 0:1]
        local = np.where((pos_valid & wrist_valid)[..., None], pos - wrist_pos, 0.0)
        local_valid = pos_valid & wrist_valid
        nodes = np.concatenate(
            [
                pos,
                vel,
                local,
                conf[..., None],
                conf_valid[..., None],
                observed[..., None].astype(np.float32),
                np.zeros((frames, 21, 1), np.float32),
                pos_valid[..., None].astype(np.float32),
                vel_valid[..., None].astype(np.float32),
                local_valid[..., None].astype(np.float32),
            ],
            axis=-1,
        )
        edges, adjacency = build_edge_tensors(pos, pos_valid, 21, HAND_EDGES)
        return nodes, edges, adjacency, pos_valid

    left = build_hand("left_hand", left_xy, left_observed)
    right = build_hand("right_hand", right_xy, right_observed)

    face_nodes = None
    face_valid = None
    face_presence = ("face" in raw_coords, "face" in raw_valid)
    if include_face and any(face_presence) and not all(face_presence):
        raise ValueError(f"{uid}: face coordinates and validity must either both be provided or both omitted")
    if include_face and all(face_presence):
        face_xy, face_observed = _validate_region(raw_coords, raw_valid, "face", frames, 18, uid)
        face_pos, face_pos_valid = normalize_with_torso(face_xy, face_observed, centers, scale)
        face_vel, face_vel_valid = compute_velocity(face_pos, face_pos_valid, times)
        face_conf, face_conf_valid = confidence_for("face", 18)
        # FACE_POINTS[0] and FACE_POINTS[4] are the two eye anchors.
        eye_dist = np.linalg.norm(face_pos[:, 4:5] - face_pos[:, 0:1], axis=-1)[..., None]
        eye_valid = face_pos_valid[:, 0:1] & face_pos_valid[:, 4:5] & (eye_dist[..., 0] > 1e-4)
        eye_center = (face_pos[:, 0:1] + face_pos[:, 4:5]) * 0.5
        local_face = np.where(
            eye_valid[..., None],
            (face_pos - eye_center) / np.maximum(eye_dist, 1e-4),
            0.0,
        )
        local_face_valid = face_pos_valid & eye_valid
        face_nodes = np.concatenate(
            [
                face_pos,
                face_vel,
                local_face,
                face_conf[..., None],
                face_conf_valid[..., None],
                face_observed[..., None].astype(np.float32),
                np.zeros((frames, 18, 1), np.float32),
                face_pos_valid[..., None].astype(np.float32),
                face_vel_valid[..., None].astype(np.float32),
                local_face_valid[..., None].astype(np.float32),
            ],
            axis=-1,
        )
        face_valid = face_pos_valid

    return StructuredPoseSample(
        uid=str(uid),
        text=str(text),
        times=times,
        left_hand_nodes=left[0],
        left_hand_edges=left[1],
        left_hand_adj=left[2],
        left_hand_valid=left[3],
        right_hand_nodes=right[0],
        right_hand_edges=right[1],
        right_hand_adj=right[2],
        right_hand_valid=right[3],
        body_nodes=body_nodes,
        body_edges=body_edges,
        body_adj=body_adj,
        body_valid=body_pos_valid,
        face_nodes=face_nodes,
        face_valid=face_valid,
    )


def pad_time_tensor(tensors: Sequence[torch.Tensor], value: float | int | bool = 0) -> torch.Tensor:
    """Right-pad same-shaped feature sequences to the longest sequence in a batch."""
    if not tensors:
        raise ValueError("Cannot collate an empty tensor sequence")
    tail_shape = tensors[0].shape[1:]
    if any(t.ndim != tensors[0].ndim or t.shape[1:] != tail_shape or t.shape[0] < 1 for t in tensors):
        raise ValueError("All tensors must have matching trailing dimensions and non-empty time axes")
    max_len = max(t.shape[0] for t in tensors)
    out = tensors[0].new_full((len(tensors), max_len, *tail_shape), value)
    for index, tensor in enumerate(tensors):
        out[index, : tensor.shape[0]] = tensor
    return out


class RAMAGCCollator:
    """Collate structured examples; target pads are masked for T5 loss and attention."""

    def __init__(self, tokenizer=None, max_target_length: int | None = None):
        self.tokenizer = tokenizer
        self.max_target_length = max_target_length
        if max_target_length is not None and max_target_length <= 0:
            raise ValueError("max_target_length must be positive")

    def __call__(self, samples: list[StructuredPoseSample]) -> dict[str, Any]:
        if not samples:
            raise ValueError("RAMAGCCollator received an empty batch")
        for sample in samples:
            if len(sample.times) < 1:
                raise ValueError(f"Sample {sample.uid} has no frames")

        lengths = torch.tensor([len(sample.times) for sample in samples], dtype=torch.long)
        batch: dict[str, Any] = {
            "lengths": lengths,
            "timestamps": pad_time_tensor([torch.as_tensor(s.times, dtype=torch.float32) for s in samples]),
            "left_hand_nodes": pad_time_tensor([torch.as_tensor(s.left_hand_nodes, dtype=torch.float32) for s in samples]),
            "left_hand_edges": pad_time_tensor([torch.as_tensor(s.left_hand_edges, dtype=torch.float32) for s in samples]),
            "left_hand_adj": pad_time_tensor([torch.as_tensor(s.left_hand_adj, dtype=torch.bool) for s in samples]),
            "left_hand_valid": pad_time_tensor([torch.as_tensor(s.left_hand_valid, dtype=torch.bool) for s in samples]),
            "right_hand_nodes": pad_time_tensor([torch.as_tensor(s.right_hand_nodes, dtype=torch.float32) for s in samples]),
            "right_hand_edges": pad_time_tensor([torch.as_tensor(s.right_hand_edges, dtype=torch.float32) for s in samples]),
            "right_hand_adj": pad_time_tensor([torch.as_tensor(s.right_hand_adj, dtype=torch.bool) for s in samples]),
            "right_hand_valid": pad_time_tensor([torch.as_tensor(s.right_hand_valid, dtype=torch.bool) for s in samples]),
            "body_nodes": pad_time_tensor([torch.as_tensor(s.body_nodes, dtype=torch.float32) for s in samples]),
            "body_edges": pad_time_tensor([torch.as_tensor(s.body_edges, dtype=torch.float32) for s in samples]),
            "body_adj": pad_time_tensor([torch.as_tensor(s.body_adj, dtype=torch.bool) for s in samples]),
            "body_valid": pad_time_tensor([torch.as_tensor(s.body_valid, dtype=torch.bool) for s in samples]),
            "uids": [s.uid for s in samples],
            "texts": [s.text for s in samples],
        }

        if any(sample.face_nodes is not None for sample in samples):
            face_nodes = [
                torch.as_tensor(s.face_nodes, dtype=torch.float32)
                if s.face_nodes is not None else torch.zeros((len(s.times), 18, 13), dtype=torch.float32)
                for s in samples
            ]
            face_valid = [
                torch.as_tensor(s.face_valid, dtype=torch.bool)
                if s.face_valid is not None else torch.zeros((len(s.times), 18), dtype=torch.bool)
                for s in samples
            ]
            batch["face_nodes"] = pad_time_tensor(face_nodes)
            batch["face_valid"] = pad_time_tensor(face_valid)
        else:
            batch["face_nodes"] = None
            batch["face_valid"] = None

        if self.tokenizer is not None:
            token_args = {
                "padding": True,
                "truncation": self.max_target_length is not None,
                "return_tensors": "pt",
            }
            if self.max_target_length is not None:
                token_args["max_length"] = self.max_target_length
            encoded = self.tokenizer([s.text for s in samples], **token_args)
            if "input_ids" not in encoded or "attention_mask" not in encoded:
                raise ValueError("Tokenizer must return input_ids and attention_mask")
            target_mask = encoded["attention_mask"].bool()
            batch["labels"] = encoded["input_ids"].masked_fill(~target_mask, -100)
            batch["decoder_attention_mask"] = target_mask.long()

        return batch


class RAMAGCDataset(Dataset):
    """In-memory Dataset over pre-extracted ``StructuredPoseSample`` records."""

    def __init__(self, samples: list[StructuredPoseSample]):
        if not isinstance(samples, list):
            raise TypeError("samples must be a list of StructuredPoseSample objects")
        if not all(isinstance(sample, StructuredPoseSample) for sample in samples):
            raise TypeError("Every item must be a StructuredPoseSample")
        self.samples = samples
        self.lengths = [len(sample.times) for sample in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> StructuredPoseSample:
        return self.samples[index]
