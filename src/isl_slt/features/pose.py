"""Positions, motion and independently ablatable anatomy features."""

from dataclasses import dataclass

import numpy as np

from isl_slt.config import Features
from isl_slt.data.pose import PoseSequence

BODY = "POSE_LANDMARKS"
LEFT, RIGHT = "LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"
UPPER = ("LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST")
HAND = ("WRIST", "THUMB_CMC", "THUMB_MCP", "THUMB_IP", "THUMB_TIP", "INDEX_FINGER_MCP", "INDEX_FINGER_PIP",
        "INDEX_FINGER_DIP", "INDEX_FINGER_TIP", "MIDDLE_FINGER_MCP", "MIDDLE_FINGER_PIP", "MIDDLE_FINGER_DIP",
        "MIDDLE_FINGER_TIP", "RING_FINGER_MCP", "RING_FINGER_PIP", "RING_FINGER_DIP", "RING_FINGER_TIP",
        "PINKY_MCP", "PINKY_PIP", "PINKY_DIP", "PINKY_TIP")
FACE = ("33", "133", "159", "145", "362", "263", "386", "374", "70", "63", "105", "336", "296", "334",
        "61", "291", "13", "14")
HAND_EDGES = tuple((start, end) for finger in ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12),
                                             (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
                   for start, end in zip(finger, finger[1:]))


@dataclass
class Prepared:
    values: np.ndarray
    times: np.ndarray
    observed: np.ndarray
    imputed: np.ndarray
    position_valid: np.ndarray
    velocity_valid: np.ndarray
    regions: np.ndarray
    segments: np.ndarray
    continuous_valid: np.ndarray
    metadata: dict


def selected_joints(settings: Features) -> tuple[tuple[str, str], ...]:
    names = tuple((BODY, name) for name in UPPER) + tuple((component, name) for component in (LEFT, RIGHT) for name in HAND)
    return names + tuple(("FACE_LANDMARKS", name) for name in FACE) if settings.face else names


def interpolate_short_gaps(values: np.ndarray, valid: np.ndarray, times: np.ndarray, segments: np.ndarray,
                           maximum_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    result, imputed = values.copy(), np.zeros_like(valid)
    for joint in range(values.shape[1]):
        observations = np.flatnonzero(valid[:, joint])
        for left, right in zip(observations, observations[1:]):
            if right <= left + 1 or segments[left] != segments[right]:
                continue
            # Bound the complete interval between observed endpoints, independent of FPS.
            duration = times[right] - times[left]
            if 0 < duration <= maximum_seconds + 1e-10:
                alpha = (times[left + 1:right] - times[left]) / duration
                result[left + 1:right, joint] = values[left, joint] + alpha[:, None] * (
                    values[right, joint] - values[left, joint])
                imputed[left + 1:right, joint] = True
    return result, imputed


def motion(values: np.ndarray, valid: np.ndarray, times: np.ndarray, segments: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError("Motion requires increasing finite timestamps")
    velocity, mask = np.zeros_like(values), np.zeros_like(valid)
    dt = np.diff(times)
    mask[1:] = valid[1:] & valid[:-1] & (segments[1:] == segments[:-1])[:, None]
    differences = (values[1:] - values[:-1]) / dt[:, None, None]
    velocity[1:] = np.where(mask[1:, :, None], differences, 0)
    return velocity, mask


def normalize_positions(xy: np.ndarray, valid: np.ndarray, segments: np.ndarray, epsilon: float):
    normalized, usable = np.zeros_like(xy), np.zeros_like(valid)
    anchors = valid[:, 0] & valid[:, 1]
    widths = np.linalg.norm(xy[:, 0] - xy[:, 1], axis=-1)
    anchors &= np.isfinite(widths) & (widths > epsilon)
    for segment in np.unique(segments):
        frames = segments == segment
        good = frames & anchors
        if not good.any():
            continue
        centers = (xy[:, 0] + xy[:, 1]) / 2
        fallback = np.median(centers[good], axis=0)
        center = np.where(anchors[:, None], centers, fallback)
        scale = max(float(np.median(widths[good])), epsilon)
        normalized[frames] = (xy[frames] - center[frames, None]) / scale
        usable[frames] = valid[frames]
    normalized = np.where(usable[..., None], normalized, 0)
    if not usable.any():
        raise ValueError("No segment contains usable shoulder anchors; fix the pose schema or asset")
    return normalized, usable, anchors


def prepare_pose(sequence: PoseSequence, settings: Features, context_enabled: bool = False) -> Prepared:
    if settings.schema_version != "upper-hands-v1":
        raise ValueError(f"Unknown feature schema: {settings.schema_version}")
    names = selected_joints(settings)
    lookup = {name: index for index, name in enumerate(sequence.names)}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"Requested components/joints absent from actual pose header: {missing}")
    indices = [lookup[name] for name in names]
    xy = sequence.coordinates[:, indices, :2]
    observed = sequence.valid[:, indices, :2].all(axis=-1)
    xy = np.where(observed[..., None], xy, 0)
    positions, usable, anchors = normalize_positions(xy, observed, sequence.segments, settings.epsilon)
    positions, imputed = interpolate_short_gaps(positions, usable, sequence.times, sequence.segments,
                                               settings.interpolation_seconds)
    position_valid = usable | imputed
    velocity, velocity_valid = motion(positions, usable & anchors[:, None], sequence.times, sequence.segments)
    blocks = [positions.reshape(len(xy), -1), velocity.reshape(len(xy), -1), observed.astype(np.float32),
              imputed.astype(np.float32), position_valid.astype(np.float32), velocity_valid.astype(np.float32)]
    validity = [np.repeat(position_valid, 2, axis=1), np.repeat(velocity_valid, 2, axis=1),
                *[np.zeros_like(observed) for _ in range(4)]]
    continuous = [True, True, False, False, False, False]
    if not settings.velocity:
        blocks = [blocks[index] for index in (0, 2, 3, 4)]
        validity = [validity[index] for index in (0, 2, 3, 4)]
        continuous = [True, False, False, False]
    if settings.local_hands:
        for start in (6, 27):
            local_valid = position_valid[:, start:start + 21] & position_valid[:, start:start + 1]
            local = positions[:, start:start + 21] - positions[:, start:start + 1]
            blocks.extend([np.where(local_valid[..., None], local, 0).reshape(len(xy), -1), local_valid.astype(np.float32)])
            validity.extend([np.repeat(local_valid, 2, axis=1), np.zeros_like(local_valid)])
            continuous.extend([True, False])
    if settings.bones:
        edges = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5)]
        edges += [(start + a, start + b) for start in (6, 27) for a, b in HAND_EDGES]
        bone_valid = np.stack([position_valid[:, a] & position_valid[:, b] for a, b in edges], axis=1)
        bones = np.stack([positions[:, b] - positions[:, a] for a, b in edges], axis=1)
        blocks.extend([np.where(bone_valid[..., None], bones, 0).reshape(len(xy), -1), bone_valid.astype(np.float32)])
        validity.extend([np.repeat(bone_valid, 2, axis=1), np.zeros_like(bone_valid)])
        continuous.extend([True, False])
    if settings.depth_scales:
        for component, scale in settings.depth_scales.items():
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Explicit component-local depth scales must be finite and positive")
            depth_indices = [index for index, name in enumerate(names) if name[0] == component]
            if not depth_indices:
                raise ValueError(f"Depth requested for unselected component {component}")
            raw_indices = [indices[index] for index in depth_indices]
            zvalid = sequence.valid[:, raw_indices, 2]
            if not zvalid.any():
                raise ValueError(f"No valid depth for {component}")
            depth = sequence.coordinates[:, raw_indices, 2] / scale
            blocks.extend([np.where(zvalid, depth, 0), zvalid.astype(np.float32)])
            validity.extend([zvalid, np.zeros_like(zvalid)])
            continuous.extend([True, False])
    if context_enabled:
        blocks.append(np.stack([sequence.regions == region for region in (-1, 0, 1)], axis=1).astype(np.float32))
        validity.append(np.zeros_like(blocks[-1], dtype=bool))
        continuous.append(False)
    values = np.concatenate(blocks, axis=1).astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Prepared features contain nonfinite values")
    metadata = {"schema": settings.schema_version, "joints": [list(name) for name in names],
                "normalization": "valid-shoulder-midpoint-frame-center/segment-median-width-v1",
                "input_dim": values.shape[1], "sequence": sequence.metadata,
                "continuous_channels": np.flatnonzero(np.concatenate([
                    np.full(block.shape[1], enabled, dtype=bool) for block, enabled in zip(blocks, continuous)
                ])).tolist(),
                "full_duration_resample_fps": settings.resample_fps}
    return Prepared(values, sequence.times, observed, imputed, position_valid, velocity_valid,
                    sequence.regions, sequence.segments, np.concatenate(validity, axis=1), metadata)


def resample_sequence(sequence: PoseSequence, fps: float | None) -> PoseSequence:
    """Explicit nearest-observation full-duration resampling, jointly applied to all pose channels."""
    if fps is None or len(sequence.times) < 2:
        return sequence
    grid = np.arange(sequence.times[0], sequence.times[-1] + 1e-9, 1 / fps)
    if not len(grid) or sequence.times[-1] - grid[-1] > 1e-8:
        grid = np.r_[grid, sequence.times[-1]]
    right = np.searchsorted(sequence.times, grid).clip(0, len(sequence.times) - 1)
    left = (right - 1).clip(0)
    indices = np.where(abs(sequence.times[left] - grid) <= abs(sequence.times[right] - grid), left, right)
    valid = sequence.valid[indices].copy()
    # Do not fill a missing physical interval by repeatedly selecting a distant observation.
    valid[abs(sequence.times[indices] - grid) > 1 / fps] = False
    return PoseSequence(sequence.coordinates[indices], valid, sequence.names, grid, sequence.segments[indices],
                        sequence.regions[indices], sequence.width, sequence.height,
                        {**sequence.metadata, "resampled_fps": fps, "original_frames": len(sequence.times)})
