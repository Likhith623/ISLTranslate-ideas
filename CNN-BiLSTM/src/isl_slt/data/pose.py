"""Header-driven .pose reader. Never imports MediaPipe or assumes a landmark storage order."""

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np

from isl_slt.config import Features
from isl_slt.data.manifests import Record
from isl_slt.utils.io import read_json


@dataclass
class PoseSequence:
    coordinates: np.ndarray  # [time, joints, XYZ]; absent depth stays invalid, not invented.
    valid: np.ndarray  # [time, joints, XYZ]
    names: tuple[tuple[str, str], ...]
    times: np.ndarray  # seconds, source-relative when verified bounds exist
    segments: np.ndarray  # real continuity regions; motion must never cross a region boundary
    regions: np.ndarray  # -1 previous, 0 center, +1 following
    width: float
    height: float
    metadata: dict


def decode_pose(pose, record: Record, settings: Features) -> PoseSequence:
    data = np.ma.asarray(pose.body.data)
    confidence = np.ma.asarray(pose.body.confidence)
    if data.ndim != 4 or confidence.shape != data.shape[:-1] or not data.shape[0]:
        raise ValueError(f"{record.uid}: expected nonempty [T, people, joints, dimensions] pose data")
    person = record.person_index if record.person_index is not None else settings.person_index
    if person is None:
        if data.shape[1] != 1:
            raise ValueError(f"{record.uid}: {data.shape[1]} people; specify person_index explicitly")
        person = 0
    if person >= data.shape[1]:
        raise ValueError(f"{record.uid}: person_index exceeds stored people")
    names, component_formats = [], {}
    for component in pose.header.components:
        aliases = [key for key, value in settings.component_names.items() if value == component.name]
        if len(aliases) > 1:
            raise ValueError("Stored component maps to multiple canonical names")
        component_name = aliases[0] if aliases else component.name
        if component.format not in ("XYC", "XYZC"):
            raise ValueError(f"Unsupported component format {component.name}:{component.format}; expected XYC/XYZC")
        if component_name in component_formats:
            raise ValueError(f"Ambiguous duplicate component {component.name}")
        component_formats[component_name] = component.format
        names.extend((component_name, str(point)) for point in component.points)
    if len(names) != data.shape[2] or len(set(names)) != len(names):
        raise ValueError("Pose header point names/counts do not uniquely describe the body array")
    array = np.asarray(data.data[:, person], dtype=np.float32)
    masked = np.ma.getmaskarray(data)[:, person]
    coordinates = np.full((*array.shape[:2], 3), np.nan, dtype=np.float32)
    valid = np.zeros_like(coordinates, dtype=bool)
    conf = np.asarray(confidence.data[:, person])
    conf_mask = np.ma.getmaskarray(confidence)[:, person]
    for joint, (component, _) in enumerate(names):
        dimensions = len(component_formats[component]) - 1
        if dimensions > array.shape[-1]:
            raise ValueError("Component dimensions exceed stored coordinate dimensions")
        coordinates[:, joint, :dimensions] = array[:, joint, :dimensions]
        validity = ~masked[:, joint, :dimensions] & np.isfinite(array[:, joint, :dimensions])
        if component in settings.confidence_thresholds:
            threshold = settings.confidence_thresholds[component]
            validity &= (~conf_mask[:, joint] & np.isfinite(conf[:, joint]) &
                         (conf[:, joint] >= threshold))[:, None]
        valid[:, joint, :dimensions] = validity
    width, height = float(pose.header.dimensions.width), float(pose.header.dimensions.height)
    if record.coordinate_space == "normalized_image":
        if not np.isfinite([width, height]).all() or min(width, height) <= 0:
            raise ValueError("Normalized image coordinates require valid header width/height")
        coordinates[..., 0] *= width
        coordinates[..., 1] *= height
    stored_fps = float(pose.body.fps)
    fps = record.fps or stored_fps
    if record.fps and stored_fps > 0 and abs(record.fps - stored_fps) / record.fps > 0.01:
        raise ValueError(f"{record.uid}: annotation FPS conflicts with .pose header")
    if record.timestamps_path:
        times = np.asarray(read_json(record.timestamps_path), dtype=np.float64)
        if not np.isfinite(fps) or fps <= 0:
            fps = 0.0  # Unknown extraction FPS, not an invented timing calibration.
    else:
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"{record.uid}: supply real timestamps or positive verified extraction FPS")
        times = np.arange(len(array), dtype=np.float64) / fps
    if times.shape != (len(array),) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f"{record.uid}: frame timestamps must be finite and strictly increasing")
    sampling_interval = float(np.median(np.diff(times))) if len(times) > 1 else 1 / max(fps, 1)
    if record.timing_verified:
        if times[0] < -1e-6 or times[-1] > record.end_seconds - record.start_seconds + 1e-6:
            raise ValueError("Frame offsets do not fit verified source clip bounds")
        times = times + record.start_seconds
    segments = np.cumsum(np.r_[False, np.diff(times) > settings.discontinuity_seconds]).astype(np.int64)
    return PoseSequence(coordinates, valid, tuple(names), times, segments, np.zeros(len(times), np.int8),
                        width, height, {"uid": record.uid, "person_index": person, "fps": fps,
                        "sampling_interval_seconds": sampling_interval,
                        "components": component_formats, "coordinate_space": "image_pixels",
                        "stored_confidence_policy": "Preserve pose-format masks; apply configured component thresholds only"})


def read_pose(record: Record, settings: Features) -> PoseSequence:
    from pose_format import Pose
    with Path(record.pose_path).open("rb") as stream:
        data = stream.read()
    if "pose" in record.asset_hashes and hashlib.sha256(data).hexdigest() != record.asset_hashes["pose"]:
        raise ValueError(f"{record.uid}: pose bytes changed since immutable manifest creation")
    pose = Pose.read(data)
    return decode_pose(pose, record, settings)
