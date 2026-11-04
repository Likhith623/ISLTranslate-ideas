"""Timestamp-aware video crops and explicit frozen RGB caches. No implicit weight downloads."""

from pathlib import Path

import numpy as np
import torch
from torch import nn

from isl_slt.config import Config
from isl_slt.data.manifests import Record
from isl_slt.data.pose import PoseSequence
from isl_slt.features.cache import preprocessing_fingerprint
from isl_slt.utils.io import atomic_open, file_hash, fingerprint, read_json, write_json


def decode_at_times(path: str, times: np.ndarray, tolerance: float) -> tuple[list, np.ndarray]:
    import cv2
    capture = cv2.VideoCapture(path)
    if not capture.isOpened():
        raise ValueError(f"Cannot decode video: {path}")
    output, valid, previous, current, last_time = [], [], None, None, -1.0
    try:
        for desired in times:
            while current is None or current[0] < desired:
                ok, frame = capture.read()
                if not ok:
                    break
                actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
                if not np.isfinite(actual) or actual <= last_time:
                    raise ValueError("Video backend did not provide increasing presentation timestamps")
                last_time = actual
                previous, current = current, (actual, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            choices = [value for value in (previous, current) if value is not None]
            closest = min(choices, key=lambda value: abs(value[0] - desired)) if choices else None
            matched = closest is not None and abs(closest[0] - desired) <= tolerance
            output.append(closest[1] if matched else None)
            valid.append(matched)
    finally:
        capture.release()
    return output, np.asarray(valid, bool)


def square_crop(frame: np.ndarray, points: np.ndarray, valid: np.ndarray, size: int, padding: float = 0.25):
    import cv2
    usable = points[valid & np.isfinite(points).all(axis=-1)]
    if len(usable) < 2:
        return None
    low, high = np.percentile(usable, [2, 98], axis=0)
    side = max(float(np.max(high - low)) * (1 + padding * 2), 8)
    center = (low + high) / 2
    x, y = np.floor(center - side / 2).astype(int)
    side = int(np.ceil(side))
    height, width = frame.shape[:2]
    left, top, right, bottom = max(x, 0), max(y, 0), min(x + side, width), min(y + side, height)
    if right <= left or bottom <= top:
        return None
    canvas = np.zeros((side, side, 3), np.uint8)
    canvas[top - y:bottom - y, left - x:right - x] = frame[top:bottom, left:right]
    return cv2.resize(canvas, (size, size), interpolation=cv2.INTER_LINEAR)


def load_crops(center: Record, source_records: list[Record], sequence: PoseSequence, config: Config):
    """Return [T, views, C, H, W], masks; failed hand views use upper-body pixels with false masks."""
    if center.image_transform is None:
        raise ValueError(f"{center.uid}: RGB requires a verified pose-pixel to video-pixel affine transform")
    views = 3 if config.rgb.hand_crops else 1
    images = np.zeros((len(sequence.times), views, config.rgb.crop_size, config.rgb.crop_size, 3), np.uint8)
    masks = np.zeros((len(sequence.times), views), bool)
    from isl_slt.data.context import SourceIndex
    indexed = source_records if isinstance(source_records, SourceIndex) else SourceIndex(source_records)
    relevant = indexed.dependencies(center, config.context) if config.context.enabled else [center]
    if not config.context.enabled:
        relevant = [center]
    owners = []
    for time in sequence.times:
        candidates = [r for r in relevant if not config.context.enabled or r.start_seconds <= time < r.end_seconds]
        if not candidates:
            raise ValueError(f"No video covers context timestamp {time}")
        owners.append(min(candidates, key=lambda r: (r.key != center.key, r.uid)))
    for record in {r.key: r for r in owners}.values():
        if not record.video_path or record.image_transform is None:
            raise ValueError(f"{record.uid}: RGB needs a video and explicit coordinate transform")
        indices = np.asarray([i for i, owner in enumerate(owners) if owner.key == record.key])
        local_times = sequence.times[indices] - (record.start_seconds if record.timing_verified else 0)
        frames, decoded = decode_at_times(record.video_path, local_times, config.rgb.sync_tolerance_seconds)
        transform = np.asarray(record.image_transform, np.float32)
        xy = sequence.coordinates[..., :2]
        homogeneous = np.concatenate([xy, np.ones((*xy.shape[:2], 1), np.float32)], axis=-1)
        transformed = homogeneous @ transform.T
        valid = sequence.valid[..., :2].all(axis=-1)
        body_indices = [i for i, (component, point) in enumerate(sequence.names)
                        if component in ("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS") or
                        component == "POSE_LANDMARKS" and point in (
                            "LEFT_SHOULDER", "RIGHT_SHOULDER", "LEFT_ELBOW", "RIGHT_ELBOW", "LEFT_WRIST", "RIGHT_WRIST")]
        stable_points, stable_valid = transformed[:, body_indices].reshape(-1, 2), valid[:, body_indices].reshape(-1)
        for index, frame, ok in zip(indices, frames, decoded):
            if not ok:
                continue
            upper = square_crop(frame, stable_points, stable_valid, config.rgb.crop_size)
            if upper is None:
                raise ValueError(f"{record.uid}: no usable stable upper-body crop")
            images[index, 0], masks[index, 0] = upper, True
            if config.rgb.hand_crops:
                for view, component in enumerate(("LEFT_HAND_LANDMARKS", "RIGHT_HAND_LANDMARKS"), start=1):
                    points = [j for j, name in enumerate(sequence.names) if name[0] == component]
                    hand = square_crop(frame, transformed[index, points], valid[index, points], config.rgb.crop_size)
                    images[index, view] = hand if hand is not None else upper
                    masks[index, view] = hand is not None
    tensor = torch.from_numpy(images.transpose(0, 1, 4, 2, 3).copy()).float() / 255
    mean = tensor.new_tensor([0.485, 0.456, 0.406])[None, None, :, None, None]
    std = tensor.new_tensor([0.229, 0.224, 0.225])[None, None, :, None, None]
    return (tensor - mean) / std, torch.from_numpy(masks)


class RGBEncoder(nn.Module):
    def __init__(self, weights_path: str | None):
        super().__init__()
        from torchvision.models import resnet18
        network = resnet18(weights=None)
        if weights_path is not None:
            path = Path(weights_path)
            if not path.is_file():
                raise FileNotFoundError("Prepare local ResNet18 weights with prepare-model --allow-network")
            network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        self.network = nn.Sequential(*list(network.children())[:-1])
        self.tuning = False
        self.set_tuning(False)

    def set_tuning(self, enabled: bool) -> None:
        self.tuning = enabled
        for parameter in self.network.parameters():
            parameter.requires_grad_(False)
        if enabled:
            for parameter in self.network[7].parameters():
                parameter.requires_grad_(True)
        self.train(self.training)

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep every batch-normalization statistic fixed, even during final residual-block tuning.
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
        return self

    def forward(self, images: torch.Tensor, chunk_size: int = 64) -> torch.Tensor:
        shape = images.shape[:-3]
        flat = images.reshape(-1, *images.shape[-3:])
        # Chunking bounds temporary CNN activation size without dropping frames.
        outputs = [self.network(chunk).flatten(1) for chunk in flat.split(chunk_size)]
        return torch.cat(outputs).reshape(*shape, 512)


def rgb_signature(config: Config, center: Record, dependencies: list[Record],
                  verify_assets: bool = True, weights_sha256: str | None = None, signatures=None) -> dict:
    from isl_slt.features.cache import SignatureSession
    signatures = signatures or SignatureSession()
    dependency_signatures = []
    for record in sorted(dependencies, key=lambda r: r.key):
        if not record.video_path or record.image_transform is None:
            raise ValueError(f"{record.uid}: RGB requires video and verified coordinate-transform metadata")
        if "video" not in record.asset_hashes:
            raise ValueError(f"{record.uid}: audited video checksum is required")
        video_hash = signatures.hashes.hash(record.video_path) if verify_assets else record.asset_hashes["video"]
        if record.asset_hashes.get("video", video_hash) != video_hash:
            raise ValueError(f"{record.uid}: video bytes changed since immutable manifest creation")
        signature = dict(signatures.signature(record, verify_assets))
        if config.context.enabled:
            signature.pop("source")
        dependency_signatures.append({**signature, "video_sha256": video_hash, "image_transform": record.image_transform})
    return {"version": "rgb-context-cache-v2" if config.context.enabled else "rgb-cache-v1",
            **({"context_source": {"source": center.source_id, "partition": center.split}} if config.context.enabled else {}),
            "pose_preprocessing": preprocessing_fingerprint(config),
            "settings": config.rgb.model_dump(mode="json"),
            "weights_sha256": signatures.hashes.hash(config.rgb.weights_path) if verify_assets else weights_sha256,
            "center": center.key, "dependencies": dependency_signatures}


def rgb_paths(config: Config, record: Record) -> tuple[Path, Path]:
    descriptor = fingerprint({"pose": preprocessing_fingerprint(config), "rgb": config.rgb.model_dump(mode="json")})
    path = Path(config.data.rgb_dir) / descriptor / record.key
    return path.with_suffix(".npz"), path.with_suffix(".json")


def write_rgb(config: Config, record: Record, dependencies: list[Record], values: np.ndarray,
              masks: np.ndarray, times: np.ndarray, signatures=None) -> None:
    arrays, meta = rgb_paths(config, record)
    with atomic_open(arrays) as stream:
        np.savez_compressed(stream, values=values, masks=masks, times=times)
    write_json(meta, {"signature": rgb_signature(config, record, dependencies, signatures=signatures), "sha256": file_hash(arrays)})


def read_rgb(config: Config, record: Record, dependencies: list[Record], times: np.ndarray,
             verify_assets: bool = False, verify_arrays: bool = True, signatures=None):
    arrays, meta = rgb_paths(config, record)
    metadata = read_json(meta)
    incompatible_assets = metadata["signature"] != rgb_signature(config, record, dependencies, verify_assets,
                                                                metadata["signature"]["weights_sha256"], signatures)
    if incompatible_assets or (verify_arrays and metadata["sha256"] != file_hash(arrays)):
        raise ValueError(f"{record.uid}: incompatible RGB cache; regenerate it explicitly")
    with np.load(arrays, allow_pickle=False) as loaded:
        if not np.array_equal(times, loaded["times"]):
            raise ValueError("Pose/RGB timelines differ")
        return torch.from_numpy(loaded["values"].copy()), torch.from_numpy(loaded["masks"].copy())
