"""Manifest-based RGB+pose dataset, timestamp alignment, ROI derivation and split checks.

Each record's NPZ schema is documented in docs/data_contract.md. The loader does
not guess the binary layout of a ``.pose`` file: convert it through the parent
ISLTranslate pose parser and export the documented arrays first.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


REGION_ORDER = ("face", "left_hand", "right_hand", "body")


def validate_times(times: np.ndarray, name: str) -> None:
    if times.ndim != 1 or len(times) < 1 or not np.isfinite(times).all():
        raise ValueError(f"{name} must be a nonempty finite vector")
    if len(times) > 1 and np.any(np.diff(times) <= 0):
        raise ValueError(f"{name} must be strictly increasing")


def letterbox_frames(frame_tensor: torch.Tensor, output_size: int) -> tuple[torch.Tensor, dict[str, float]]:
    """Resize frames without changing aspect ratio, then pad to a square canvas.

    Returns frames [T,3,S,S] and affine scale/pad values in original pixels.
    Pose points and ROI boxes must be transformed with the same affine map.
    """
    if frame_tensor.ndim != 4 or frame_tensor.shape[1] != 3:
        raise ValueError("frame_tensor must have shape [T,3,H,W]")
    if output_size < 2:
        raise ValueError("output_size must be >= 2")
    _, _, height, width = frame_tensor.shape
    scale = min(output_size / height, output_size / width)
    resized_h = max(1, min(output_size, int(round(height * scale))))
    resized_w = max(1, min(output_size, int(round(width * scale))))
    resized = F.interpolate(frame_tensor, size=(resized_h, resized_w),
                            mode="bilinear", align_corners=False)
    top = (output_size - resized_h) // 2
    bottom = output_size - resized_h - top
    left = (output_size - resized_w) // 2
    right = output_size - resized_w - left
    boxed = F.pad(resized, (left, right, top, bottom), value=0.0)
    return boxed, {
        # Multipliers act on normalized original-frame coordinates.
        "scale_x": resized_w / output_size, "scale_y": resized_h / output_size,
        "left": float(left), "top": float(top), "output_size": float(output_size),
    }


def transform_points_to_letterbox(points: np.ndarray, affine: dict[str, float]) -> np.ndarray:
    """Transform normalized original-frame x/y points to the letterboxed canvas."""
    out = np.asarray(points, dtype=np.float32).copy()
    size = affine["output_size"]
    out[..., 0] = (out[..., 0] * affine["scale_x"] + affine["left"] / size)
    out[..., 1] = (out[..., 1] * affine["scale_y"] + affine["top"] / size)
    return out


def transform_boxes_to_letterbox(boxes: np.ndarray, affine: dict[str, float]) -> np.ndarray:
    """Transform normalized xyxy boxes from original frame to letterboxed canvas."""
    out = np.asarray(boxes, dtype=np.float32).copy()
    size = affine["output_size"]
    out[..., [0, 2]] = out[..., [0, 2]] * affine["scale_x"] + affine["left"] / size
    out[..., [1, 3]] = out[..., [1, 3]] * affine["scale_y"] + affine["top"] / size
    return np.clip(out, 0.0, 1.0)


def align_pose_to_video(
    video_times: np.ndarray,
    pose_times: np.ndarray,
    pose_xy: np.ndarray,
    pose_confidence: np.ndarray,
    pose_valid: np.ndarray,
    max_gap_seconds: float = 0.25,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Linearly align pose to video time without inventing long-gap observations.

    Interpolation is performed independently per landmark. Both bracketing
    landmarks must be valid and their time gap <= max_gap_seconds. At exact
    observations, the source point is copied. Output shape is [Tv,66,(2/1/1)].
    """
    validate_times(video_times, "video_times")
    validate_times(pose_times, "pose_times")
    if pose_xy.shape != (len(pose_times), 66, 2):
        raise ValueError("pose_xy must have shape [Tp,66,2]")
    if pose_confidence.shape != (len(pose_times), 66) or pose_valid.shape != (len(pose_times), 66):
        raise ValueError("pose confidence/valid arrays must have shape [Tp,66]")
    if max_gap_seconds < 0:
        raise ValueError("max_gap_seconds cannot be negative")
    out_xy = np.zeros((len(video_times), 66, 2), dtype=np.float32)
    out_conf = np.zeros((len(video_times), 66), dtype=np.float32)
    out_valid = np.zeros((len(video_times), 66), dtype=bool)
    right = np.searchsorted(pose_times, video_times, side="left")
    for i, (tv, r) in enumerate(zip(video_times, right)):
        if r < len(pose_times) and abs(pose_times[r] - tv) <= 1e-7:
            ok = pose_valid[r]
            out_xy[i, ok] = pose_xy[r, ok]
            out_conf[i, ok] = pose_confidence[r, ok]
            out_valid[i] = ok
            continue
        if r == 0 or r >= len(pose_times):
            continue
        l = r - 1
        gap = pose_times[r] - pose_times[l]
        if gap <= 0 or gap > max_gap_seconds:
            continue
        ok = pose_valid[l] & pose_valid[r]
        alpha = float((tv - pose_times[l]) / gap)
        out_xy[i, ok] = (1.0 - alpha) * pose_xy[l, ok] + alpha * pose_xy[r, ok]
        out_conf[i, ok] = (1.0 - alpha) * pose_confidence[l, ok] + alpha * pose_confidence[r, ok]
        out_valid[i] = ok
    return out_xy, out_conf, out_valid


def _bbox(points: np.ndarray, valid: np.ndarray, margin: float) -> tuple[np.ndarray, bool]:
    p = points[valid]
    if p.shape[0] < 2 or not np.isfinite(p).all():
        return np.zeros(4, np.float32), False
    lo = p.min(axis=0)
    hi = p.max(axis=0)
    centre = (lo + hi) / 2
    size = np.maximum(hi - lo, 0.025) * margin
    lo, hi = np.clip(centre - size / 2, 0, 1), np.clip(centre + size / 2, 0, 1)
    if np.any(hi <= lo):
        return np.zeros(4, np.float32), False
    return np.asarray([lo[0], lo[1], hi[0], hi[1]], np.float32), True


def derive_rois(pose_xy: np.ndarray, pose_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Derive conservative face/hand/body normalized boxes from the 66-point schema."""
    t = len(pose_xy)
    boxes = np.zeros((t, 4, 4), dtype=np.float32)
    valid = np.zeros((t, 4), dtype=bool)
    for i in range(t):
        boxes[i, 0], valid[i, 0] = _bbox(pose_xy[i, 48:66], pose_valid[i, 48:66], 2.0)
        boxes[i, 1], valid[i, 1] = _bbox(pose_xy[i, 0:21], pose_valid[i, 0:21], 2.1)
        boxes[i, 2], valid[i, 2] = _bbox(pose_xy[i, 21:42], pose_valid[i, 21:42], 2.1)
        # Upper-body ROI covers shoulders, elbows, wrists and both hands.
        boxes[i, 3], valid[i, 3] = _bbox(pose_xy[i, 0:48].reshape(-1, 2),
                                        pose_valid[i, 0:48].reshape(-1), 1.8)
        if not valid[i, 3]:
            # Full frame fallback is explicitly marked as a fallback, not as a detected body box.
            boxes[i, 3] = np.asarray([0, 0, 1, 1], np.float32)
            valid[i, 3] = True
    return boxes, valid


@dataclass
class ManifestRecord:
    uid: str
    npz: str
    translation: str
    source_id: str
    signer_id: str | None = None
    fingerspelling_text: str | None = None
    fingerspelling_spans: list | None = None
    fingerspelling_label_known: bool = False


class ISLVideoDataset(Dataset):
    def __init__(self, manifest: str | Path, crop_size: int = 112, image_size: int = 224,
                 max_pose_gap_seconds: float = 0.25, require_translation: bool = True):
        self.path = Path(manifest)
        self.root = self.path.parent
        self.crop_size = crop_size
        self.image_size = image_size
        self.max_pose_gap_seconds = max_pose_gap_seconds
        self.require_translation = require_translation
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        self.records: list[ManifestRecord] = []
        for line_no, line in enumerate(self.path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            required = {"uid", "npz", "source_id"}
            if self.require_translation:
                required.add("translation")
            missing = required - row.keys()
            if missing:
                raise ValueError(f"{self.path}:{line_no} missing keys: {sorted(missing)}")
            for field in ("uid", "npz", "source_id"):
                value = row.get(field)
                if not isinstance(value, str) or not value.strip():
                    raise ValueError(f"{self.path}:{line_no} field {field!r} must be a nonempty string")
            translation = row.get("translation", "")
            if not isinstance(translation, str):
                raise ValueError(f"{self.path}:{line_no} field 'translation' must be a string")
            if self.require_translation and not translation.strip():
                raise ValueError(f"{self.path}:{line_no} field 'translation' must be nonempty in supervised mode")
            signer_id = row.get("signer_id")
            if signer_id is not None and (not isinstance(signer_id, str) or not signer_id.strip()):
                raise ValueError(f"{self.path}:{line_no} field 'signer_id' must be null or a nonempty string")
            fingerspelling_text = row.get("fingerspelling_text")
            if fingerspelling_text is not None and not isinstance(fingerspelling_text, str):
                raise ValueError(f"{self.path}:{line_no} field 'fingerspelling_text' must be a string")
            fingerspelling_spans = row.get("fingerspelling_spans")
            if fingerspelling_spans is not None:
                if not isinstance(fingerspelling_spans, list):
                    raise ValueError(f"{self.path}:{line_no} field 'fingerspelling_spans' must be a list of [start,end] pairs")
                for span in fingerspelling_spans:
                    if not isinstance(span, (list, tuple)) or len(span) != 2:
                        raise ValueError(f"{self.path}:{line_no} each fingerspelling_spans entry must be a [start_sec, end_sec] pair")
                    s_t, e_t = float(span[0]), float(span[1])
                    if not (math.isfinite(s_t) and math.isfinite(e_t) and e_t >= s_t):
                        raise ValueError(f"{self.path}:{line_no} fingerspelling_spans intervals must be finite with start <= end")
            fs_known_explicit = row.get("fingerspelling_label_known")
            if fs_known_explicit is not None:
                fs_known = bool(fs_known_explicit)
            else:
                fs_known = (fingerspelling_text is not None) or (fingerspelling_spans is not None)
            self.records.append(ManifestRecord(
                uid=row["uid"], npz=row["npz"], translation=translation,
                source_id=row["source_id"], signer_id=signer_id,
                fingerspelling_text=fingerspelling_text,
                fingerspelling_spans=fingerspelling_spans,
                fingerspelling_label_known=fs_known,
            ))
        if not self.records:
            raise ValueError(f"Manifest contains no records: {self.path}")
        ids = [r.uid for r in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("Manifest uid values must be unique")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        rec = self.records[index]
        npz_path = Path(rec.npz)
        if not npz_path.is_absolute():
            npz_path = self.root / npz_path
        if not npz_path.exists():
            raise FileNotFoundError(npz_path)
        with np.load(npz_path, allow_pickle=False) as z:
            required = {"frames", "video_timestamps", "pose_xy", "pose_timestamps"}
            missing = required - set(z.files)
            if missing:
                raise ValueError(f"{npz_path} missing arrays: {sorted(missing)}")
            frames = np.asarray(z["frames"])
            video_times = np.asarray(z["video_timestamps"], dtype=np.float64)
            pose_xy_raw = np.asarray(z["pose_xy"], dtype=np.float32)
            pose_times = np.asarray(z["pose_timestamps"], dtype=np.float64)
            if "pose_valid" in z:
                pose_valid_raw = np.asarray(z["pose_valid"], dtype=bool)
            else:
                pose_valid_raw = np.isfinite(pose_xy_raw).all(-1)
            confidence_known = "pose_confidence" in z
            if confidence_known:
                pose_conf_raw = np.asarray(z["pose_confidence"], dtype=np.float32)
            else:
                pose_conf_raw = pose_valid_raw.astype(np.float32)
            boxes_raw = np.asarray(z["boxes"], dtype=np.float32) if "boxes" in z else None
            box_valid_raw = np.asarray(z["box_valid"], dtype=bool) if "box_valid" in z else None
            box_scores_raw = np.asarray(z["box_scores"], dtype=np.float32) if "box_scores" in z else None
            npz_fs_spans = np.asarray(z["fingerspelling_spans"], dtype=bool) if "fingerspelling_spans" in z else None
        if frames.ndim != 4:
            raise ValueError(f"frames must be [T,H,W,3] or [T,3,H,W], got {frames.shape}")
        if frames.shape[-1] == 3:
            frames = np.transpose(frames, (0, 3, 1, 2))
        elif frames.shape[1] != 3:
            raise ValueError("frames channel dimension must be 3")
        if frames.shape[-2] <= 0 or frames.shape[-1] <= 0:
            raise ValueError("video frames must have positive spatial dimensions")
        if len(frames) != len(video_times) or len(frames) < 1:
            raise ValueError("video frame count must equal video_timestamps length and be nonzero")
        validate_times(video_times, "video_timestamps")
        validate_times(pose_times, "pose_timestamps")
        if pose_xy_raw.shape != (len(pose_times), 66, 2):
            raise ValueError("pose_xy must follow the 66-point ISL-MEGA order")
        if pose_valid_raw.shape != (len(pose_times), 66):
            raise ValueError("pose_valid must have shape [Tp,66]")
        if pose_conf_raw.shape != (len(pose_times), 66):
            raise ValueError("pose_confidence must have shape [Tp,66]")
        pose_valid_raw &= np.isfinite(pose_xy_raw).all(-1)
        pose_xy_raw = np.nan_to_num(pose_xy_raw, nan=0.0, posinf=0.0, neginf=0.0)
        pose_conf_raw = np.nan_to_num(pose_conf_raw, nan=0.0).clip(0, 1)
        pose_xy, pose_conf, pose_valid = align_pose_to_video(
            video_times, pose_times, pose_xy_raw, pose_conf_raw, pose_valid_raw, self.max_pose_gap_seconds
        )
        if not np.isfinite(pose_xy).all():
            raise ValueError("Aligned pose contains nonfinite coordinates")
        if not np.isfinite(frames).all():
            raise ValueError("frames contain NaN or infinite values")
        is_uint8 = frames.dtype == np.uint8
        is_float = np.issubdtype(frames.dtype, np.floating)
        if not is_uint8 and not is_float:
            raise ValueError(f"Unsupported frame dtype {frames.dtype}; use uint8 or floating-point [0,1]")
        if is_float and frames.size and (float(frames.min()) < 0.0 or float(frames.max()) > 1.0):
            raise ValueError("Floating-point frames must be RGB values in [0,1]; uint8 is accepted for [0,255]")

        # Preserve aspect ratio. Apply the identical affine transform to keypoints
        # and boxes so RGB appearance and pose geometry remain registered.
        was_uint8 = is_uint8
        frame_tensor = torch.from_numpy(np.ascontiguousarray(frames)).float()
        if was_uint8:
            # Scale by storage dtype, not observed max: a dark uint8 frame whose
            # maximum pixel is 1 must represent 1/255, not unit brightness.
            frame_tensor = frame_tensor / 255.0
        frame_tensor = frame_tensor.clamp(0, 1)
        frame_tensor, affine = letterbox_frames(frame_tensor, self.image_size)
        pose_xy = transform_points_to_letterbox(pose_xy, affine)
        pose_in_bounds = ((pose_xy >= 0.0) & (pose_xy <= 1.0)).all(-1)
        pose_valid &= pose_in_bounds
        body_fallback = np.zeros(len(video_times), dtype=bool)
        if boxes_raw is None:
            boxes, box_valid = derive_rois(pose_xy, pose_valid)
            # The whole-frame fallback is useful as a global visual/motion view,
            # but is not a detected body crop and receives a low quality proxy.
            body_fallback = ~pose_valid[:, :48].any(axis=-1)
        else:
            if boxes_raw.shape != (len(video_times), 4, 4):
                raise ValueError("boxes must have shape [T,4,4] in face/left/right/body order")
            box_finite = np.isfinite(boxes_raw).all(-1)
            boxes = transform_boxes_to_letterbox(
                np.nan_to_num(boxes_raw, nan=0.0, posinf=1.0, neginf=0.0), affine
            )
            if box_valid_raw is not None and box_valid_raw.shape != (len(video_times), 4):
                raise ValueError("box_valid must have shape [T,4]")
            positive_area = (boxes[..., 2] > boxes[..., 0]) & (boxes[..., 3] > boxes[..., 1])
            box_valid = (box_valid_raw & box_finite & positive_area) if box_valid_raw is not None else (box_finite & positive_area)
        if box_scores_raw is None:
            box_scores = box_valid.astype(np.float32)
            box_scores[:, 3] = np.where(body_fallback, 0.25, box_scores[:, 3])
        else:
            if box_scores_raw.shape != (len(video_times), 4):
                raise ValueError("box_scores must have shape [T,4]")
            box_scores = np.nan_to_num(box_scores_raw, nan=0.0, posinf=1.0, neginf=0.0).clip(0, 1)
        if not np.isfinite(pose_xy).all():
            raise ValueError("Pose coordinates contain NaN or infinite values after letterboxing")
        t = len(video_times)
        known = np.full((t, 66), confidence_known, dtype=bool)
        rel_times = (video_times - video_times[0]).astype(np.float32)
        if npz_fs_spans is not None:
            if npz_fs_spans.shape != (t,):
                raise ValueError("fingerspelling_spans NPZ array must have shape [Tv]")
            fs_spans = npz_fs_spans
            fs_span_known = True
        elif rec.fingerspelling_spans is not None:
            fs_spans = np.zeros(t, dtype=bool)
            for item in rec.fingerspelling_spans:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    raise ValueError("Each fingerspelling_spans entry must be a [start_sec, end_sec] pair")
                s_t, e_t = float(item[0]), float(item[1])
                if not (math.isfinite(s_t) and math.isfinite(e_t) and e_t >= s_t):
                    raise ValueError("Each fingerspelling_spans interval must be finite with start <= end")
                fs_spans |= (rel_times >= s_t) & (rel_times <= e_t)
            fs_span_known = True
        else:
            fs_spans = np.zeros(t, dtype=bool)
            fs_span_known = False
        return {
            "uid": rec.uid, "source_id": rec.source_id, "signer_id": rec.signer_id,
            "frames": frame_tensor,
            "timestamps": torch.from_numpy(rel_times),
            "pose_xy": torch.from_numpy(pose_xy),
            "pose_confidence": torch.from_numpy(pose_conf),
            "pose_valid": torch.from_numpy(pose_valid),
            "confidence_known": torch.from_numpy(known),
            "boxes": torch.from_numpy(boxes),
            "box_valid": torch.from_numpy(box_valid),
            "box_scores": torch.from_numpy(box_scores),
            "translation": rec.translation,
            "fingerspelling_text": rec.fingerspelling_text or "",
            "fingerspelling_spans": torch.from_numpy(fs_spans),
            "fingerspelling_span_known": fs_span_known,
            "fingerspelling_label_known": rec.fingerspelling_label_known,
        }


def collate_examples(items: list[dict]) -> dict:
    if not items:
        raise ValueError("Cannot collate an empty batch")
    lengths = torch.tensor([x["frames"].shape[0] for x in items], dtype=torch.long)
    max_t = int(lengths.max())
    b = len(items)
    _, c, h, w = items[0]["frames"].shape
    def pad(key, shape, dtype, fill=0):
        out = torch.full((b, max_t, *shape), fill, dtype=dtype)
        for i, item in enumerate(items):
            n = item[key].shape[0]
            out[i, :n] = item[key]
        return out
    frames = pad("frames", (c, h, w), torch.float32)
    timestamps = pad("timestamps", (), torch.float32)
    pose_xy = pad("pose_xy", (66, 2), torch.float32)
    pose_conf = pad("pose_confidence", (66,), torch.float32)
    pose_valid = pad("pose_valid", (66,), torch.bool)
    confidence_known = pad("confidence_known", (66,), torch.bool)
    boxes = pad("boxes", (4, 4), torch.float32)
    box_valid = pad("box_valid", (4,), torch.bool)
    box_scores = pad("box_scores", (4,), torch.float32)
    if any("fingerspelling_spans" in x for x in items):
        fs_spans = torch.zeros((b, max_t), dtype=torch.bool)
        for i, item in enumerate(items):
            if "fingerspelling_spans" in item:
                span = item["fingerspelling_spans"]
                fs_spans[i, :span.shape[0]] = span
        fs_span_known = torch.tensor(
            [bool(x.get("fingerspelling_span_known", False)) for x in items], dtype=torch.bool
        )
    else:
        fs_spans = torch.zeros((b, max_t), dtype=torch.bool)
        fs_span_known = torch.zeros(b, dtype=torch.bool)
    frame_mask = torch.arange(max_t)[None] < lengths[:, None]
    return {
        "uids": [x["uid"] for x in items], "source_ids": [x["source_id"] for x in items],
        "signer_ids": [x["signer_id"] for x in items], "frames": frames,
        "timestamps": timestamps, "pose_xy": pose_xy, "pose_confidence": pose_conf,
        "pose_valid": pose_valid, "confidence_known": confidence_known,
        "boxes": boxes, "box_valid": box_valid, "box_scores": box_scores,
        "lengths": lengths, "frame_mask": frame_mask,
        "translations": [x["translation"] for x in items],
        "fingerspelling_texts": [x["fingerspelling_text"] for x in items],
        "fingerspelling_spans": fs_spans,
        "fingerspelling_span_known": fs_span_known,
        "fingerspelling_label_known": torch.tensor(
            [bool(x.get("fingerspelling_label_known", False)) for x in items], dtype=torch.bool
        ),
    }


def write_grouped_splits(records: list[dict], output_dir: str | Path,
                         ratios: tuple[float, float, float] = (0.8, 0.1, 0.1), seed: int = 2026,
                         group_key: str = "source_id") -> dict[str, list[dict]]:
    """Deterministic group-disjoint split builder. Group-level counts are approximate.

    When group_key is 'joint_source_signer', connected components over shared source_id
    and signer_id edges are partitioned together, preventing leakage of either key.
    """
    if len(ratios) != 3 or any(x <= 0 for x in ratios) or not math.isclose(sum(ratios), 1.0, abs_tol=1e-6):
        raise ValueError("ratios must be three positive values summing to 1")
    groups: dict[str, list[dict]] = {}
    if group_key == "joint_source_signer":
        parent: dict[str, str] = {}
        def find(u: str) -> str:
            path = []
            while parent.get(u, u) != u:
                path.append(u)
                u = parent[u]
            for p in path:
                parent[p] = u
            return u
        def union(u: str, v: str) -> None:
            root_u, root_v = find(u), find(v)
            if root_u != root_v:
                parent[root_u] = root_v
        for row in records:
            src = row.get("source_id")
            if not src:
                raise ValueError("Every record needs nonempty source_id for joint_source_signer split")
            src_node = f"src_{src}"
            sig = row.get("signer_id")
            if sig:
                union(src_node, f"sig_{sig}")
            else:
                find(src_node)
        for row in records:
            cluster = find(f"src_{row['source_id']}")
            groups.setdefault(cluster, []).append(row)
    else:
        for row in records:
            group = row.get(group_key)
            if not group:
                raise ValueError(f"Every record needs nonempty {group_key}")
            groups.setdefault(str(group), []).append(row)
    if len(groups) < 3:
        raise ValueError("At least three distinct groups are required for train/valid/test")
    rng = np.random.default_rng(seed)
    keys = np.asarray(list(groups), dtype=object)
    rng.shuffle(keys)
    # Greedy assignment by remaining record counts, preserving group integrity.
    totals = [sum(len(v) for v in groups.values()) * r for r in ratios]
    out = {"train": [], "valid": [], "test": []}
    counts = [0, 0, 0]
    names = list(out)
    for key in sorted(keys, key=lambda k: len(groups[str(k)]), reverse=True):
        idx = min(range(3), key=lambda i: counts[i] / max(totals[i], 1))
        out[names[idx]].extend(groups[str(key)])
        counts[idx] += len(groups[str(key)])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for split, rows in out.items():
        (output / f"{split}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows),
                                                encoding="utf-8")
    sizes = [len(rows) for rows in groups.values()]
    total_samples = len(records)
    largest_size = max(sizes)
    largest_pct = (largest_size / max(total_samples, 1)) * 100.0
    sorted_sizes = sorted(sizes)
    median_size = float(np.median(sorted_sizes))
    p90_size = float(np.percentile(sorted_sizes, 90))
    diagnostics = {
        "group_key": group_key,
        "total_records": total_samples,
        "total_components": len(groups),
        "largest_component_size": largest_size,
        "largest_component_pct": round(largest_pct, 2),
        "median_component_size": median_size,
        "p90_component_size": p90_size,
        "splits": {
            k: {
                "count": len(v),
                "pct": round((len(v) / max(total_samples, 1)) * 100.0, 2),
                "unique_sources": len({r["source_id"] for r in v if r.get("source_id")}),
                "unique_signers": len({r["signer_id"] for r in v if r.get("signer_id")}),
            }
            for k, v in out.items()
        },
    }
    (output / "split_diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    if largest_pct > 50.0:
        import warnings
        warnings.warn(
            f"Largest component contains {largest_pct:.1f}% of records; joint splitting may skew partition balance. "
            "Consider reporting two separate evaluation protocols: (1) official source-disjoint and (2) strict signer-independent.",
            RuntimeWarning,
        )
    if group_key == "joint_source_signer":
        validate_split_disjointness(out, "source_id")
        signer_present = any(any(r.get("signer_id") for r in split_rows) for split_rows in out.values())
        if signer_present:
            validate_split_disjointness(out, "signer_id")
    else:
        validate_split_disjointness(out, group_key)
    return out


def validate_split_disjointness(splits: dict[str, list[dict]], group_key: str = "source_id",
                                require_signer_disjoint: bool = False) -> None:
    """Reject group leakage and, when requested, require complete signer metadata."""
    names = list(splits)
    if len(names) < 2:
        raise ValueError("At least two splits are required for leakage validation")
    seen_uids: dict[str, str] = {}
    for split_name, rows in splits.items():
        if not rows:
            raise ValueError(f"Split {split_name!r} is empty")
        for index, row in enumerate(rows):
            uid = row.get("uid")
            if not isinstance(uid, str) or not uid.strip():
                raise ValueError(f"Split {split_name!r} has missing/empty uid at row {index}")
            if uid in seen_uids:
                raise ValueError(f"Duplicate uid {uid!r} appears in splits {seen_uids[uid]!r} and {split_name!r}")
            seen_uids[uid] = split_name
        missing = [i for i, row in enumerate(rows) if not row.get(group_key)]
        if missing:
            raise ValueError(f"Split {split_name!r} has missing/empty {group_key} at rows {missing[:5]}")
        if require_signer_disjoint:
            missing_signer = [i for i, row in enumerate(rows) if not row.get("signer_id")]
            if missing_signer:
                raise ValueError(f"Signer-disjoint evaluation requested but {split_name!r} has missing signer_id at rows {missing_signer[:5]}")
    for i, a in enumerate(names):
        groups_a = {str(r[group_key]) for r in splits[a]}
        for b in names[i + 1:]:
            groups_b = {str(r[group_key]) for r in splits[b]}
            overlap = groups_a & groups_b
            if overlap:
                raise ValueError(f"{group_key} leakage between {a} and {b}: {sorted(overlap)[:5]}")
    if require_signer_disjoint:
        signer_sets = {s: {str(r["signer_id"]) for r in rows} for s, rows in splits.items()}
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                overlap = signer_sets[a] & signer_sets[b]
                if overlap:
                    raise ValueError(f"signer_id leakage between {a} and {b}: {sorted(overlap)[:5]}")
