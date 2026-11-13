"""Strict, variable-length JSONL/NPZ data loading and padded collation."""
import json
from functools import partial
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import ModelConfig, DataConfig
from .tokenizer import ByteTokenizer
from .geometry import BodyFrame


def read_manifest(path: str | Path) -> list[dict]:
    path = Path(path).resolve()
    records, ids = [], set()
    with path.open(encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or not {"id", "pose", "text"} <= row.keys():
                raise ValueError(f"{path}:{line_number}: required keys: id, pose, text")
            if not isinstance(row["id"], str) or not row["id"] or row["id"] in ids:
                raise ValueError(f"{path}:{line_number}: empty/duplicate/non-string id")
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError(f"{path}:{line_number}: text must be a nonempty string")
            if not isinstance(row["pose"], str):
                raise ValueError("pose must be a filesystem path string")
            ids.add(row["id"])
            row["pose"] = str((path.parent / row["pose"]).resolve())
            if "rgb" in row:
                row["rgb"] = str((path.parent / row["rgb"]).resolve())
            records.append(row)
    if not records:
        raise ValueError(f"empty manifest: {path}")
    return records


def load_pose(path: str | Path, model: ModelConfig, max_frames: int) -> dict:
    with np.load(path, allow_pickle=False) as f:
        if not {"pose", "confidence", "timestamps"} <= set(f.files):
            raise ValueError(f"{path}: NPZ requires pose, confidence, timestamps")
        p = np.asarray(f["pose"], dtype=np.float32).copy()
        c = np.asarray(f["confidence"], dtype=np.float32).copy()
        ts = np.asarray(f["timestamps"], dtype=np.float64).copy()
    if p.ndim != 3 or p.shape[1:] != (model.joints, 3) or not 2 <= len(p) <= max_frames:
        raise ValueError(f"{path}: expected 2..{max_frames} frames with shape [T,{model.joints},3]")
    if c.shape != p.shape[:-1] or ts.shape != (len(p),):
        raise ValueError(f"{path}: confidence or timestamp shape mismatch")
    if not np.isfinite(c).all() or (c < 0).any() or (c > 1).any():
        raise ValueError(f"{path}: confidence must be finite in [0,1]")
    if not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError(f"{path}: timestamps must be finite and strictly increasing, in seconds")
    if model.depth_policy == "ignore":
        p[..., 2] = 0
    if not np.isfinite(p[c > 0]).all():
        raise ValueError(f"{path}: observed coordinates must be finite")
    # Missing joints are interpolated in physical time, with nearest endpoint
    # extrapolation. Their confidence stays zero. Entirely absent joints sit at
    # the initial shoulder midpoint and are explicitly marked absent.
    reference = BodyFrame(model.anchors, model.anchor_min_confidence, model.anchor_policy)
    eligible = reference.eligible_frames(torch.from_numpy(p)[None], torch.from_numpy(c)[None])
    if eligible.any():
        index = int(eligible.long().argmax(1)[0])
        origin = p[index, model.anchors[:2]].mean(0)
    elif model.allow_unanchored_rgb:
        origin = np.zeros(3, dtype=np.float32)
    else:
        raise ValueError("no reliable nondegenerate body frame")
    for j in range(model.joints):
        observed = c[:, j] > 0
        if not observed.any():
            p[:, j] = origin
        elif not observed.all():
            for axis in range(3):
                p[:, j, axis] = np.interp(ts, ts[observed], p[observed, j, axis])
    time_origin = float(ts[0])
    ts = (ts - ts[0]).astype(np.float32)
    if (np.diff(ts) <= 0).any():
        raise ValueError(f"{path}: timestamp spacing is below float32 precision")
    return {"pose": torch.from_numpy(p), "confidence": torch.from_numpy(c),
            "timestamps": torch.from_numpy(ts), "time_origin": time_origin}


def load_rgb(path, sample, model):
    with np.load(path, allow_pickle=False) as f:
        rgb = np.asarray(f["rgb"], dtype=np.float32).copy()
        mask = np.asarray(f["rgb_mask"])
        ts = np.asarray(f["timestamps"], dtype=np.float64)
    n = len(sample["pose"])
    if rgb.ndim != 5 or rgb.shape[:3] != (n, model.visual_regions, 3) or min(rgb.shape[-2:]) < 8:
        raise ValueError("RGB archive must contain [T,R,3,H,W] crops")
    if mask.shape != rgb.shape[:2] or mask.dtype != np.bool_:
        raise ValueError("RGB observation mask must be boolean [T,R]")
    if ts.shape != (n,) or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("invalid RGB timestamps")
    if (abs(float(ts[0]) - sample["time_origin"]) > 1e-4 or
            not np.allclose(ts - ts[0], sample["timestamps"].numpy(), atol=1e-4, rtol=1e-5)):
        raise ValueError("RGB and pose timestamps are not aligned")
    if not np.isfinite(rgb[mask]).all() or ((rgb[mask] < 0) | (rgb[mask] > 1)).any():
        raise ValueError("observed crops must be finite RGB in [0,1]")
    rgb[~mask] = 0
    return {"rgb": torch.from_numpy(rgb), "rgb_mask": torch.from_numpy(mask.copy())}


class SignDataset(Dataset):
    def __init__(self, manifest, model: ModelConfig, data: DataConfig, codec=None):
        self.records = read_manifest(manifest)
        self.model, self.data = model, data
        if model.text_backend == "hf" and codec is None:
            raise ValueError("HF datasets require the matching model codec; do not use byte targets")
        self.tokenizer = codec or ByteTokenizer()
        for row in self.records:
            if model.text_backend == "byte" and len(row["text"].encode("utf-8")) > data.max_bytes:
                raise ValueError(f"{row['id']}: target exceeds max_bytes; truncation is forbidden")
            if len(self.tokenizer.encode(row["text"])) - 1 > model.max_target_length:
                raise ValueError(f"{row['id']}: target exceeds decoder token limit")
            if model.visual_backend != "none" and "rgb" not in row:
                raise ValueError(f"{row['id']}: visual mode requires an rgb archive path")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        sample = load_pose(row["pose"], self.model, self.data.max_frames)
        if self.model.visual_backend != "none":
            sample.update(load_rgb(row["rgb"], sample, self.model))
        sample.update(id=row["id"], text=row["text"],
                      tokens=torch.tensor(self.tokenizer.encode(row["text"]), dtype=torch.long))
        return sample


def collate_signs(samples: list[dict], pad_id=0) -> dict:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    b = len(samples)
    lengths = torch.tensor([len(s["pose"]) for s in samples], dtype=torch.long)
    t, j = int(lengths.max()), samples[0]["pose"].shape[1]
    target_length = max(len(s["tokens"]) for s in samples)
    pose = torch.zeros(b, t, j, 3)
    confidence = torch.zeros(b, t, j)
    timestamps = torch.zeros(b, t)
    tokens = torch.full((b, target_length), pad_id, dtype=torch.long)
    extras = {}
    if "rgb" in samples[0]:
        shape = samples[0]["rgb"].shape[1:]
        extras = {"rgb": torch.zeros(b, t, *shape), "rgb_mask": torch.zeros(b, t, shape[0], dtype=torch.bool)}
    for i, sample in enumerate(samples):
        n, m = len(sample["pose"]), len(sample["tokens"])
        pose[i, :n] = sample["pose"]
        confidence[i, :n] = sample["confidence"]
        timestamps[i, :n] = sample["timestamps"]
        tokens[i, :m] = sample["tokens"]
        if extras:
            if sample["rgb"].shape[1:] != shape:
                raise ValueError("all crops in a batch must have the same region count and image size")
            extras["rgb"][i, :n] = sample["rgb"]
            extras["rgb_mask"][i, :n] = sample["rgb_mask"]
    return {"pose": pose, "confidence": confidence, "timestamps": timestamps, "lengths": lengths, **extras,
            "decoder_input": tokens[:, :-1], "labels": tokens[:, 1:],
            "ids": [s["id"] for s in samples], "texts": [s["text"] for s in samples]}


def make_collator(codec):
    return partial(collate_signs, pad_id=codec.pad_id)


def model_inputs(batch, decoder=True):
    names = ["pose", "confidence", "lengths", "timestamps", "rgb", "rgb_mask"]
    if decoder:
        names.append("decoder_input")
    return {name: batch[name] for name in names if name in batch}


def move_batch(batch: dict, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def assert_disjoint(datasets: dict[str, SignDataset]):
    seen_id, seen_path = {}, {}
    seen_rgb = {}
    seen_group = {}
    for split, dataset in datasets.items():
        for row in dataset.records:
            for value, seen, label in ((row["id"], seen_id, "id"), (row["pose"], seen_path, "path")):
                if value in seen:
                    raise ValueError(f"duplicate {label} across/within splits {seen[value]} and {split}: {value}")
                seen[value] = split
            if "rgb" in row:
                if row["rgb"] in seen_rgb:
                    raise ValueError(f"duplicate RGB path across/within splits: {row['rgb']}")
                seen_rgb[row["rgb"]] = split
            for field in ("source_video", "split_group"):
                value = row.get(field)
                if value is not None:
                    if not isinstance(value, str) or not value:
                        raise ValueError(f"{field} must be a nonempty string when provided")
                    key = (field, value)
                    if key in seen_group and seen_group[key] != split:
                        raise ValueError(f"source/group leakage across {seen_group[key]} and {split}: {key}")
                    seen_group[key] = split
