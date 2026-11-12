#!/usr/bin/env python3
"""Validate every NPZ sample in an ISL-MEGA JSONL manifest and summarize coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, REGION_ORDER
from isl_mega.fingerspelling import encode_characters


def distribution(values: list[float]) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(x)), "mean": float(np.mean(x)),
        "median": float(np.median(x)), "p90": float(np.percentile(x, 90)),
        "p95": float(np.percentile(x, 95)), "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--config", default=None, help="Optional YAML config providing image/crop/pose gap sizes")
    parser.add_argument("--unlabeled", action="store_true",
                        help="Allow missing translation fields for self-supervised pretraining")
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--max-pose-gap", type=float, default=0.20)
    parser.add_argument("--out", default=None, help="Optional JSON report path")
    args = parser.parse_args()

    image_size = args.image_size
    crop_size = args.crop_size
    max_pose_gap = args.max_pose_gap
    if args.config:
        cfg = load_config(args.config)
        image_size = cfg.data.image_size
        crop_size = cfg.data.crop_size
        max_pose_gap = cfg.pose.max_interpolation_gap_seconds

    ds = ISLVideoDataset(args.manifest, crop_size, image_size,
                         max_pose_gap, require_translation=not args.unlabeled)
    frame_counts: list[float] = []
    durations: list[float] = []
    point_coverage: list[float] = []
    region_coverage = {name: [] for name in REGION_ORDER}
    source_ids, signer_ids = set(), set()
    translations_empty = 0
    fs_text_count = 0
    fs_span_count = 0
    for i in range(len(ds)):
        sample = ds[i]  # Forces NPZ load, timestamp alignment, ROI transforms and tensor validation.
        t = int(sample["frames"].shape[0])
        frame_counts.append(float(t))
        durations.append(float(sample["timestamps"][-1].item()) if t > 1 else 0.0)
        point_coverage.append(float(sample["pose_valid"].float().mean().item()))
        for region, (start, stop) in zip(REGION_ORDER, ((48, 66), (0, 21), (21, 42), (42, 48))):
            region_coverage[region].append(float(sample["pose_valid"][:, start:stop].float().mean().item()))
        source_ids.add(sample["source_id"])
        if sample["signer_id"]:
            signer_ids.add(str(sample["signer_id"]))
        translations_empty += int(not sample["translation"].strip())
        if sample["fingerspelling_text"]:
            encode_characters(sample["fingerspelling_text"])
            fs_text_count += 1
        if sample["fingerspelling_span_known"]:
            fs_span_count += 1

    report = {
        "manifest": str(Path(args.manifest).resolve()),
        "records_validated": len(ds),
        "unlabeled_mode": bool(args.unlabeled),
        "unique_source_ids": len(source_ids),
        "unique_nonempty_signer_ids": len(signer_ids),
        "empty_translation_records": translations_empty,
        "fingerspelling_text_records": fs_text_count,
        "fingerspelling_span_records": fs_span_count,
        "frame_count": distribution(frame_counts),
        "duration_seconds": distribution(durations),
        "mean_pose_point_coverage": float(np.mean(point_coverage)),
        "mean_region_pose_coverage": {k: float(np.mean(v)) for k, v in region_coverage.items()},
        "region_order": list(REGION_ORDER),
        "note": "This validates file/schema/transform integrity, not annotation correctness or translation quality.",
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
