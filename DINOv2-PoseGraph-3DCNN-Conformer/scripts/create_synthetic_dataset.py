#!/usr/bin/env python3
"""Create a tiny random NPZ corpus for software smoke tests only (not research data)."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="synthetic_data")
    p.add_argument("--examples", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    args = p.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    rows = []
    for i in range(args.examples):
        t = int(rng.integers(8, 15))
        ts = np.arange(t, dtype=np.float32) / 25.0
        xy = rng.uniform(0.15, 0.85, size=(t, 66, 2)).astype(np.float32)
        conf = rng.uniform(0.6, 1.0, size=(t, 66)).astype(np.float32)
        valid = rng.random((t, 66)) > 0.03
        frames = rng.integers(0, 256, size=(t, 96, 96, 3), dtype=np.uint8)
        npz_name = f"sample_{i:04d}.npz"
        np.savez_compressed(out / npz_name, frames=frames, video_timestamps=ts,
                            pose_xy=xy, pose_timestamps=ts, pose_confidence=conf,
                            pose_valid=valid)
        rows.append({"uid": f"synthetic-{i:04d}", "npz": npz_name,
                     "translation": "this is a synthetic smoke test sentence",
                     "source_id": f"source-{i:04d}", "signer_id": f"signer-{i:04d}"})
    (out / "all.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    print(f"Created {len(rows)} synthetic records in {out}. DO NOT use them as research data.")

if __name__ == "__main__":
    main()
