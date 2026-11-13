#!/usr/bin/env python3
"""
Convert MediaPipe Holistic (543 landmark) files to Canonical 66-landmark format.

Usage:
    python scripts/convert_pose_543_to_66.py --input-dir /path/to/543_poses --output-dir /path/to/66_poses
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from articulatory_transformer.adapter_543_to_66 import (
    map_543_to_66,
    validate_canonical_66_structure,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert 543 MediaPipe poses to canonical 66 format.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing raw 543 pose files (.npy/.npz/.json)")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save canonical 66 pose files")
    parser.add_argument("--max-files", type=int, default=None, help="Optional max files to process")
    return parser.parse_args()


def process_file(in_path: Path, out_path: Path) -> dict:
    stats = {"file": in_path.name, "status": "ok", "error": None}
    try:
        if in_path.suffix == ".npy":
            import numpy as np
            arr = np.load(in_path)
            converted = map_543_to_66(arr)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(out_path, converted)
        elif in_path.suffix == ".json":
            with open(in_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            converted = map_543_to_66(data)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(converted, f)
        else:
            stats["status"] = "skipped"
            stats["error"] = f"Unsupported suffix: {in_path.suffix}"
            return stats

        val_report = validate_canonical_66_structure(converted)
        if not val_report["valid"]:
            stats["status"] = "invalid"
            stats["error"] = "; ".join(val_report["errors"])

    except Exception as e:
        stats["status"] = "failed"
        stats["error"] = str(e)

    return stats


def main():
    args = parse_args()
    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)

    if not in_dir.exists():
        print(f"Error: input directory does not exist: {in_dir}")
        sys.exit(1)

    files = sorted(list(in_dir.glob("*.npy")) + list(in_dir.glob("*.json")))
    if args.max_files:
        files = files[:args.max_files]

    print(f"Found {len(files)} files to convert from {in_dir} to {out_dir}")
    
    results = {"total": len(files), "ok": 0, "invalid": 0, "failed": 0, "skipped": 0}
    for i, f in enumerate(files):
        out_f = out_dir / f.name
        res = process_file(f, out_f)
        status = res["status"]
        results[status] = results.get(status, 0) + 1
        if (i + 1) % 50 == 0 or (i + 1) == len(files):
            print(f"Processed {i + 1}/{len(files)} files... (ok: {results['ok']}, failed: {results['failed']})")

    print("\nConversion Summary:")
    for k, v in results.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

