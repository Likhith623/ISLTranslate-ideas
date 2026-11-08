#!/usr/bin/env python3
"""Build deterministic source-disjoint train/validation/test manifests."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from isl_mega.data import write_grouped_splits


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", help="JSONL manifest containing source_id")
    p.add_argument("--out", default="data/manifests")
    p.add_argument("--group-key", default="source_id", choices=["source_id", "signer_id"])
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--ratios", nargs=3, type=float, default=(0.8, 0.1, 0.1))
    args = p.parse_args()
    source_path = Path(args.manifest).resolve()
    rows = [json.loads(x) for x in source_path.read_text().splitlines() if x.strip()]
    for row in rows:
        npz = Path(row["npz"])
        row["npz"] = str((source_path.parent / npz).resolve()) if not npz.is_absolute() else str(npz)
    splits = write_grouped_splits(rows, args.out, tuple(args.ratios), args.seed, args.group_key)
    print({k: len(v) for k, v in splits.items()})

if __name__ == "__main__":
    main()
