"""Small installed CLI for configuration and split validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import ISLVideoDataset, validate_split_disjointness
from .model import build_model


def main() -> None:
    parser = argparse.ArgumentParser(prog="isl-mega")
    sub = parser.add_subparsers(dest="command", required=True)

    config_parser = sub.add_parser("validate-config", help="Validate and print a YAML config")
    config_parser.add_argument("--config", required=True)

    split_parser = sub.add_parser("check-splits", help="Check source/signer disjointness from manifests")
    split_parser.add_argument("--config", required=True)

    model_parser = sub.add_parser("model-info", help="Build the visual model and report parameter counts")
    model_parser.add_argument("--config", required=True)

    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "validate-config":
        print(json.dumps(cfg.to_dict(), indent=2))
        return

    if args.command == "check-splits":
        common = (cfg.data.crop_size, cfg.data.image_size, cfg.pose.max_interpolation_gap_seconds)
        train = ISLVideoDataset(cfg.data.train_manifest, *common)
        valid = ISLVideoDataset(cfg.data.valid_manifest, *common)
        test = ISLVideoDataset(cfg.data.test_manifest, *common)
        split_records = {"train": [vars(r) for r in train.records],
                         "valid": [vars(r) for r in valid.records],
                         "test": [vars(r) for r in test.records]}
        if cfg.data.strict_source_disjoint:
            validate_split_disjointness(split_records, "source_id")
        if cfg.data.require_signer_disjoint:
            validate_split_disjointness(split_records, "signer_id", require_signer_disjoint=True)
        print(json.dumps({"status": "passed", "train": len(train),
                          "valid": len(valid), "test": len(test)}, indent=2))
        return

    model = build_model(cfg, with_decoder=False)
    groups = {}
    for name in ("appearance", "pose", "motion", "fingerspelling", "fusion", "temporal", "planner"):
        module = getattr(model, name)
        groups[name] = sum(p.numel() for p in module.parameters()) if module is not None else 0
    print(json.dumps({"visual_model_parameters_excluding_text_decoder": sum(groups.values()),
                      "groups": groups}, indent=2))


if __name__ == "__main__":
    main()
