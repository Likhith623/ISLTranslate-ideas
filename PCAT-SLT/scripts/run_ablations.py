"""Write fully resolved multi-seed ablation configs; optionally train each one."""
import argparse
import copy
from pathlib import Path
import subprocess
import sys
import yaml
from pcat_slt.config import load_config, from_dict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="runs/ablations")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 41])
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    base = load_config(args.config).to_dict()
    output = Path(args.output)
    variants = {"pcat": {}, "first_order": {"operator": "first_order"},
                "signature": {"operator": "signature"}, "no_transport": {"operator": "no_transport"},
                "temporal": {"operator": "temporal"}, "euclidean": {"clock": "euclidean"},
                "toggle_timing": {"timing": not base["model"]["timing"]},
                "single_scale": {"progress_scales": []},
                "no_progress_coordinate": {"progress_augmentation": False},
                "coarse_only": {"detail_memory": False, "body_frame": "fixed"},
                "givens": {"transport": "givens"}, "cayley": {"transport": "cayley"},
                "dense_areas": {"area_sketch_dim": 0, "area_sketch_heads": 1},
                "fixed_cells": {"adaptive_step": 0.0},
                "joint_clock_only": {"local_clocks": False},
                "fixed_body": {"body_frame": "fixed"},
                "no_pixels": {"visual_backend": "none", "allow_unanchored_rgb": False},
                "byte_decoder": {"text_backend": "byte", "text_config": {}}}
    for name, changes in variants.items():
        for seed in args.seeds:
            cfg = copy.deepcopy(base)
            cfg["model"].update(changes)
            cfg["train"].update(seed=seed, output_dir=str(output / name / str(seed)))
            from_dict(cfg)
            path = output / "configs" / f"{name}_{seed}.yaml"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
            command = [sys.executable, "-m", "pcat_slt.train", "--config", str(path)]
            print(" ".join(command), flush=True)
            if args.execute:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
