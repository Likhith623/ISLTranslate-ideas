"""Pixel-dependent diagnostic data, not a linguistic or pretrained-model benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
from .synthetic import make_synthetic


def make_visual_synthetic(output="data/visual_demo", seed=17, samples=24):
    root = make_synthetic(output, seed, samples)
    rng = np.random.default_rng(seed + 100)
    for split in ("train", "val", "test"):
        path = root / f"{split}.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for i, row in enumerate(rows):
            with np.load(root / row["pose"], allow_pickle=False) as f:
                payload = {key: f[key].copy() for key in f.files}
            # Remove the old event-order cue: pose is static; targets depend on
            # the observed crop pattern only. Random identities are not labels.
            payload["pose"][:] = payload["pose"][0]
            np.savez_compressed(root / row["pose"], **payload)
            t = len(payload["pose"])
            rgb = rng.uniform(.05, .1, size=(t, 4, 3, 24, 24)).astype(np.float32)
            if i % 2:
                rgb[:, 2, :, :, 6:18] = .95
            else:
                rgb[:, 2, :, 6:18, :] = .95
            name = row["id"] + "_rgb.npz"
            np.savez_compressed(root / name, rgb=rgb, rgb_mask=np.ones((t, 4), dtype=bool),
                                timestamps=payload["timestamps"])
            row.update(text="a" if i % 2 == 0 else "b", rgb=name,
                       diagnostic="pixel-pattern; not real signs or mouthings")
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return Path(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/visual_demo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    print(make_visual_synthetic(args.output, args.seed, args.samples))


if __name__ == "__main__":
    main()
