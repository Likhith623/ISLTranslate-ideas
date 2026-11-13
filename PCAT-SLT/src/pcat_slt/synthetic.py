"""Generate an explicitly nonlinguistic two-event order task for smoke runs."""
import argparse
import json
from pathlib import Path
import numpy as np


def make_synthetic(output="data/demo", seed=17, samples=24):
    if samples < 4:
        raise ValueError("at least four training examples required")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("demo output must be empty to avoid overwriting data")
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    for split, count in (("train", samples), ("val", max(4, samples // 4)), ("test", max(4, samples // 4))):
        rows = []
        for i in range(count):
            label = i % 2
            frames = int(rng.integers(9, 23))
            # Both classes have exactly the same per-stream endpoints; their
            # hand-vs-face event order differs. These are NOT real signs.
            u = np.linspace(0, 1, frames, dtype=np.float32)
            boundary = float(rng.uniform(0.35, 0.65))
            first = np.clip(u / boundary, 0, 1)
            second = np.clip((u - boundary) / (1 - boundary), 0, 1)
            hand, face = (first, second) if label == 0 else (second, first)
            base = rng.normal(0, 0.05, size=(66, 3)).astype(np.float32)
            base[0], base[1], base[2] = [-0.5, 0, 0], [0.5, 0, 0], [0, 0.7, 0]
            base[3:24] += [-0.3, 0.25, 0.1]
            base[24:45] += [0.3, 0.25, 0.1]
            base[45:63] += [0, 0.8, 0]
            pose = np.broadcast_to(base, (frames, 66, 3)).copy()
            pose[:, 3:24, 0] += hand[:, None] * 0.5
            pose[:, 45:63, 1] += face[:, None] * 0.15
            # Proper rotation, positive scale, constant translation.
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 0] *= np.linalg.det(rotation)
            pose = (rng.uniform(0.7, 1.6) * pose @ rotation.T + rng.normal(size=(1, 1, 3))).astype(np.float32)
            name = f"{split}_{i:05d}.npz"
            timestamps = (u * rng.uniform(0.5, 2.5)).astype(np.float32)
            np.savez_compressed(output / name, pose=pose, confidence=np.ones((frames, 66), np.float32),
                                timestamps=timestamps)
            rows.append({"id": f"{split}_{i:05d}", "pose": name, "text": "hf" if label == 0 else "fh",
                         "signer": f"synthetic-{split}-{i}", "synthetic": True})
        with (output / f"{split}.jsonl").open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="data/demo")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--samples", type=int, default=24)
    args = parser.parse_args()
    print(make_synthetic(args.output, args.seed, args.samples))


if __name__ == "__main__":
    main()
