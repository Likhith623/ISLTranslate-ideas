"""Crop actual image observations using explicit normalized 2D boxes and timestamps."""

import argparse
import re
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps


def prepare(frames, boxes_path, output, size=112):
    if size < 8:
        raise ValueError("crop size must be >=8")
    if Path(output).exists():
        raise FileExistsError(output)
    paths = sorted(
        (p for p in Path(frames).iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}),
        key=lambda p: [int(s) if s.isdigit() else s for s in re.split(r"(\d+)", p.name)],
    )
    with np.load(boxes_path, allow_pickle=False) as f:
        boxes = f["boxes"]
        mask = f["rgb_mask"]
        ts = f["timestamps"]
        if "frame_names" in f.files:
            names = [str(s) for s in f["frame_names"]]
            if any(Path(name).name != name for name in names) or len(set(names)) != len(names):
                raise ValueError("frame_names must be unique filenames")
            paths = [Path(frames) / name for name in names]
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or len(paths) != len(boxes):
        raise ValueError("need one [R,4] normalized xyxy box set per sorted frame")
    if mask.shape != boxes.shape[:2] or mask.dtype != np.bool_ or ts.shape != (len(paths),):
        raise ValueError("bad crop observation mask or timestamp shape")
    if not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError("timestamps must be finite and strictly increase")
    rgb = np.zeros((*mask.shape, 3, size, size), np.float32)
    for i, path in enumerate(paths):
        with Image.open(path) as image:
            image = image.convert("RGB")
            for region in np.flatnonzero(mask[i]):
                box = boxes[i, region]
                if (
                    not np.isfinite(box).all()
                    or (box < 0).any()
                    or (box > 1).any()
                    or box[2] <= box[0]
                    or box[3] <= box[1]
                ):
                    raise ValueError(f"invalid observed box at frame {i}, region {region}")
                pixels = box * np.array([image.width, image.height, image.width, image.height])
                crop = ImageOps.pad(image.crop(tuple(pixels)), (size, size), method=Image.Resampling.BICUBIC)
                rgb[i, region] = np.asarray(crop, dtype=np.float32).transpose(2, 0, 1) / 255
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, rgb=rgb, rgb_mask=mask, timestamps=ts, frame_names=np.asarray([p.name for p in paths])
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--boxes", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=112)
    args = parser.parse_args()
    prepare(args.frames, args.boxes, args.output, args.size)


if __name__ == "__main__":
    main()
