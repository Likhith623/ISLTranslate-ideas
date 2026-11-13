"""Translate a single canonical 3D landmark NPZ with trained weights."""
import argparse
import torch
from .data import load_pose, load_rgb
from .runtime import load_checkpoint, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pose", required=True)
    parser.add_argument("--rgb")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--max-new-tokens", type=int)
    args = parser.parse_args()
    device = resolve_device(args.device)
    model, cfg, _ = load_checkpoint(args.checkpoint, device)
    sample = load_pose(args.pose, cfg.model, cfg.data.max_frames)
    extras = {}
    if cfg.model.visual_backend != "none":
        if not args.rgb:
            raise ValueError("--rgb is required by this visual checkpoint")
        extras = {k: v[None].to(device) for k, v in load_rgb(args.rgb, sample, cfg.model).items()}
    model.eval()
    ids = model.generate(sample["pose"][None].to(device), sample["confidence"][None].to(device),
                         torch.tensor([len(sample["pose"])], device=device),
                         sample["timestamps"][None].to(device), args.max_new_tokens, **extras)
    print(model.codec.decode(ids[0]))


if __name__ == "__main__":
    main()
