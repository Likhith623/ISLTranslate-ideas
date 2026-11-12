#!/usr/bin/env python3
"""Sign-aware global-to-local self-distillation over unlabeled signing videos."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, collate_examples
from isl_mega.vision import crop_regions
from isl_mega.pretraining import SignAwareDistillation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/smoke.yaml")
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="artifacts/sign_ssl.pt")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=2)
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Unlabeled source videos are sufficient for self-supervised pretraining.
    ds = ISLVideoDataset(args.manifest, cfg.data.crop_size, cfg.data.image_size,
                         cfg.pose.max_interpolation_gap_seconds, require_translation=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                        collate_fn=collate_examples, pin_memory=(cfg.data.pin_memory and device.type == "cuda"))
    model = SignAwareDistillation(cfg.vision).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=0.04)
    model.train()
    for epoch in range(args.epochs):
        running, crop_total, count = 0.0, 0, 0
        for batch in loader:
            frames = batch["frames"].to(device)
            mask = batch["frame_mask"].to(device)
            boxes = batch["boxes"].to(device)
            box_valid = batch["box_valid"].to(device)
            crops = crop_regions(frames, boxes, box_valid, cfg.vision.crop_size)
            local = crops.reshape(-1, 4, 3, cfg.vision.crop_size, cfg.vision.crop_size)
            local_valid = box_valid.reshape(-1, 4)
            global_frames = frames.reshape(-1, 3, frames.shape[-2], frames.shape[-1])
            frame_valid = mask.reshape(-1) & local_valid.any(-1)
            if not frame_valid.any():
                continue
            valid_local_mask = local_valid[frame_valid]
            crop_count = int(valid_local_mask.sum().item())
            loss, center = model(global_frames[frame_valid], local[frame_valid], valid_local_mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), 1.0)
            optimizer.step()
            model.update_teacher(0.996)
            model.update_center(center)
            running += float(loss.detach()) * crop_count
            crop_total += crop_count
            count += 1
        print(f"epoch={epoch+1} loss={running/max(crop_total,1):.5f} batches={count} crops={crop_total}")
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    torch.save({"student_backbone": model.student.state_dict(),
                "teacher_backbone": model.teacher.state_dict(),
                "student_head": model.student_head.state_dict(),
                "teacher_head": model.teacher_head.state_dict(),
                "center": model.center.detach().cpu(),
                "vision_config": cfg.vision.__dict__}, tmp)
    os.replace(tmp, out)
    print(f"Saved student backbone to {out}")

if __name__ == "__main__":
    main()
