#!/usr/bin/env python3
"""Sign-aware global-to-local self-distillation over unlabeled signing videos."""

from __future__ import annotations
import argparse
import os
import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from isl_mega.checkpoint import config_fingerprint, file_fingerprint
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
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--allow-external-manifest", action="store_true",
                   help="Allow an explicitly audited external SSL corpus")
    args = p.parse_args()
    if args.epochs <= 0 or args.batch_size <= 0:
        raise SystemExit("--epochs and --batch-size must be positive")
    cfg = load_config(args.config)
    manifest_path = Path(args.manifest).resolve()
    train_path = Path(cfg.data.train_manifest).resolve()
    if manifest_path != train_path and not args.allow_external_manifest:
        raise SystemExit("SSL manifest must be the configured training manifest unless --allow-external-manifest is set")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    # Unlabeled source videos are sufficient for self-supervised pretraining.
    ds = ISLVideoDataset(args.manifest, cfg.data.crop_size, cfg.data.image_size,
                         cfg.pose.max_interpolation_gap_seconds, require_translation=False)
    heldout = []
    for heldout_path in (cfg.data.valid_manifest, cfg.data.test_manifest):
        heldout_ds = ISLVideoDataset(heldout_path, cfg.data.crop_size, cfg.data.image_size,
                                     cfg.pose.max_interpolation_gap_seconds)
        heldout.extend(heldout_ds.records)
    ssl_uids = {record.uid for record in ds.records}
    ssl_sources = {record.source_id for record in ds.records}
    heldout_uids = {record.uid for record in heldout}
    heldout_sources = {record.source_id for record in heldout}
    if ssl_uids & heldout_uids or ssl_sources & heldout_sources:
        raise SystemExit("SSL manifest overlaps validation/test UIDs or source_ids")
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=cfg.data.num_workers,
                        collate_fn=collate_examples, pin_memory=(cfg.data.pin_memory and device.type == "cuda"),
                        generator=generator)
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
                "vision_config": cfg.vision.__dict__,
                "provenance": {
                    "config_sha256": config_fingerprint(cfg),
                    "manifest_sha256": file_fingerprint(manifest_path),
                    "seed": args.seed,
                    "device": str(device),
                }}, tmp)
    os.replace(tmp, out)
    print(f"Saved student backbone to {out}")

if __name__ == "__main__":
    main()
