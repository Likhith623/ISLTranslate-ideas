#!/usr/bin/env python3
"""Generate translations and compute corpus metrics on a held-out manifest."""

from __future__ import annotations
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, collate_examples, validate_split_disjointness
from isl_mega.model import build_model
from isl_mega.checkpoint import load_checkpoint
from isl_mega.metrics import translation_metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="runs/evaluation/predictions.jsonl")
    p.add_argument("--beams", type=int, default=4)
    args = p.parse_args()
    cfg = load_config(args.config)
    manifest = args.manifest or cfg.data.test_manifest
    ds = ISLVideoDataset(manifest, cfg.data.crop_size, cfg.data.image_size,
                         cfg.pose.max_interpolation_gap_seconds)
    if cfg.data.strict_source_disjoint or cfg.data.require_signer_disjoint:
        train_meta = ISLVideoDataset(cfg.data.train_manifest, cfg.data.crop_size, cfg.data.image_size,
                                     cfg.pose.max_interpolation_gap_seconds)
        valid_meta = ISLVideoDataset(cfg.data.valid_manifest, cfg.data.crop_size, cfg.data.image_size,
                                     cfg.pose.max_interpolation_gap_seconds)
        split_records = {
            "train": [vars(r) for r in train_meta.records],
            "valid": [vars(r) for r in valid_meta.records],
            "test": [vars(r) for r in ds.records],
        }
        if cfg.data.strict_source_disjoint:
            validate_split_disjointness(split_records, "source_id")
        if cfg.data.require_signer_disjoint:
            validate_split_disjointness(split_records, "signer_id", require_signer_disjoint=True)
    loader = DataLoader(ds, batch_size=cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate_examples)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install `pip install -e '.[text]'` to decode with FLAN-T5") from exc
    tokenizer = AutoTokenizer.from_pretrained(cfg.decoder.tokenizer_name or cfg.decoder.model_name,
                                               local_files_only=cfg.decoder.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, with_decoder=True).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device)
    model.eval(); predictions = []; refs = []; rows = []
    with torch.no_grad():
        for batch in loader:
            keys = ("frames", "timestamps", "lengths", "pose_xy", "pose_confidence", "pose_valid",
                    "confidence_known", "boxes", "box_valid", "box_scores")
            inputs = {k: batch[k].to(device) for k in keys}
            generated = model.generate(**inputs, num_beams=args.beams,
                                       max_new_tokens=cfg.decoder.max_target_tokens)
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for uid, src, signer, pred, ref in zip(
                batch["uids"], batch["source_ids"], batch["signer_ids"], texts, batch["translations"]
            ):
                predictions.append(pred)
                refs.append(ref)
                rows.append({"uid": uid, "source_id": src, "signer_id": signer,
                             "prediction": pred, "reference": ref})
    metrics = translation_metrics(predictions, refs)
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
