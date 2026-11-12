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


def build_eval_split_records(manifest: str | Path, cfg, ds_records: list[dict]) -> dict[str, list[dict]]:
    common = (cfg.data.crop_size, cfg.data.image_size, cfg.pose.max_interpolation_gap_seconds)
    manifest_path = Path(manifest).resolve()
    train_path = Path(cfg.data.train_manifest).resolve()
    valid_path = Path(cfg.data.valid_manifest).resolve()
    test_path = Path(cfg.data.test_manifest).resolve()

    split_records: dict[str, list[dict]] = {}
    if manifest_path == train_path:
        split_records["train"] = ds_records
    else:
        train_meta = ISLVideoDataset(cfg.data.train_manifest, *common)
        split_records["train"] = [vars(r) for r in train_meta.records]

    if manifest_path == valid_path:
        split_records["valid"] = ds_records
    else:
        valid_meta = ISLVideoDataset(cfg.data.valid_manifest, *common)
        split_records["valid"] = [vars(r) for r in valid_meta.records]

    if manifest_path == test_path:
        if "test" not in split_records:
            split_records["test"] = ds_records
    elif manifest_path in {train_path, valid_path}:
        if test_path.exists() and test_path not in {train_path, valid_path}:
            test_meta = ISLVideoDataset(cfg.data.test_manifest, *common)
            split_records["test"] = [vars(r) for r in test_meta.records]
    else:
        split_records["eval"] = ds_records
    return split_records


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/research.yaml")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", default=None)
    p.add_argument("--out", default="runs/evaluation/predictions.jsonl")
    p.add_argument("--beams", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=None)
    args = p.parse_args()
    if args.beams < 1:
        raise SystemExit("--beams must be >= 1")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    cfg = load_config(args.config)
    manifest = args.manifest or cfg.data.test_manifest
    ds = ISLVideoDataset(manifest, cfg.data.crop_size, cfg.data.image_size,
                         cfg.pose.max_interpolation_gap_seconds)
    if cfg.data.strict_source_disjoint or cfg.data.require_signer_disjoint:
        split_records = build_eval_split_records(manifest, cfg, [vars(r) for r in ds.records])
        if cfg.data.strict_source_disjoint:
            validate_split_disjointness(split_records, "source_id")
        if cfg.data.require_signer_disjoint:
            validate_split_disjointness(split_records, "signer_id", require_signer_disjoint=True)
    loader = DataLoader(ds, batch_size=args.batch_size or cfg.train.batch_size, shuffle=False,
                        num_workers=cfg.data.num_workers, collate_fn=collate_examples)
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install `pip install -e '.[text]'` to decode with FLAN-T5") from exc
    tokenizer = AutoTokenizer.from_pretrained(cfg.decoder.tokenizer_name or cfg.decoder.model_name,
                                               local_files_only=cfg.decoder.local_files_only)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg, with_decoder=True).to(device)
    load_checkpoint(args.checkpoint, model, map_location=device, restore_rng=False)
    model.eval(); predictions = []; refs = []; rows = []
    active_plans_total = 0.0
    plan_samples = 0
    stream_weights_sum = None
    spot_prob_sum = 0.0
    spot_frames = 0
    with torch.no_grad():
        for batch in loader:
            keys = ("frames", "timestamps", "lengths", "pose_xy", "pose_confidence", "pose_valid",
                    "confidence_known", "boxes", "box_valid", "box_scores")
            inputs = {k: batch[k].to(device) for k in keys}
            encoded = model.encode(**inputs)
            generated = model.decoder.generate(
                encoded["memory"], encoded["memory_mask"], encoded["plan_tokens"],
                num_beams=args.beams,
                max_new_tokens=cfg.decoder.max_target_tokens,
                plan_mask=encoded["plan_mask"],
            )
            active_plans_total += float(encoded["plan_mask"].sum().item())
            plan_samples += int(encoded["plan_mask"].shape[0])
            mask = encoded["memory_mask"][..., None].float()
            batch_w = (encoded["modality_weights"] * mask).sum(dim=(0, 1))
            stream_weights_sum = batch_w if stream_weights_sum is None else (stream_weights_sum + batch_w)
            if encoded.get("spot_prob") is not None and encoded.get("ctc_available") is not None:
                av = encoded["ctc_available"]
                spot_prob_sum += float((encoded["spot_prob"] * av).sum().item())
                spot_frames += int(av.sum().item())
            texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for uid, src, signer, pred, ref in zip(
                batch["uids"], batch["source_ids"], batch["signer_ids"], texts, batch["translations"]
            ):
                predictions.append(pred)
                refs.append(ref)
                rows.append({"uid": uid, "source_id": src, "signer_id": signer,
                             "prediction": pred, "reference": ref})
    metrics = translation_metrics(predictions, refs)
    diagnostics = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(manifest),
        "num_examples": len(rows),
        "beams": args.beams,
        "mean_active_plan_queries": active_plans_total / max(plan_samples, 1),
        "mean_spot_prob": (spot_prob_sum / spot_frames) if spot_frames > 0 else None,
        "mean_stream_weights": (stream_weights_sum / stream_weights_sum.sum().clamp_min(1e-8)).tolist()
        if stream_weights_sum is not None else None,
    }
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    (out.parent / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out.parent / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
