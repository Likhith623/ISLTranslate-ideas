#!/usr/bin/env python3
"""Train ISL-MEGA with source-disjoint validation and test manifest checks."""

from __future__ import annotations

import argparse
import csv
import json
import random
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from isl_mega.checkpoint import file_fingerprint, load_checkpoint, save_checkpoint
from isl_mega.config import load_config
from isl_mega.data import ISLVideoDataset, collate_examples, validate_split_disjointness
from isl_mega.losses import (
    compute_ctc_batch,
    compute_cross_modal_alignment_loss,
    compute_fingerspelling_spotting_loss,
    compute_fusion_entropy_loss,
    compute_quality_calibration_loss,
)
from isl_mega.model import build_model
from isl_mega.pretraining import load_ssl_backbone


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def model_inputs(batch: dict, device: torch.device) -> dict:
    keys = ("frames", "timestamps", "lengths", "pose_xy", "pose_confidence", "pose_valid",
            "confidence_known", "boxes", "box_valid", "box_scores")
    return {key: batch[key].to(device, non_blocking=True) for key in keys}


def tokenize_targets(tokenizer, texts: list[str], device: torch.device, max_tokens: int) -> torch.Tensor:
    encoded = tokenizer(texts, padding=True, truncation=False, return_tensors="pt")
    if encoded.input_ids.shape[1] > max_tokens:
        raise ValueError(
            f"Target length {encoded.input_ids.shape[1]} exceeds configured max {max_tokens}; "
            "increase max_target_tokens rather than silently truncating references"
        )
    labels = encoded.input_ids.masked_fill(encoded.attention_mask == 0, -100)
    return labels.to(device)


def run_epoch(model, loader, tokenizer, cfg, device, optimizer=None, scaler=None,
              max_steps: int | None = None, epoch: int = 0) -> dict:
    training = optimizer is not None
    model.train(training)
    sums = {
        "loss": 0.0, "translation": 0.0, "ctc": 0.0, "ot": 0.0, "cross_modal": 0.0,
        "fs_spot": 0.0, "plan_div": 0.0, "plan_ord": 0.0, "fusion_ent": 0.0,
        "fusion_calibration": 0.0,
        "counterfactual": 0.0, "cf_nll_gap": 0.0,
    }
    count = 0
    total_examples = 0
    for step, batch in enumerate(loader):
        if max_steps is not None and step >= max_steps:
            break
        inputs = model_inputs(batch, device)
        labels = tokenize_targets(tokenizer, batch["translations"], device,
                                  cfg.decoder.max_target_tokens)
        span_targets = batch.get("fingerspelling_spans")
        span_known = batch.get("fingerspelling_span_known")
        label_known = batch.get("fingerspelling_label_known")
        if span_targets is not None:
            span_targets = span_targets.to(device, non_blocking=True)
        if span_known is not None:
            span_known = span_known.to(device, non_blocking=True)
        if label_known is not None:
            label_known = label_known.to(device, non_blocking=True)
        amp_enabled = device.type == "cuda" and cfg.train.amp_dtype != "none"
        amp_dtype = torch.bfloat16 if cfg.train.amp_dtype == "bf16" else torch.float16
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_enabled):
            out = model(
                labels=labels,
                semantic_ot=cfg.decoder.use_semantic_ot and cfg.train.ot_weight > 0,
                counterfactual_grounding=(
                    cfg.decoder.use_counterfactual_grounding and cfg.train.counterfactual_weight > 0
                ),
                **inputs,
            )
            translation = out["decoder_loss"]
            zero = translation.new_zeros(())
            ctc_available = out.get("ctc_available")
            ctc_logits = out.get("ctc_logits")
            if ctc_available is not None and ctc_logits is not None and cfg.train.ctc_weight > 0:
                ctc = compute_ctc_batch(
                    ctc_logits, inputs["lengths"], batch["fingerspelling_texts"],
                    ctc_available, span_targets=span_targets, span_known=span_known,
                    label_known=label_known,
                )
            else:
                ctc = zero
            spot_logits = out.get("spot_logits")
            if spot_logits is not None and ctc_available is not None and cfg.train.fs_spot_weight > 0:
                fs_spot = compute_fingerspelling_spotting_loss(
                    spot_logits, ctc_available, inputs["lengths"],
                    batch["fingerspelling_texts"], span_targets=span_targets,
                    span_known=span_known, label_known=label_known,
                )
            else:
                fs_spot = zero
            ot = out.get("decoder_semantic_ot_loss", zero)
            if cfg.train.cross_modal_weight > 0:
                cross_modal = compute_cross_modal_alignment_loss(
                    out["modality_embeddings"], out["stream_available"],
                    batch["source_ids"], cfg.train.cross_modal_temperature,
                )
            else:
                cross_modal = zero
            plan_div = out.get("plan_diversity_loss", zero) if cfg.train.plan_diversity_weight > 0 else zero
            plan_ord = out.get("plan_order_loss", zero) if cfg.train.plan_order_weight > 0 else zero
            if cfg.train.fusion_entropy_weight > 0:
                fusion_ent = compute_fusion_entropy_loss(
                    out["modality_weights"], out["stream_available"], out["memory_mask"],
                )
            else:
                fusion_ent = zero
            if cfg.train.fusion_calibration_weight > 0:
                fusion_calibration = compute_quality_calibration_loss(
                    out["modality_weights"], out["stream_quality"], out["stream_available"],
                    out["memory_mask"],
                )
            else:
                fusion_calibration = zero
            counterfactual = out.get("decoder_counterfactual_loss", zero) if cfg.train.counterfactual_weight > 0 else zero
            cf_nll_gap = out.get("decoder_counterfactual_nll_gap", zero)
            loss = (
                translation
                + cfg.train.ctc_weight * ctc
                + cfg.train.fs_spot_weight * fs_spot
                + cfg.train.ot_weight * ot
                + cfg.train.cross_modal_weight * cross_modal
                + cfg.train.plan_diversity_weight * plan_div
                + cfg.train.plan_order_weight * plan_ord
                + cfg.train.fusion_entropy_weight * fusion_ent
                + cfg.train.fusion_calibration_weight * fusion_calibration
                + cfg.train.counterfactual_weight * counterfactual
            )
        if training:
            if scaler is not None and scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm,
                                               error_if_nonfinite=False)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm,
                                               error_if_nonfinite=True)
                optimizer.step()
        batch_examples = len(batch["uids"])
        total_examples += batch_examples
        for k, val in (
            ("loss", loss), ("translation", translation), ("ctc", ctc), ("ot", ot),
            ("cross_modal", cross_modal), ("fs_spot", fs_spot), ("plan_div", plan_div),
            ("plan_ord", plan_ord), ("fusion_ent", fusion_ent),
            ("fusion_calibration", fusion_calibration),
            ("counterfactual", counterfactual), ("cf_nll_gap", cf_nll_gap),
        ):
            sums[k] += float(val.detach()) * batch_examples
        count += 1
        if training and (step + 1) % cfg.train.log_every == 0:
            print(f"epoch={epoch + 1} step={step + 1} loss={float(loss.detach()):.5f}")
    if total_examples == 0:
        raise RuntimeError("Epoch completed without processing any examples")
    return {**{k: v / total_examples for k, v in sums.items()}, "batches": count, "examples": total_examples}


def build_epoch_scheduler(optimizer, epochs: int, warmup_ratio: float):
    """Epoch-stepped linear warmup/decay schedule (one scheduler step per epoch)."""
    warmup_epochs = int(round(epochs * warmup_ratio))

    def factor(epoch_index: int) -> float:
        if warmup_epochs > 0 and epoch_index < warmup_epochs:
            return float(epoch_index + 1) / warmup_epochs
        if warmup_epochs == 0:
            return max(1.0 - epoch_index / max(epochs, 1), 0.0)
        decay_epochs = max(epochs - warmup_epochs, 1)
        progress = (epoch_index - warmup_epochs) / decay_epochs
        return max(1.0 - progress, 0.0)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/research.yaml")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--ssl-checkpoint", default=None,
                        help="Optional Sign-aware student checkpoint from pretrain_visual.py")
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Debug-only cap for a single epoch; never use for reported experiments")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.max_steps is not None and args.max_steps <= 0:
        raise SystemExit("--max-steps must be positive")
    seed_everything(cfg.train.seed)

    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install `pip install -e '.[text]'` to train with FLAN-T5") from exc

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    if device.type == "cuda" and cfg.train.amp_dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("bf16 requested but this CUDA device does not support bfloat16; use amp_dtype: fp16")

    common = (cfg.data.crop_size, cfg.data.image_size, cfg.pose.max_interpolation_gap_seconds)
    train_ds = ISLVideoDataset(cfg.data.train_manifest, *common)
    valid_ds = ISLVideoDataset(cfg.data.valid_manifest, *common)
    test_ds = ISLVideoDataset(cfg.data.test_manifest, *common)
    split_records = {
        "train": [vars(r) for r in train_ds.records],
        "valid": [vars(r) for r in valid_ds.records],
        "test": [vars(r) for r in test_ds.records],
    }
    if cfg.data.strict_source_disjoint:
        validate_split_disjointness(split_records, "source_id")
    else:
        warnings.warn("Source-disjoint enforcement is disabled; results are not suitable for a strict paper claim.",
                      RuntimeWarning)
    if cfg.data.require_signer_disjoint:
        validate_split_disjointness(split_records, "signer_id", require_signer_disjoint=True)

    use_pin_memory = cfg.data.pin_memory and device.type == "cuda"
    loader_generator = torch.Generator()
    loader_generator.manual_seed(cfg.train.seed)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.train.batch_size, shuffle=True,
        num_workers=cfg.data.num_workers, pin_memory=use_pin_memory,
        collate_fn=collate_examples, worker_init_fn=seed_worker, generator=loader_generator,
    )
    valid_loader = DataLoader(
        valid_ds, batch_size=cfg.train.batch_size, shuffle=False,
        num_workers=cfg.data.num_workers, pin_memory=use_pin_memory,
        collate_fn=collate_examples, worker_init_fn=seed_worker, generator=loader_generator,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.decoder.tokenizer_name or cfg.decoder.model_name,
        local_files_only=cfg.decoder.local_files_only,
    )
    model = build_model(cfg, with_decoder=True).to(device)
    if args.ssl_checkpoint:
        # The SSL checkpoint stores the complete VisualBackbone state_dict. This
        # works for both the native ViT and timm/DINOv2 research backbone, provided
        # the pretraining and fine-tuning vision configurations are identical.
        load_ssl_backbone(model, args.ssl_checkpoint, strict=True)

    decoder_params = [p for n, p in model.named_parameters()
                      if n.startswith("decoder.") and p.requires_grad]
    visual_params = [p for n, p in model.named_parameters()
                     if not n.startswith("decoder.") and p.requires_grad]
    param_groups = [
        group for group in (
            {"params": visual_params, "lr": cfg.train.visual_lr},
            {"params": decoder_params, "lr": cfg.train.decoder_lr},
        )
        if group["params"]
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=cfg.train.weight_decay)
    scheduler = build_epoch_scheduler(optimizer, cfg.train.epochs, cfg.train.warmup_ratio)
    amp_fp16 = device.type == "cuda" and cfg.train.amp_dtype == "fp16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_fp16)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp_fp16)

    start_epoch = 0
    global_step = 0
    best_valid_translation = float("inf")
    if args.resume:
        state = load_checkpoint(args.resume, model, optimizer, scheduler, map_location=device,
                    scaler=scaler, expected_config=cfg)
        start_epoch = int(state["epoch"])
        global_step = int(state.get("step", 0))
        best_valid_translation = float(
            state.get("extra", {}).get(
                "best_valid_translation",
                state.get("extra", {}).get("best_valid_loss", float("inf")),
            )
        )
        loader_state = state.get("extra", {}).get("train_loader_generator_state")
        if loader_state is not None:
            loader_generator.set_state(loader_state)
        if start_epoch >= cfg.train.epochs:
            raise ValueError("Checkpoint epoch is already at or beyond configured total epochs")

    out_dir = Path(cfg.train.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg.save_json(out_dir / "resolved_config.json")
    (out_dir / "manifest_sha256.json").write_text(json.dumps({
        name: file_fingerprint(path) for name, path in (
            ("train", cfg.data.train_manifest),
            ("valid", cfg.data.valid_manifest),
            ("test", cfg.data.test_manifest),
        )
    }, indent=2), encoding="utf-8")
    fieldnames = [
        "epoch", "visual_lr", "train_loss", "valid_loss", "train_translation",
        "valid_translation", "train_ctc", "train_fs_spot", "train_ot",
        "train_cross_modal", "train_plan_div", "train_plan_ord",
        "train_fusion_ent", "train_fusion_calibration", "train_counterfactual",
        "train_cf_nll_gap", "valid_cf_nll_gap",
    ]
    log_path = out_dir / "metrics.csv"
    if args.resume and log_path.exists() and log_path.stat().st_size > 0:
        existing_header = log_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        if existing_header != fieldnames:
            log_path = out_dir / f"metrics_resumed_epoch_{start_epoch + 1}.csv"
    mode = "a" if (args.resume and log_path.exists() and log_path.stat().st_size > 0) else "w"
    write_header = mode == "w"
    with log_path.open(mode, newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for epoch in range(start_epoch, cfg.train.epochs):
            current_lr = float(optimizer.param_groups[0]["lr"])
            train_metrics = run_epoch(model, train_loader, tokenizer, cfg, device,
                                      optimizer=optimizer, scaler=scaler,
                                      max_steps=args.max_steps, epoch=epoch)
            scheduler.step()
            with torch.no_grad():
                valid_metrics = run_epoch(model, valid_loader, tokenizer, cfg, device)
            global_step += train_metrics["batches"]
            row = {
                "epoch": epoch + 1,
                "visual_lr": current_lr,
                "train_loss": train_metrics["loss"],
                "valid_loss": valid_metrics["loss"],
                "train_translation": train_metrics["translation"],
                "valid_translation": valid_metrics["translation"],
                "train_ctc": train_metrics["ctc"],
                "train_fs_spot": train_metrics["fs_spot"],
                "train_ot": train_metrics["ot"],
                "train_cross_modal": train_metrics["cross_modal"],
                "train_plan_div": train_metrics["plan_div"],
                "train_plan_ord": train_metrics["plan_ord"],
                "train_fusion_ent": train_metrics["fusion_ent"],
                "train_fusion_calibration": train_metrics["fusion_calibration"],
                "train_counterfactual": train_metrics["counterfactual"],
                "train_cf_nll_gap": train_metrics["cf_nll_gap"],
                "valid_cf_nll_gap": valid_metrics["cf_nll_gap"],
            }
            writer.writerow(row)
            handle.flush()
            print(row)
            is_best = valid_metrics["translation"] < best_valid_translation
            if is_best:
                best_valid_translation = valid_metrics["translation"]
            checkpoint_extra = {
                "best_valid_translation": best_valid_translation,
                "valid_loss": valid_metrics["loss"],
                "valid_translation": valid_metrics["translation"],
                "train_loader_generator_state": loader_generator.get_state(),
            }
            save_checkpoint(out_dir / "last.pt", model, optimizer, scheduler,
                            epoch + 1, global_step, cfg, extra=checkpoint_extra, scaler=scaler)
            if is_best:
                save_checkpoint(out_dir / "best_valid.pt", model, optimizer, scheduler,
                                epoch + 1, global_step, cfg, extra=checkpoint_extra, scaler=scaler)
            if (epoch + 1) % cfg.train.save_every_epochs == 0:
                save_checkpoint(out_dir / f"epoch_{epoch + 1:03d}.pt", model,
                                optimizer, scheduler, epoch + 1, global_step, cfg,
                                extra=checkpoint_extra, scaler=scaler)
            if args.max_steps is not None:
                print("Debug max_steps was set; training exited after one capped epoch.")
                break


if __name__ == "__main__":
    main()
