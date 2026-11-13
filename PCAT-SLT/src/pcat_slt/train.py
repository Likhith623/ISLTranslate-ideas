"""Training with token-correct accumulation, AMP, validation and epoch resume."""
import argparse
import json
import math
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from .config import load_config
from .data import SignDataset, make_collator, model_inputs, move_batch, assert_disjoint
from .model import SignTranslator
from .text import codec_payload
from .runtime import (resolve_device, seed_all, seed_worker, atomic_save, capture_rng,
                      restore_rng, write_metadata, load_checkpoint, file_sha256, input_fingerprint)


def token_loss(logits, labels, smoothing=0.0, pad_id=0):
    return F.cross_entropy(logits.flatten(0, 1), labels.flatten(), ignore_index=pad_id,
                           label_smoothing=smoothing, reduction="sum")


def make_loader(dataset, cfg, epoch, shuffle):
    generator = torch.Generator().manual_seed(cfg.train.seed + epoch)
    return DataLoader(dataset, batch_size=cfg.data.batch_size, shuffle=shuffle,
                      num_workers=cfg.data.workers, collate_fn=make_collator(dataset.tokenizer),
                      worker_init_fn=seed_worker, generator=generator,
                      pin_memory=cfg.train.device == "cuda")


@torch.no_grad()
def validation_nll(model, loader, device):
    model.eval()
    total_loss, tokens = 0.0, 0
    for batch in loader:
        batch = move_batch(batch, device)
        logits = model(**model_inputs(batch))
        total_loss += float(token_loss(logits, batch["labels"], pad_id=model.codec.pad_id))
        tokens += int(batch["labels"].ne(model.codec.pad_id).sum())
    return total_loss / tokens


def train(cfg, resume=None, stop_after_epoch=None):
    cfg.validate()
    device = resolve_device(cfg.train.device)
    seed_all(cfg.train.seed, cfg.train.deterministic)
    resume_state = None
    if resume:
        model, old_cfg, resume_state = load_checkpoint(resume, device)
        if not cfg.model.text_config:
            cfg.model.text_config = old_cfg.model.text_config
        if not cfg.model.visual_config:
            cfg.model.visual_config = old_cfg.model.visual_config
    else:
        model = SignTranslator(cfg.model).to(device)
    train_data = SignDataset(cfg.data.train, cfg.model, cfg.data, model.codec)
    val_data = SignDataset(cfg.data.val, cfg.model, cfg.data, model.codec)
    assert_disjoint({"train": train_data, "val": val_data})
    output = Path(cfg.train.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    batches = math.ceil(len(train_data) / cfg.data.batch_size)
    updates = math.ceil(batches / cfg.train.accumulation) * cfg.train.epochs

    def lr_factor(step):
        warmup = cfg.train.warmup_steps
        if step < warmup:
            return (step + 1) / max(1, warmup)
        progress = min(1.0, (step - warmup) / max(1, updates - warmup))
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    amp = cfg.train.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start, best, global_step = 0, math.inf, 0
    manifest_hashes = {name: file_sha256(getattr(cfg.data, name)) for name in ("train", "val")}
    input_hashes = {"train": input_fingerprint(train_data), "val": input_fingerprint(val_data)}
    if resume:
        state = resume_state
        old, new = old_cfg.to_dict(), cfg.to_dict()
        old["train"].pop("output_dir")
        new["train"].pop("output_dir")
        if old != new or state["manifests"] != manifest_hashes or state["inputs"] != input_hashes:
            raise ValueError("resume requires the same config (except output_dir) and unchanged input files")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start, best, global_step = state["epoch"], state["best_val_nll"], state["global_step"]
        restore_rng(state["rng"])
    elif (output / "last.pt").exists():
        raise FileExistsError("output already has last.pt; select --resume or a new output_dir")
    write_metadata(cfg, output / "run.json", input_hashes)
    if start >= cfg.train.epochs:
        return output / "last.pt"
    final_epoch = cfg.train.epochs if stop_after_epoch is None else min(stop_after_epoch, cfg.train.epochs)
    if final_epoch <= start:
        raise ValueError("stop_after_epoch must exceed the completed checkpoint epoch")
    for epoch in range(start, final_epoch):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss, running_tokens, pending_tokens = 0.0, 0, 0
        loader = make_loader(train_data, cfg, epoch, True)
        for index, batch in enumerate(loader):
            batch = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(**model_inputs(batch))
                loss = token_loss(logits, batch["labels"], cfg.train.label_smoothing, model.codec.pad_id)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss at epoch {epoch+1}, batch {index}")
            count = int(batch["labels"].ne(model.codec.pad_id).sum())
            pending_tokens += count
            running_tokens += count
            running_loss += float(loss.detach())
            scaler.scale(loss).backward()
            if (index + 1) % cfg.train.accumulation == 0 or index + 1 == len(loader):
                scaler.unscale_(optimizer)
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        parameter.grad.div_(pending_tokens)
                finite = bool(torch.stack([torch.isfinite(p.grad).all() for p in model.parameters()
                                           if p.grad is not None]).all())
                if finite:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip, error_if_nonfinite=True)
                elif not amp:
                    raise FloatingPointError("nonfinite gradients without AMP")
                # GradScaler detects overflow during unscale_, skips the update,
                # and lowers its scale; do not abort valid mixed-precision runs.
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()
                    global_step += 1
                optimizer.zero_grad(set_to_none=True)
                pending_tokens = 0
        val_nll = validation_nll(model, make_loader(val_data, cfg, epoch, False), device)
        if not math.isfinite(val_nll):
            raise FloatingPointError("validation NLL is nonfinite")
        improved = val_nll < best
        best = min(best, val_nll)
        state = {"format_version": 3, "config": cfg.to_dict(), "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "scaler": scaler.state_dict(), "rng": capture_rng(), "epoch": epoch + 1,
                 "global_step": global_step, "best_val_nll": best, "manifests": manifest_hashes,
                 "inputs": input_hashes, "text_codec": codec_payload(model.codec)}
        atomic_save(state, output / "last.pt")
        if improved:
            atomic_save(state, output / "best.pt")
        report = {"epoch": epoch + 1, "global_step": global_step,
                  "train_smoothed_ce": running_loss / running_tokens,
                  "val_nll": val_nll, "best_val_nll": best,
                  "lr": optimizer.param_groups[0]["lr"]}
        print(json.dumps(report), flush=True)
        with (output / "history.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
    return output / "last.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    train(load_config(args.config), args.resume, args.stop_after_epoch)


if __name__ == "__main__":
    main()
