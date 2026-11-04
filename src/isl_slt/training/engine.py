"""Single-device stage engine with window-normalized losses, free validation and boundary resume."""

import itertools
import math
import uuid
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from isl_slt.config import Config, Stage
from isl_slt.data.dataset import Collator, LengthBuckets, PoseDataset
from isl_slt.data.lexical import BPETokenizer, Lexicon
from isl_slt.data.manifests import manifest_hash
from isl_slt.data.splits import checked_splits
from isl_slt.evaluation.generation import select_decoding
from isl_slt.losses.objectives import loss_for_window, window_denominators
from isl_slt.models.factory import build_model, load_tokenizer
from isl_slt.training.checkpoint import bundle_tokenizer, compatibility, load_checkpoint, restore_tokenizer, save_checkpoint
from isl_slt.training.checkpoint import guarded_improvement, publish_snapshots, validate_resume_compatibility
from isl_slt.utils.io import append_event, directory_lock, file_hash, read_json, versions, write_json, write_jsonl
from isl_slt.utils.runtime import Budget, autocast, device_for, environment_report, restore_rng, rng_state, seed_everything, synchronize


def optimizer_for(model, config: Config):
    t5_ids = {id(parameter) for parameter in model.t5.parameters()} if hasattr(model, "t5") else set()
    rgb = model.visual.rgb_encoder
    rgb_ids = {id(parameter) for parameter in rgb.parameters()} if rgb is not None else set()
    groups = {}
    rates = {"new": config.training.lr_new_layers, "t5": config.training.lr_t5, "rgb": config.training.lr_rgb}
    for name, parameter in model.named_parameters():
        kind = "t5" if id(parameter) in t5_ids else "rgb" if id(parameter) in rgb_ids else "new"
        decay = parameter.ndim > 1 and not name.endswith("bias")
        group = groups.setdefault((kind, decay), {"params": [], "kind": kind, "base_lr": rates[kind],
            "lr": rates[kind], "weight_decay": config.training.weight_decay if decay else 0.0})
        group["params"].append(parameter)
    # Frozen parameters remain registered, with grad=None, preserving groups across stage changes.
    return torch.optim.AdamW(list(groups.values()))


def set_stage(model, stage: Stage) -> None:
    if hasattr(model, "t5"):
        for parameter in model.t5.parameters():
            parameter.requires_grad_(not stage.freeze_t5)
        if stage.freeze_t5:
            model.t5.eval()  # Eval changes dropout, never autograd connectivity.
    if model.visual.rgb_encoder is not None:
        model.visual.rgb_encoder.set_tuning(stage.tune_rgb)


def parameter_inventory(model, config: Config) -> dict:
    """Count unique parameters (including tied T5 embeddings) without changing stage state."""
    parameters = {id(p): p for p in model.parameters()}
    visual = {id(p) for p in model.visual.parameters()}
    t5 = {id(p) for p in model.t5.parameters()} if hasattr(model, "t5") else set()
    rgb_module = model.visual.rgb_encoder
    rgb = {id(p) for p in rgb_module.parameters()} if rgb_module is not None else set()
    tuned_rgb = {id(p) for p in rgb_module.network[7].parameters()} if rgb_module is not None else set()
    def count(ids):
        return sum(parameters[key].numel() for key in ids)
    stages = {}
    for stage in config.training.stages:
        enabled = (parameters.keys() - t5 - rgb) | (t5 if not stage.freeze_t5 else set()) | (tuned_rgb if stage.tune_rgb else set())
        stages[stage.name] = {"trainable": count(enabled), "visual_trainable": count(enabled & visual),
                              "t5_trainable": count(enabled & t5), "rgb_trainable": count(enabled & rgb)}
    return {"total": count(parameters.keys()), "visual": count(visual), "t5": count(t5),
            "other": count(parameters.keys() - visual - t5), "per_stage": stages,
            "counting_policy": "Unique parameter objects; tied weights counted once; buffers excluded"}


class StageScheduler:
    """A new cosine schedule per explicit stage; optimizer moments survive transitions."""
    def __init__(self, optimizer, warmup_ratio: float):
        self.optimizer, self.warmup_ratio = optimizer, warmup_ratio
        self.stage, self.completed, self.total, self.warmup = None, 0, 1, 0

    def configure(self, name: str, total: int) -> None:
        self.stage, self.completed, self.total = name, 0, max(total, 1)
        self.warmup = math.ceil(self.total * self.warmup_ratio)
        self.apply()

    def apply(self) -> None:
        upcoming = self.completed + 1
        if self.warmup and upcoming <= self.warmup:
            factor = upcoming / self.warmup
        else:
            progress = (self.completed - self.warmup) / max(self.total - self.warmup, 1)
            factor = (1 + math.cos(math.pi * min(max(progress, 0), 1))) / 2
        for group in self.optimizer.param_groups:
            group["lr"] = group["base_lr"] * factor

    def step(self) -> None:
        self.completed += 1
        self.apply()

    def state_dict(self):
        return {key: getattr(self, key) for key in ("stage", "completed", "total", "warmup", "warmup_ratio")}

    def load_state_dict(self, state):
        for key, value in state.items():
            setattr(self, key, value)
        self.apply()


def loader_for(dataset, tokenizer, config: Config, *, training: bool, offset: int = 0):
    sampler = LengthBuckets(dataset.lengths, config.training.micro_batch_size, config.training.max_frames_per_batch,
                            config.training.seed, training)
    sampler.epoch = dataset.epoch
    batches = sampler.batches()[offset:]
    generator = torch.Generator().manual_seed(config.training.seed + dataset.epoch)
    return DataLoader(dataset, batch_sampler=batches, collate_fn=Collator(tokenizer, labels=training),
                      num_workers=config.data.workers, generator=generator,
                      persistent_workers=config.data.persistent_workers and config.data.workers > 0,
                      pin_memory=config.data.pin_memory and config.runtime.device != "cpu" and torch.cuda.is_available()), sampler


def update_window(model, optimizer, scheduler, scaler, batches, config: Config, device, weight: float) -> dict:
    token_count, lexical_count = window_denominators(batches)
    optimizer.zero_grad(set_to_none=True)
    raw_ce, raw_lexical, positives = 0.0, 0.0, 0
    for batch in batches:
        moved = batch.to(device)
        with autocast(config.runtime, device):
            output = model(labels=moved.labels, **moved.visual)
            loss, components = loss_for_window(output, moved, token_count, lexical_count, weight)
        if not bool(torch.isfinite(loss)):
            optimizer.zero_grad(set_to_none=True)
            return {"updated": False, "reason": "nonfinite_loss"}
        scaler.scale(loss).backward()
        raw_ce += components["ce_sum"]
        raw_lexical += components["lexical_sum"]
        positives += int(batch.lexical.sum()) if batch.lexical is not None else 0
    scaler.unscale_(optimizer)
    gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)
    finite = bool(torch.isfinite(gradient_norm))
    if finite:
        before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        updated = scaler.get_scale() >= before
    else:
        # GradScaler recorded nonfinite gradients during unscale_; update its scale without an optimizer step.
        if scaler.is_enabled():
            scaler.update()
        updated = False
    if updated:
        scheduler.step()
    diagnostics = {}
    if config.lexical.gradient_diagnostics:
        for name in ("visual", "bridge", "lexical_head"):
            module = getattr(model, name, None)
            if module is not None and finite:
                diagnostics[name] = math.sqrt(sum(float(p.grad.detach().float().square().sum())
                                                   for p in module.parameters() if p.grad is not None))
    optimizer.zero_grad(set_to_none=True)
    return {"updated": updated, "ce": raw_ce / token_count, "lexical_raw": raw_lexical / max(lexical_count, 1),
            "lexical_weight": weight, "lexical_weighted": weight * raw_lexical / max(lexical_count, 1),
            "positive_labels": positives, "label_slots": lexical_count, "valid_tokens": token_count,
            "gradient_norm": float(gradient_norm) if finite else None, "gradient_diagnostics": diagnostics}


def run_training(config: Config, resume: str | None = None, allow_code_change: bool = False,
                 allow_environment_change: bool = False) -> dict:
    if config.training.debug_only and not config.training.overfit_uids:
        raise ValueError("Debug-only training requires overfit --uids with 32–64 inspected training UIDs")
    if not resume and (allow_code_change or allow_environment_change):
        raise ValueError("Code/environment transition overrides apply only with --resume")
    with directory_lock(config.run_dir):
        if config.runtime.precision == "bf16":
            report = environment_report(config.runtime)
            report_path = Path(config.run_dir) / "env-check.json"
            write_json(report_path, report)
            if not report["configured_precision_ok"]:
                raise RuntimeError(f"BF16 environment probes failed; inspect {report_path} before training")
            seed_everything(config.training.seed, config.runtime.deterministic,
                            config.runtime.determinism_mode == "warn")
        for temporary in (Path(config.run_dir) / "checkpoints").glob(".last.pt.*"):
            if temporary.is_file() and not temporary.is_symlink():
                temporary.unlink()
        return _run_training(config, resume, allow_code_change, allow_environment_change)


def _run_training(config: Config, resume: str | None = None, allow_code_change: bool = False,
                  allow_environment_change: bool = False) -> dict:
    partitions, split_metadata = checked_splits(config.data)
    context_partitions = dict(partitions)
    partitions = dict(partitions)
    if config.training.overfit_uids:
        requested = [line.strip() for line in Path(config.training.overfit_uids).read_text().splitlines() if line.strip()]
        if len(set(requested)) != len(requested) or not 32 <= len(requested) <= 64:
            raise ValueError("Overfit requires 32–64 unique manually inspected TRAIN UIDs")
        selected = [row for row in partitions["train"] if row.uid in requested]
        if len(selected) != len(requested):
            raise ValueError("Overfit UIDs must all belong to the immutable training partition")
        partitions["train"], partitions["validation"] = selected, selected
        split_metadata = {**split_metadata, "hashes": {**split_metadata["hashes"], "overfit": manifest_hash(selected)}}
    seed_everything(config.training.seed, config.runtime.deterministic, config.runtime.determinism_mode == "warn")
    device, root, budget = device_for(config.runtime), Path(config.run_dir), Budget(config.runtime.wall_time_seconds)
    backend = {"device": str(device), "torch": str(torch.__version__), "hip": torch.version.hip,
               "cuda": torch.version.cuda}
    if not resume and (root / "resolved.json").exists():
        raise ValueError("Run directory already exists; resume explicitly or choose a fresh run_dir")
    root.mkdir(parents=True, exist_ok=True)
    training_records = partitions["train"] + (partitions.get("pretrain", []) if any(
        stage.dataset == "pretrain" for stage in config.training.stages) else [])
    if resume:
        restored = load_checkpoint(resume)
        if restored.get("inference_only"):
            raise ValueError("Averaged inference checkpoints cannot resume optimizer training")
        if restored["runtime_backend"] != backend and not allow_environment_change:
            raise ValueError("Exact training resume requires the same PyTorch/backend/device; inference may use another backend")
        restore_tokenizer(restored["tokenizer"], root / "tokenizer")
        tokenizer = load_tokenizer(config, root / "tokenizer")
        lexicon = Lexicon.from_payload(restored["lexicon"]) if restored["lexicon"] else None
    else:
        restored = None
        tokenizer = BPETokenizer.fit(training_records, config.model.bpe_vocab_size) if config.model.decoder == "lstm" else load_tokenizer(config)
        tokenizer.save_pretrained(root / "tokenizer")
        lexicon = Lexicon.fit(training_records, config.lexical) if config.lexical.weight else None
    lexical_payload = lexicon.payload() if lexicon else None
    if config.rgb.mode == "cached" and not restored:
        from isl_slt.features.rgb import rgb_paths
        _, metadata_path = rgb_paths(config, training_records[0])
        rgb_sha = read_json(metadata_path)["signature"]["weights_sha256"]
    else:
        rgb_sha = file_hash(config.rgb.weights_path) if not restored and config.rgb.mode != "off" else None
    rgb_initializer = restored.get("rgb_initializer") if restored else (
        {"revision": config.rgb.revision, "sha256": rgb_sha,
         "crop_revision": config.rgb.crop_revision} if config.rgb.mode != "off" else None)
    if restored and config.rgb.mode == "cached" and config.data.verify_assets and (
        rgb_initializer is None or file_hash(config.rgb.weights_path) != rgb_initializer["sha256"]
    ):
        raise ValueError("Cached-RGB resume requires the original CNN initializer checksum")
    tokenizer_bundle = bundle_tokenizer(root / "tokenizer")
    for stage in config.training.stages:
        if stage.dataset == "pretrain" and "pretrain" not in partitions:
            raise ValueError("Pretraining stage requires a filtered pretrain manifest and contamination report")
    datasets = {}
    for name in {stage.dataset for stage in config.training.stages}:
        partition = "train" if name == "target" else "pretrain"
        datasets[name] = PoseDataset(partitions[partition], config, lexicon, training=True,
                                     context_records=context_partitions[partition])
    validation = PoseDataset(partitions["validation"], config, training=False,
        context_records=context_partitions["train" if config.training.overfit_uids else "validation"])
    input_dim = datasets["target"].input_dim
    if any(dataset.input_dim != input_dim for dataset in [*datasets.values(), validation]):
        raise ValueError("Stage/validation feature dimensions differ")
    if config.rgb.mode == "cached" and any(
        dataset.rgb_initializer != rgb_initializer["sha256"] for dataset in [*datasets.values(), validation]
    ):
        raise ValueError("Training stages/validation use different RGB initializers")
    artifacts = {name: dataset.cache_fingerprint for name, dataset in datasets.items()}
    artifacts["validation"] = validation.cache_fingerprint
    expected = compatibility(config, input_dim, split_metadata, tokenizer_bundle, lexical_payload, rgb_initializer, artifacts)
    if any((dataset.statistics["signature"] if dataset.statistics else None) != expected["statistics"]
           for dataset in [*datasets.values(), validation]):
        raise ValueError("Feature statistics changed across dataset preparation or checkpoint binding")
    transition = None
    if restored is not None:
        transition = validate_resume_compatibility(restored["compatibility"], expected, allow_code_change, allow_environment_change)
        if restored["runtime_backend"] != backend:
            transition = {**(transition or {}), "runtime_backend_change": {
                "previous": restored["runtime_backend"], "current": backend}, "exact_resume": False}
        if transition:
            append_event(root / "events.jsonl", kind="resume_identity_transition", **transition)
    model = build_model(config, input_dim, tokenizer, len(lexicon.words) if lexicon else 0,
                        checkpoint_t5_config=restored["t5_config"] if restored else None,
                        restoring=restored is not None).to(device)
    optimizer = optimizer_for(model, config)
    scheduler = StageScheduler(optimizer, config.training.warmup_ratio)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda" and config.runtime.precision == "fp16")
    state = {"stage_index": 0, "epoch_in_stage": 0, "global_epoch": 0, "batch_offset": 0, "updates": 0,
             "selected_beams": None, "decoding_partition": None,
             "best_bleu": -1.0, "best_chrf": -1.0, "bad_epochs": 0}
    run_id = str(uuid.uuid4())
    if restored:
        model.load_state_dict(restored["model"], strict=True)
        optimizer.load_state_dict(restored["optimizer"])
        scheduler.load_state_dict(restored["scheduler"])
        scaler.load_state_dict(restored["scaler"])
        state, run_id = restored["state"], restored["run_id"]
        if transition and transition.get("exact_resume") is False:
            state["exact_resume"] = False
        restore_rng(restored["rng"])
    write_json(root / "resolved.json", config.model_dump(mode="json"))
    write_json(root / "compatibility.json", expected)
    if lexicon:
        write_json(root / "lexicon.json", lexical_payload)
    asset = read_json(Path(config.model.t5_dir) / "asset.json") if config.model.decoder == "t5" and not restored else (
        restored.get("initializer") if restored else {"type": "training-only-BPE"})
    package_versions = versions()
    counts = parameter_inventory(model, config)
    write_json(root / "parameter_counts.json", counts)

    def payload():
        payload = {"format": "isl-slt-checkpoint-v1", "run_id": run_id, "inference_only": False,
            "checkpoint_kind": "resume",
            "model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(), "state": dict(state), "rng": rng_state(),
            "config": config.model_dump(mode="json"), "compatibility": expected,
            "tokenizer": tokenizer_bundle, "lexicon": lexical_payload,
            "t5_config": model.t5.config.to_dict() if hasattr(model, "t5") else None,
            "input_dim": input_dim, "initializer": asset, "versions": package_versions,
            "feature_statistics": datasets["target"].statistics,
            "runtime_backend": backend,
            "rgb_initializer": rgb_initializer,
            "parameter_counts": {**counts, "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}}
        return payload

    def save(name: str):
        save_checkpoint(root / "checkpoints" / name, payload())

    for stage_index in range(state["stage_index"], len(config.training.stages)):
        stage = config.training.stages[stage_index]
        dataset = datasets[stage.dataset]
        state["stage_index"] = stage_index
        if scheduler.stage != stage.name:
            steps = 0
            for epoch in range(stage.epochs):
                dataset.epoch = state["global_epoch"] + epoch
                _, sampler = loader_for(dataset, tokenizer, config, training=True)
                steps += sampler.updates(config.training.gradient_accumulation_steps)
            scheduler.configure(stage.name, steps)
            if stage.dataset == "target" and stage_index and config.training.stages[stage_index - 1].dataset == "pretrain":
                state.update(best_bleu=-1.0, best_chrf=-1.0, bad_epochs=0)
        for epoch in range(state["epoch_in_stage"], stage.epochs):
            state["epoch_in_stage"] = epoch
            dataset.epoch = state["global_epoch"]
            model.train()
            set_stage(model, stage)
            training_started = time.perf_counter()
            loader, _ = loader_for(dataset, tokenizer, config, training=True, offset=state["batch_offset"])
            iterator = iter(loader)
            state.setdefault("epoch_train_seconds", 0.0)
            state.setdefault("epoch_examples", 0)
            state.setdefault("epoch_frames", 0)
            while window := list(itertools.islice(iterator, config.training.gradient_accumulation_steps)):
                if budget.expired:
                    state["epoch_train_seconds"] += time.perf_counter() - training_started
                    save("last.pt")
                    return {"status": "wall_time_limit", "state": state, "run_dir": str(root)}
                ramp = min(1, (state["global_epoch"] + 1) / max(config.lexical.ramp_epochs, 1))
                synchronize(device)
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                update_started = time.perf_counter()
                learning_rates = {f"{group.get('kind', 'group')}-{index}": group["lr"]
                                  for index, group in enumerate(optimizer.param_groups)}
                outcome = update_window(model, optimizer, scheduler, scaler, window, config, device,
                                        config.lexical.weight * ramp)
                synchronize(device)
                elapsed = max(time.perf_counter() - update_started, 1e-12)
                examples = sum(batch.visual["features"].shape[0] for batch in window)
                frames = sum(int(batch.visual["lengths"].sum()) for batch in window)
                telemetry = {"update_seconds": elapsed, "examples": examples, "frames": frames,
                    "examples_per_second": examples / elapsed, "frames_per_second": frames / elapsed,
                    "learning_rates": learning_rates,
                    "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None}
                state["batch_offset"] += len(window)
                state["updates"] += int(outcome["updated"])
                state["epoch_examples"] += examples
                state["epoch_frames"] += frames
                append_event(root / "events.jsonl", kind="update", stage=stage.name,
                             epoch=state["global_epoch"], update=state["updates"], **telemetry, **outcome)
            state["epoch_train_seconds"] += time.perf_counter() - training_started
            if budget.expired:
                save("last.pt")
                return {"status": "wall_time_limit", "state": state, "run_dir": str(root)}
            validation_loader, _ = loader_for(validation, tokenizer, config, training=False)
            synchronize(device)
            validation_started = time.perf_counter()
            (beams, metrics, rows), candidate_scores = select_decoding(
                model, tokenizer, validation_loader, config, device, config.generation.train_beams)
            synchronize(device)
            validation_seconds = time.perf_counter() - validation_started
            state["selected_beams"] = beams
            state["decoding_policy"] = "training-validation"
            state["decoding_partition"] = "inspected-training-debug" if config.training.overfit_uids else "validation"
            bleu, chrf = metrics["scores"]["bleu4"], metrics["scores"]["chrf++"]
            improved_bleu, improved_chrf = bleu > state["best_bleu"], chrf > state["best_chrf"]
            improved_guarded = guarded_improvement(bleu, chrf, state["best_bleu"], state["best_chrf"],
                                                  config.training.guarded_chrf_tolerance)
            state["bad_epochs"] = 0 if improved_bleu else state["bad_epochs"] + 1
            state["best_bleu"], state["best_chrf"] = max(bleu, state["best_bleu"]), max(chrf, state["best_chrf"])
            state.update(global_epoch=state["global_epoch"] + 1, epoch_in_stage=epoch + 1, batch_offset=0)
            prefix = "pretrain/" if stage.dataset == "pretrain" else ""
            write_jsonl(root / "validation" / f"epoch-{state['global_epoch']:04d}.jsonl", rows)
            write_json(root / "validation" / f"epoch-{state['global_epoch']:04d}.json", {
                "selection_partition": "inspected-training-debug" if config.training.overfit_uids else "validation",
                "selected_beams": beams, "candidates": candidate_scores})
            append_event(root / "events.jsonl", kind="validation", stage=stage.name,
                         epoch=state["global_epoch"], selected_beams=beams, metrics=metrics)
            append_event(root / "events.jsonl", kind="epoch", stage=stage.name, epoch=state["global_epoch"],
                         train_seconds=state["epoch_train_seconds"], validation_seconds=validation_seconds,
                         examples=state["epoch_examples"], frames=state["epoch_frames"],
                         examples_per_second=state["epoch_examples"] / max(state["epoch_train_seconds"], 1e-12))
            state.update(epoch_train_seconds=0.0, epoch_examples=0, epoch_frames=0)
            names = [prefix + name for improved, name in (
                (improved_bleu, "best-bleu.pt"), (improved_chrf, "best-chrf.pt"),
                (improved_guarded, "best-guarded.pt")) if improved]
            if config.training.retain_epoch_checkpoints:
                names.append(f"epoch-{state['global_epoch']:04d}.pt")
            modes = publish_snapshots([root / "checkpoints" / name for name in names], payload(),
                                      config.training.inference_snapshot_dtype) if names else {}
            append_event(root / "events.jsonl", kind="checkpoint_storage", epoch=state["global_epoch"], modes=modes)
            save("last.pt")
            retained = sorted((root / "checkpoints").glob("epoch-*.pt"))
            keep = config.training.retain_epoch_checkpoints
            for old in retained[:max(0, len(retained) - keep)]:
                old.unlink()
            if stage_index == len(config.training.stages) - 1 and state["bad_epochs"] >= config.training.early_stopping_patience:
                state["epoch_in_stage"] = stage.epochs
                break
        state.update(stage_index=stage_index + 1, epoch_in_stage=0, batch_offset=0)
        save("last.pt")
    return {"status": "completed", "state": state, "run_dir": str(root)}
