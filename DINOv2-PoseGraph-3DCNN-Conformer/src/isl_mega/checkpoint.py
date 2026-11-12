"""Atomic checkpoint save/load utilities."""

from __future__ import annotations

import os
from pathlib import Path
import random
import hashlib
import json
import numpy as np
import torch


def config_fingerprint(config) -> str | None:
    if config is None:
        return None
    payload = config.to_dict() if hasattr(config, "to_dict") else config
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_fingerprint(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, step=0, config=None, extra=None,
                    scaler=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "epoch": int(epoch), "step": int(step),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": [s.cpu() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else None,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "provenance": {"config_sha256": config_fingerprint(config)},
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        state["scaler"] = scaler.state_dict()
    temp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(state, temp)
        os.replace(temp, path)
    except Exception:
        if temp.exists():
            temp.unlink()
        raise


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True,
                    restore_rng: bool = True, scaler=None, expected_config=None):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in state:
        raise ValueError("Checkpoint does not contain a 'model' state_dict")
    if expected_config is not None:
        expected_hash = config_fingerprint(expected_config)
        stored_hash = state.get("provenance", {}).get("config_sha256")
        if stored_hash is None:
            stored_hash = config_fingerprint(state.get("config"))
        if stored_hash != expected_hash:
            raise ValueError("Checkpoint configuration does not match the active configuration")
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if (scaler is not None and getattr(scaler, "is_enabled", lambda: False)()
            and isinstance(state.get("scaler"), dict) and state["scaler"]):
        scaler.load_state_dict(state["scaler"])
    if restore_rng:
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            cuda_rng = [s.cpu() for s in state["cuda_rng"]]
            device_count = torch.cuda.device_count()
            if len(cuda_rng) == device_count:
                torch.cuda.set_rng_state_all(cuda_rng)
            elif cuda_rng and device_count > 0:
                current = torch.cuda.current_device()
                chosen = cuda_rng[current] if current < len(cuda_rng) else cuda_rng[0]
                torch.cuda.set_rng_state(chosen, device=current)
    return state
