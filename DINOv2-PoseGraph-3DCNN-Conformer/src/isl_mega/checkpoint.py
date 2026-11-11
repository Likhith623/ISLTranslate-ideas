"""Atomic checkpoint save/load utilities."""

from __future__ import annotations

import os
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, epoch=0, step=0, config=None, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "model": model.state_dict(), "epoch": int(epoch), "step": int(step),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "config": config.to_dict() if hasattr(config, "to_dict") else config,
        "extra": extra or {},
    }
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        state["scheduler"] = scheduler.state_dict()
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temp)
    os.replace(temp, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, map_location="cpu", strict=True,
                    restore_rng: bool = True):
    state = torch.load(path, map_location=map_location, weights_only=False)
    if "model" not in state:
        raise ValueError("Checkpoint does not contain a 'model' state_dict")
    model.load_state_dict(state["model"], strict=strict)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
    if scheduler is not None and "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    if restore_rng:
        if "python_rng" in state:
            random.setstate(state["python_rng"])
        if "numpy_rng" in state:
            np.random.set_state(state["numpy_rng"])
        if "torch_rng" in state:
            torch.set_rng_state(state["torch_rng"].cpu())
        if torch.cuda.is_available() and state.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
