"""Reproducibility, portable trusted-state serialization and device selection."""
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import numpy as np
import torch
from .config import from_dict
from .model import SignTranslator


def resolve_device(name):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable on this execution host")
    if name not in {"cpu", "cuda", "mps"}:
        raise ValueError("only CPU, CUDA and MPS are supported")
    return torch.device(name)


def seed_all(seed, deterministic=True):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)


def seed_worker(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def capture_rng():
    state = np.random.get_state()
    return {"python": random.getstate(),
            "numpy": [state[0], state[1].tolist(), state[2], state[3], state[4]],
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "mps": torch.mps.get_rng_state() if torch.backends.mps.is_available() else None}


def restore_rng(state):
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state["cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])
    if torch.backends.mps.is_available() and state.get("mps") is not None:
        torch.mps.set_rng_state(state["mps"].cpu())


def atomic_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    temporary.replace(path)


def load_checkpoint(path, device="cpu"):
    # Only primitive containers/tensors are stored; never unpickle arbitrary classes.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") not in {2, 3}:
        raise ValueError("checkpoint format mismatch: v0.1 weights require the archived v0.1 source release")
    cfg = from_dict(checkpoint["config"])
    model = SignTranslator(cfg.model, initialize_pretrained=False,
                           text_codec_payload=checkpoint.get("text_codec")).to(device)
    model.load_state_dict(checkpoint["model"])
    return model, cfg, checkpoint


def file_sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def input_fingerprint(dataset):
    """Fingerprint both manifest semantics and every referenced NPZ file."""
    digest = hashlib.sha256()
    for row in dataset.records:
        digest.update(json.dumps(row, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        digest.update(file_sha256(row["pose"]).encode("ascii"))
        if "rgb" in row:
            digest.update(file_sha256(row["rgb"]).encode("ascii"))
    return digest.hexdigest()


def write_metadata(cfg, output, input_hashes=None):
    metadata = {"config": cfg.to_dict(), "python": platform.python_version(),
                "platform": platform.platform(), "torch": str(torch.__version__),
                "numpy": np.__version__, "cuda": torch.version.cuda, "input_fingerprints": input_hashes,
                "manifest_sha256": {name: file_sha256(getattr(cfg.data, name))
                                     for name in ("train", "val")}}
    Path(output).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata["manifest_sha256"]
