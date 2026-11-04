"""Canonical fingerprints and atomic local artifacts. No network side effects."""

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import tempfile
import zlib
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator


@contextmanager
def directory_lock(directory: str | Path, name: str = ".writer.lock"):
    """Advisory POSIX ownership; never unlink the shared lock inode."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / name).open("a+b") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            raise ValueError(f"Another writer owns {directory}") from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def fingerprint(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def file_hash(path: str | Path) -> str:
    with Path(path).open("rb") as stream:
        return stream_identity(stream)["sha256"]


def stream_identity(source, *, destination=None, max_bytes=None, on_progress=None) -> dict:
    """Hash/count a binary stream, optionally copying it. Caller owns both streams.

    Read at most one byte past the bound and reject it before writing. Callers
    validate expected checksums inside atomic_open before publishing the result.
    """
    if max_bytes is not None and max_bytes < 0:
        raise ValueError("max_bytes must be nonnegative")
    size, crc, digest = 0, 0, hashlib.sha256()
    while block := source.read(1024 * 1024 if max_bytes is None else min(1024 * 1024, max_bytes - size + 1)):
        size += len(block)
        if max_bytes is not None and size > max_bytes:
            raise OSError("Stream exceeds declared size")
        if destination is not None:
            destination.write(block)
        digest.update(block)
        crc = zlib.crc32(block, crc)
        if on_progress is not None:
            on_progress(size)
    return {"bytes": size, "crc32": crc & 0xffffffff, "sha256": digest.hexdigest()}


class HashSession:
    """Command-scoped hashing: verify each unchanged file once, reject in-command mutations."""
    def __init__(self):
        self.entries = {}

    def hash(self, path):
        path = Path(path).resolve()
        status = path.stat()
        identity = (status.st_size, status.st_mtime_ns, status.st_ino)
        if path in self.entries:
            previous, digest = self.entries[path]
            if identity != previous:
                raise ValueError(f"Asset changed during verification: {path}")
            return digest
        digest = file_hash(path)
        self.entries[path] = (identity, digest)
        return digest


@contextmanager
def atomic_open(path: str | Path, mode: str = "wb", *, replace: bool = True) -> Iterator:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            # link publishes complete bytes atomically and refuses an existing destination.
            os.link(temporary, path)
        if os.name == "posix":
            parent = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(parent)
            finally:
                os.close(parent)
    finally:
        Path(temporary).unlink(missing_ok=True)


def write_json(path: str | Path, value: Any, immutable: bool = False) -> None:
    if immutable and Path(path).exists():
        if read_json(path) != value:
            raise ValueError(f"Refusing to replace immutable artifact: {path}")
        return
    with atomic_open(path) as stream:
        stream.write(canonical(value) + b"\n")


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict], immutable: bool = False) -> None:
    if immutable and Path(path).exists():
        if read_jsonl(path) != rows:
            raise ValueError(f"Refusing to replace immutable manifest: {path}")
        return
    with atomic_open(path) as stream:
        for row in rows:
            stream.write(canonical(row) + b"\n")


@lru_cache(maxsize=1)
def versions() -> dict:
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ("isl-slt", "torch", "torchvision", "transformers", "numpy", "pose-format",
                 "sacrebleu", "tokenizers", "pydantic"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    try:
        result["git"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        result["git"] = None
    return result


def append_event(path: str | Path, **event: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("ab") as stream:
        stream.write(canonical(event) + b"\n")


def execution_identity(root: str | Path | None = None, extras=()) -> dict:
    """Bind continuation to actual source bytes and resolved environment, without credentials."""
    package = Path(__file__).resolve().parents[1]
    root = Path(root).resolve() if root else Path.cwd().resolve()
    source = root / "src" / "isl_slt"
    source = source if source.is_dir() else package
    code = {"source_sha256": fingerprint({str(path.relative_to(source)): file_hash(path)
                for path in sorted(source.rglob("*.py"))}), "git_commit": None, "git_dirty": None}
    try:
        code["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
        code["git_dirty"] = bool(subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    packages = sorted((distribution.metadata.get("Name", "").lower().replace("_", "-"), distribution.version)
                      for distribution in importlib.metadata.distributions())
    import tomllib
    from packaging.requirements import Requirement
    import torch
    declaration = root / "pyproject.toml"
    if declaration.is_file():
        with declaration.open("rb") as stream:
            project = tomllib.load(stream)["project"]
        requirements = list(project["dependencies"])
        for extra in extras:
            requirements.extend(project.get("optional-dependencies", {}).get(extra, []))
    else:
        requirements = importlib.metadata.requires("isl-slt") or []
    names = {"torch", "torchvision"}
    for raw in requirements:
        requirement = Requirement(raw)
        if requirement.marker is None or any(requirement.marker.evaluate({"extra": extra}) for extra in ("", *extras)):
            names.add(requirement.name.lower().replace("_", "-"))
    inventory = dict(packages)
    environment = {"format": "isl-slt-environment-v2", "compatibility": {
        "python": ".".join(platform.python_version_tuple()[:2]),
        "packages": {name: inventory.get(name) for name in sorted(names)},
        "torch": str(torch.__version__), "cuda": torch.version.cuda, "hip": torch.version.hip},
        "provenance": {"python": platform.python_version(), "platform": platform.platform(), "packages": packages,
            "declared_requirements": sorted(requirements),
            "lock_sha256": file_hash(root / "uv.lock") if (root / "uv.lock").is_file() else None,
            "pyproject_sha256": file_hash(declaration) if declaration.is_file() else None}}
    return {"code": code, "environment": environment}


def identity_diff(before, after, prefix="") -> list[dict]:
    if isinstance(before, dict) and isinstance(after, dict):
        return [entry for key in sorted(before.keys() | after.keys())
                for entry in identity_diff(before.get(key), after.get(key), f"{prefix}.{key}" if prefix else key)]
    return [] if before == after else [{"field": prefix, "previous": before, "current": after}]
