import importlib.metadata
from pathlib import Path

import pytest

from isl_slt.utils.environment import check_managed_environment


def test_managed_environment_rejects_missing_dependency(monkeypatch):
    monkeypatch.setattr(importlib.metadata, "requires", lambda name: ["not-installed-package>=1"])
    def missing(name):
        raise importlib.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(importlib.metadata, "version", missing)
    with pytest.raises(ValueError, match="Managed environment mismatch"):
        check_managed_environment()


def test_shared_env_template_has_no_nvidia_workspace_setting(monkeypatch):
    import os
    import torch
    from isl_slt.utils.runtime import seed_everything
    template = Path(__file__).resolve().parents[1] / ".env.example"
    settings = dict(line.split("=", 1) for line in template.read_text().splitlines()
                    if line.strip() and not line.startswith("#"))
    assert "CUBLAS_WORKSPACE_CONFIG" not in settings
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    monkeypatch.setattr(torch.version, "hip", "synthetic-rocm")
    seed_everything(0)
    assert "CUBLAS_WORKSPACE_CONFIG" not in os.environ
