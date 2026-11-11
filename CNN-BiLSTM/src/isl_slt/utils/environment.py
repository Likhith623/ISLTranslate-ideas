"""Validate managed notebook environments without requiring or replacing their torch build."""

import importlib.metadata
import platform

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.version import Version


def check_managed_environment() -> dict:
    python = platform.python_version()
    if Version(python) not in SpecifierSet(">=3.11,<3.13"):
        raise ValueError(f"Managed Python {python} is unsupported; use Python 3.11 or 3.12")
    installed, errors = {}, []
    requirements = importlib.metadata.requires("isl-slt") or []
    for text in [*requirements, "torch>=2.6,<3"]:
        requirement = Requirement(text)
        if requirement.marker and not requirement.marker.evaluate({"extra": ""}):
            continue
        try:
            actual = importlib.metadata.version(requirement.name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"Missing {requirement.name}")
            continue
        installed[requirement.name] = actual
        if Version(actual) not in requirement.specifier:
            errors.append(f"{requirement.name} {actual} does not satisfy {requirement.specifier}")
    if errors:
        raise ValueError("Managed environment mismatch: " + "; ".join(errors))
    import torch
    return {"python": python, "packages": installed, "torch": str(torch.__version__),
            "hip": torch.version.hip, "cuda": torch.version.cuda,
            "gpu_available": torch.cuda.is_available(), "numerical_probe": "Run env-check separately"}
