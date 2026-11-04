from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from isl_slt.config import Config
from isl_slt.data.manifests import load_manifest
from isl_slt.workflows import audit


def test_missing_poses_do_not_freeze_and_retry_succeeds(tmp_path, monkeypatch):
    config = Config()
    config.data.interpreter_csv = None
    config.data.target.source_mapping = None
    config.data.target.dataset_revision = "synthetic-v1"
    config.data.target.annotations = str(tmp_path / "rows.csv")
    config.data.target.pose_root = str(tmp_path)
    config.data.manifest = str(tmp_path / "manifest.jsonl")
    Path(config.data.target.annotations).write_text("uid,text\nparent-1,test sentence\n")
    with pytest.raises(ValueError, match="No manifests were frozen"):
        audit(config)
    assert not Path(config.data.manifest).exists()
    audit(config, allow_missing_poses=True)
    draft = Path(config.data.manifest).with_suffix(".draft.json")
    with pytest.raises(ValueError, match="Draft manifests"):
        load_manifest(draft)
    assert not Path(config.data.manifest).exists()
    (tmp_path / "parent-1.pose").write_bytes(b"synthetic fixture handled by test reader")
    monkeypatch.setattr("isl_slt.data.pose.read_pose", lambda *args: SimpleNamespace(
        times=np.arange(2), metadata={"fps": 25, "components": []}))
    monkeypatch.setattr("isl_slt.features.pose.prepare_pose", lambda *args: SimpleNamespace(
        observed=np.ones((2, 48), bool), values=np.zeros((2, 384))))
    audit(config)
    assert "pose" in load_manifest(config.data.manifest)[0].asset_hashes


def test_invalid_pose_leaves_existing_manifest_untouched(tmp_path, monkeypatch):
    config = Config()
    config.data.interpreter_csv = None
    config.data.target.source_mapping = None
    config.data.target.dataset_revision = "synthetic-v1"
    config.data.target.annotations = str(tmp_path / "rows.csv")
    config.data.target.pose_root = str(tmp_path)
    config.data.manifest = str(tmp_path / "manifest.jsonl")
    Path(config.data.target.annotations).write_text("uid,text\nparent-1,text\n")
    Path(config.data.manifest).write_text("existing immutable bytes\n")
    (tmp_path / "parent-1.pose").write_bytes(b"invalid")
    def invalid(*args):
        raise ValueError("invalid pose")
    monkeypatch.setattr("isl_slt.data.pose.read_pose", invalid)
    with pytest.raises(ValueError, match="Audit failed"):
        audit(config)
    assert Path(config.data.manifest).read_text() == "existing immutable bytes\n"
