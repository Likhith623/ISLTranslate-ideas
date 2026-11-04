from pathlib import Path

import numpy as np
import pytest

from isl_slt.config import Config
from isl_slt.data.dataset import PoseDataset
from isl_slt.features.cache import read_features, write_features
from isl_slt.features.pose import Prepared
from isl_slt.utils.io import file_hash
from isl_slt.data.dataset import Collator
from torch.utils.data import DataLoader


def test_preflight_hashes_once_and_rejects_later_mutation(tmp_path, record_factory, monkeypatch):
    from isl_slt.features import cache
    record, config = cached_example(tmp_path, record_factory)
    calls = []
    original = cache.file_hash
    def counted(path):
        calls.append(str(path))
        return original(path)
    monkeypatch.setattr(cache, "file_hash", counted)
    dataset = PoseDataset([record], config)
    preflight_calls = len(calls)
    dataset[0]
    dataset[0]
    assert len(calls) == preflight_calls
    arrays, _ = cache.cache_paths(config, record)
    arrays.write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed during"):
        dataset[0]


def test_worker_loader_iterates_validated_cache(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    dataset = PoseDataset([record], config)
    loader = DataLoader(dataset, batch_size=1, num_workers=2, collate_fn=Collator(labels=False))
    assert sum(len(batch.records) for batch in loader) == 1


def cached_example(tmp_path, record_factory):
    record = record_factory()
    Path(record.pose_path).write_bytes(b"synthetic raw bytes")
    record.asset_hashes = {"pose": file_hash(record.pose_path)}
    config = Config()
    config.data.feature_dir = str(tmp_path / "features")
    valid = np.ones((2, 1), bool)
    prepared = Prepared(np.ones((2, 2), np.float32), np.array([0., .04]), valid, ~valid, valid,
                        valid, np.zeros(2, int), np.zeros(2, int), np.ones((2, 2), bool),
                        {"input_dim": 2, "continuous_channels": [0, 1]})
    write_features(config, record, prepared, [record])
    return record, config


def test_validated_cache_survives_raw_removal(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    Path(record.pose_path).unlink()
    dataset = PoseDataset([record], config)
    assert len(dataset) == 1
    assert dataset[0]["features"].shape == (2, 2)
    with pytest.raises(FileNotFoundError):
        read_features(config, record, verify_assets=True)


def test_manifest_changes_cannot_reuse_cache(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    changed = record.model_copy(update={"fps": 999.})
    with pytest.raises(ValueError, match="metadata changed"):
        read_features(config, changed)


def test_full_verification_does_not_change_cache_identity(tmp_path, record_factory):
    record, config = cached_example(tmp_path, record_factory)
    cached = PoseDataset([record], config)
    verified = PoseDataset([record], config, preflight=True)
    assert cached.cache_fingerprint == verified.cache_fingerprint


def test_verify_cache_explicitly_requires_raw_assets(tmp_path, record_factory, monkeypatch):
    from isl_slt.data import splits
    from isl_slt.workflows import verify_cache
    record, config = cached_example(tmp_path, record_factory)
    Path(record.pose_path).unlink()
    monkeypatch.setattr(splits, "checked_splits", lambda data: ({"train": [record]}, {}))
    assert verify_cache(config, False)["raw_assets_verified"] is False
    with pytest.raises(FileNotFoundError):
        verify_cache(config)
