from pathlib import Path

from isl_slt.data.context import SourceIndex
from isl_slt.features.cache import SignatureSession
from isl_slt.utils import io
from isl_slt.config import Config, Context
from isl_slt.data.dataset import dependencies_for
import pytest


def test_source_signatures_hash_each_asset_once(record_factory, count_file_hash_calls):
    records = []
    for number in range(200):
        record = record_factory(f"parent-{number}", split="train")
        Path(record.pose_path).write_bytes(str(number).encode())
        record.asset_hashes = {"pose": io.file_hash(record.pose_path)}
        records.append(record)
    calls = count_file_hash_calls()
    index, session = SourceIndex(records), SignatureSession()
    for record in records:
        for dependency in index.dependencies(record):
            session.signature(dependency)
    assert len(calls) == 200


def test_context_window_queries_only_overlaps(record_factory):
    records = [record_factory(f"source-{i}", split="train", timing_verified=True,
                              start_seconds=float(i), end_seconds=float(i + 1), source_duration_seconds=200)
               for i in range(200)]
    index = SourceIndex(records)
    config = Config()
    config.context.enabled = True
    for i, record in enumerate(records):
        selected = dependencies_for(record, index, config)
        assert [row.uid for row in selected] == [row.uid for row in records[max(0, i - 1):min(200, i + 2)]]
    assert index.query_nodes_visited < 200 * 40
    assert len(index.source_identity) == 1
    cross = SourceIndex([records[0], records[1].model_copy(update={"split": "test"})])
    with pytest.raises(ValueError, match="crosses partitions"):
        cross.dependencies(records[0], Context(enabled=True))


def test_non_context_cache_identity_is_unchanged():
    from isl_slt.features.cache import CACHE_VERSION, preprocessing_fingerprint
    config = Config()
    assert preprocessing_fingerprint(config) == io.fingerprint({"version": CACHE_VERSION,
        "features": config.features.model_dump(mode="json"), "context": config.context.model_dump(mode="json")})
