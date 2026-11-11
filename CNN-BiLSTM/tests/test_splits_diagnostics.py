import pytest
import torch

from isl_slt.config import Data
from isl_slt.data.splits import assert_leakage_free, make_splits
from isl_slt.data.splits import export_splits
from isl_slt.evaluation.diagnostics import perturb_items


def test_source_leakage_and_pretraining_holdout_fail(record_factory):
    first = record_factory("parent-1", split="train")
    second = record_factory("parent-2", split="test")
    with pytest.raises(ValueError, match="Leakage"):
        assert_leakage_free({"train": [first], "test": [second]})
    with pytest.raises(ValueError, match="Leakage"):
        assert_leakage_free({"pretrain": [first.model_copy(update={"split": "pretrain"})], "test": [second]})


def test_external_membership_is_named_honestly(tmp_path, record_factory):
    settings = Data(protocol="external-membership-gold-safe", split_dir=str(tmp_path / "split"))
    records = {name: [record_factory(f"{name}-1", split=name)] for name in ("train", "validation", "test", "gold")}
    metadata = export_splits(records, settings)
    assert metadata["algorithm"] == "external-membership"
    assert metadata["protocol"] == "external-membership-gold-safe"


def test_splits_are_deterministic_and_reserve_whole_gold_parent(record_factory):
    records = [record_factory(f"parent{source}-{clip}", interpreter_text="interpreter" if source == 0 and clip == 0 else None)
               for source in range(8) for clip in range(2)]
    first, second = make_splits(records, Data()), make_splits(records, Data())
    assert {name: [row.uid for row in rows] for name, rows in first.items()} == {
        name: [row.uid for row in rows] for name, rows in second.items()}
    assert {row.source_id for row in first["gold"]} == {"parent0"}
    assert len(first["gold"]) == 2


def test_paired_diagnostics_keep_lengths_and_swap_different_donor(record_factory):
    items = [{"features": torch.arange(24).reshape(4, 6).float(), "record": record_factory(f"parent-{i}")}
             for i in range(3)]
    blank, _ = perturb_items(items, "blank", 7)
    assert blank[0]["features"].shape == items[0]["features"].shape
    assert torch.count_nonzero(blank[0]["features"]) == 0
    shuffled, _ = perturb_items(items, "shuffle", 7)
    assert sorted(shuffled[0]["features"][:, 0].tolist()) == sorted(items[0]["features"][:, 0].tolist())
    _, flags = perturb_items(items, "incorrect", 7)
    assert all(flag["uid"] != flag["donor_uid"] for flag in flags)
