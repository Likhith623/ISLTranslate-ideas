import pytest

from isl_slt.config import Adapter
from isl_slt.data.manifests import resolve_source
from isl_slt.config import Config
from isl_slt.workflows import make_source_map
import csv


def test_candidate_export_records_unresolved_without_verifying(tmp_path):
    annotations = tmp_path / "input.csv"
    annotations.write_text("uid,text\n0EaLX2t1wPU_e1,word\ngarbage,other\n")
    config = Config()
    config.data.target.annotations = str(annotations)
    config.data.target.source_mapping = str(tmp_path / "authoritative.csv")
    output = tmp_path / "candidates.csv"
    with pytest.raises(ValueError, match="Unresolved"):
        make_source_map(config, str(output))
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert {row["verified"] for row in rows} == {"false"}
    assert not (tmp_path / "authoritative.csv").exists()


@pytest.mark.parametrize("uid,parent", [
    ("1782bea75c7d-1", "1782bea75c7d"), ("UbhONRGm-90--61", "UbhONRGm-90-"),
    ("0EaLX2t1wPU_e1", "0EaLX2t1wPU"), ("0EaLX2t1wPU_w", "0EaLX2t1wPU"),
    ("garbage", None),
])
def test_candidates_never_claim_verification(uid, parent):
    row = resolve_source(uid, {}, Adapter().candidate_uid_patterns)
    assert row["source_id"] == parent
    assert row["source_verified"] is False


def test_authoritative_source_keeps_provenance():
    override = {"source_id": "physical-parent", "source_verified": True, "source_provenance": "reviewed-recording"}
    assert resolve_source("UbhONRGm-90--61", {"UbhONRGm-90--61": override},
                          Adapter().candidate_uid_patterns)["source_id"] == "physical-parent"


def test_source_promotion_requires_explicit_def_approval_and_never_overwrites(tmp_path):
    config = Config()
    annotations = tmp_path / "input.csv"
    annotations.write_text("uid,text\nUbhONRGm-90--61,sentence\n0EaLX2t1wPU_e1,definition\n")
    config.data.target.annotations = str(annotations)
    config.data.target.dataset_revision = "pinned-test-revision"
    config.data.target.source_mapping = str(tmp_path / "sources.csv")
    with pytest.raises(ValueError, match="approve-def"):
        make_source_map(config, promote=True, evidence="reviewed publisher naming and DEF identities")
    assert not (tmp_path / "sources.csv").exists()
    make_source_map(config, promote=True, evidence="reviewed publisher naming and DEF identities", approve_def=True)
    with (tmp_path / "sources.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0]["source_id"] == "UbhONRGm-90-"
    assert {r["verified"] for r in rows} == {"true"}
    assert "pinned-test-revision" in rows[0]["provenance"]
    before = (tmp_path / "sources.csv").read_bytes()
    with pytest.raises(ValueError, match="overwrite"):
        make_source_map(config, promote=True, evidence="again", approve_def=True)
    assert (tmp_path / "sources.csv").read_bytes() == before


def test_source_promotion_requires_evidence_and_pinned_revision(tmp_path):
    config = Config()
    with pytest.raises(ValueError, match="evidence"):
        make_source_map(config, promote=True)
    with pytest.raises(ValueError, match="revision"):
        make_source_map(config, promote=True, evidence="publisher naming")


def test_custom_pattern_is_not_promoted_by_reusing_rule_group_names(tmp_path):
    config = Config()
    config.data.target.annotations = str(tmp_path / "input.csv")
    config.data.target.source_mapping = str(tmp_path / "authoritative.csv")
    config.data.target.dataset_revision = "fixture-v1"
    config.data.target.candidate_uid_pattern = r"^(?P<source>.+)_(?P<sequence>\d+)$"
    (tmp_path / "input.csv").write_text("uid,text\ncustom_1,text\n")
    with pytest.raises(ValueError, match="Custom UID patterns"):
        make_source_map(config, promote=True, evidence="custom evidence")
    assert not (tmp_path / "authoritative.csv").exists()
