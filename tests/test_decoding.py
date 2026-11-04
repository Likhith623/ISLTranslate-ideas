import pytest

from isl_slt.evaluation.selection import load_decoding_selection
from isl_slt.utils.io import write_json


@pytest.mark.parametrize("field", ["checkpoint_sha256", "validation_membership_hash", "generation_hash", "partition"])
def test_decoding_selection_rejects_wrong_identity(tmp_path, field):
    selection = {"checkpoint_sha256": "checkpoint", "validation_membership_hash": "validation",
                 "generation_hash": "generation", "partition": "validation", "selected_beams": 4}
    path = tmp_path / "decoding.json"
    write_json(path, selection)
    assert load_decoding_selection(path, "checkpoint", "validation", "generation")["selected_beams"] == 4
    write_json(path, selection | {field: "wrong"})
    with pytest.raises(ValueError, match="must match"):
        load_decoding_selection(path, "checkpoint", "validation", "generation")
