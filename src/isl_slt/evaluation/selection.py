"""One validation-only decoding contract shared by every prediction entry point."""

from isl_slt.utils.io import read_json


def load_decoding_selection(path, checkpoint_sha, validation_hash, generation_hash):
    selection = read_json(path)
    if (selection.get("checkpoint_sha256") != checkpoint_sha or selection.get("partition") != "validation"
        or selection.get("validation_membership_hash") != validation_hash
        or selection.get("generation_hash") != generation_hash):
        raise ValueError("Decoding selection must match this checkpoint, validation membership and effective generation")
    beams = selection.get("selected_beams")
    if isinstance(beams, bool) or not isinstance(beams, int) or beams < 1:
        raise ValueError("Decoding selection has an invalid beam count")
    return selection
