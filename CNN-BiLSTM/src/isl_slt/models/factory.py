"""Local-asset model construction and explicit checkpoint reconstruction."""

from pathlib import Path

from isl_slt.config import Config
from isl_slt.data.lexical import BPETokenizer
from isl_slt.models.encoders import VisualEncoder
from isl_slt.models.lstm import AttentionLSTM
from isl_slt.models.t5 import T5Translator
from isl_slt.utils.io import file_hash, read_json


def load_tokenizer(config: Config, directory: str | Path | None = None):
    directory = Path(directory or config.model.t5_dir)
    if config.model.decoder == "lstm":
        return BPETokenizer.from_pretrained(directory)
    from transformers import AutoTokenizer
    if not directory.is_dir():
        raise FileNotFoundError("Run the explicit prepare-model command before loading T5 assets")
    return AutoTokenizer.from_pretrained(str(directory), local_files_only=True, trust_remote_code=False)


def build_model(config: Config, input_dim: int, tokenizer, lexical_size: int = 0,
                checkpoint_t5_config: dict | None = None, restoring: bool = False):
    visual = VisualEncoder(input_dim, config, initialize_rgb=not restoring)
    if config.model.decoder == "lstm":
        return AttentionLSTM(visual, len(tokenizer), config.model.decoder_hidden, config.model.decoder_embedding,
                             config.model.dropout, lexical_size)
    from transformers import T5Config, T5ForConditionalGeneration
    if restoring:
        if checkpoint_t5_config is None:
            raise ValueError("Checkpoint reconstruction requires its saved T5 configuration")
        # Transient initialization is immediately replaced by strict checkpoint loading; never a fallback.
        t5 = T5ForConditionalGeneration(T5Config.from_dict(checkpoint_t5_config))
    else:
        directory = Path(config.model.t5_dir)
        metadata = read_json(directory / "asset.json")
        if metadata["repository"] != config.model.t5_repo or metadata["requested_revision"] != config.model.t5_revision:
            raise ValueError("Prepared T5 repository/revision differs from configuration")
        for name, expected in metadata["files"].items():
            if file_hash(directory / name) != expected:
                raise ValueError(f"Prepared T5 checksum changed: {name}")
        t5 = T5ForConditionalGeneration.from_pretrained(str(directory), local_files_only=True,
                                                       trust_remote_code=False, use_safetensors=True)
    return T5Translator(visual, t5, lexical_size)
