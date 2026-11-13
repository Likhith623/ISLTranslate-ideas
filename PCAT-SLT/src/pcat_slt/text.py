"""Optional pretrained seq2seq decoding with portable tokenizer assets."""

from pathlib import Path
import tempfile
from torch import nn
from .tokenizer import ByteTokenizer


class HFCodec:
    def __init__(self, tokenizer, start_id, target_lang=""):
        self.tokenizer = tokenizer
        self.bos_id = int(start_id)
        self.pad_id = tokenizer.pad_token_id
        self.eos_id = tokenizer.eos_token_id
        self.vocab_size = len(tokenizer)
        self.target_lang = target_lang
        if self.pad_id is None or self.eos_id is None or self.pad_id == self.eos_id:
            raise ValueError("seq2seq tokenizer needs PAD and EOS tokens")
        if target_lang and hasattr(tokenizer, "tgt_lang"):
            tokenizer.tgt_lang = target_lang

    def encode(self, text):
        labels = self.tokenizer(text_target=text, add_special_tokens=True)["input_ids"]
        return [self.bos_id] + labels

    def decode(self, ids, errors="replace"):
        return self.tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    def serialize(self):
        with tempfile.TemporaryDirectory(prefix="pcat-tokenizer-") as directory:
            self.tokenizer.save_pretrained(directory)
            files = {p.name: p.read_bytes() for p in Path(directory).iterdir() if p.is_file()}
        return {"files": files, "start_id": self.bos_id, "target_lang": self.target_lang}

    @classmethod
    def restore(cls, payload):
        from transformers import AutoTokenizer

        with tempfile.TemporaryDirectory(prefix="pcat-tokenizer-") as directory:
            for name, content in payload["files"].items():
                if Path(name).name != name:
                    raise ValueError("invalid tokenizer asset name")
                Path(directory, name).write_bytes(content)
            tokenizer = AutoTokenizer.from_pretrained(
                directory, local_files_only=True, trust_remote_code=False
            )
        return cls(tokenizer, payload["start_id"], payload.get("target_lang", ""))


class PretrainedDecoder(nn.Module):
    def __init__(self, cfg, initialize_pretrained=True, codec_payload=None):
        super().__init__()
        from transformers import AutoConfig, AutoModelForSeq2SeqLM, AutoTokenizer

        if initialize_pretrained:
            architecture = AutoConfig.from_pretrained(
                cfg.text_model, revision=cfg.text_revision, trust_remote_code=False
            )
            if architecture.model_type not in {"t5", "mt5"}:
                raise ValueError(
                    "verified pretrained text families are T5/mT5; other models need their own shift rules"
                )
            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                cfg.text_model, revision=cfg.text_revision, trust_remote_code=False
            )
            tokenizer = AutoTokenizer.from_pretrained(
                cfg.text_model, revision=cfg.text_revision, trust_remote_code=False
            )
            start = self.model.config.decoder_start_token_id
            if start is None:
                raise ValueError("pretrained model lacks decoder_start_token_id")
            self.model.config.decoder_start_token_id = int(start)
            self.model.generation_config.decoder_start_token_id = int(start)
            self.codec = HFCodec(tokenizer, start, cfg.text_target_lang)
            cfg.text_config = self.model.config.to_dict()
        else:
            if not cfg.text_config or codec_payload is None:
                raise ValueError("checkpoint must contain text architecture and tokenizer assets")
            raw = dict(cfg.text_config)
            kind = raw.pop("model_type")
            self.model = AutoModelForSeq2SeqLM.from_config(AutoConfig.for_model(kind, **raw))
            self.codec = HFCodec.restore(codec_payload)
        if self.model.config.model_type not in {"t5", "mt5"}:
            raise ValueError("unsupported pretrained text family")
        # External PCAT memories always bypass the text encoder. Keep its
        # metadata/embedding shell for generation API compatibility, but do not
        # retain unused encoder transformer blocks or optimize their weights.
        self.model.get_encoder().block = nn.ModuleList()
        if self.codec.vocab_size > self.model.get_input_embeddings().num_embeddings:
            raise ValueError("tokenizer vocabulary exceeds model embedding table")
        self.projection = nn.Linear(cfg.width, self.model.config.d_model)
        self.frozen = cfg.text_frozen
        if self.frozen:
            self.model.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        if self.frozen:
            self.model.eval()
        return self

    def forward(self, ids, memory, padding_mask):
        from transformers.modeling_outputs import BaseModelOutput

        target_mask = ids.ne(self.codec.pad_id)
        target_mask[:, 0] = True  # T5 uses PAD as decoder start.
        return self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=self.projection(memory)),
            attention_mask=~padding_mask,
            decoder_input_ids=ids,
            decoder_attention_mask=target_mask,
            use_cache=False,
        ).logits

    def generate(self, memory, padding_mask, limit, beams):
        from transformers.modeling_outputs import BaseModelOutput

        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=self.projection(memory)),
            attention_mask=~padding_mask,
            max_new_tokens=limit,
            num_beams=beams,
            decoder_start_token_id=self.codec.bos_id,
        )


def codec_payload(codec):
    return None if isinstance(codec, ByteTokenizer) else codec.serialize()


def load_text_codec(cfg):
    if cfg.text_backend == "byte":
        return ByteTokenizer()
    from transformers import AutoConfig, AutoTokenizer

    config = AutoConfig.from_pretrained(cfg.text_model, revision=cfg.text_revision, trust_remote_code=False)
    if config.model_type not in {"t5", "mt5"}:
        raise ValueError("verified pretrained text families are T5/mT5")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.text_model, revision=cfg.text_revision, trust_remote_code=False
    )
    start = config.decoder_start_token_id
    if start is None:
        raise ValueError("decoder_start_token_id required")
    return HFCodec(tokenizer, start, cfg.text_target_lang)
