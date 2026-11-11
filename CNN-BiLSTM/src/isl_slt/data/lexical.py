"""Training-only weak English lexical supervision and training-only byte-level BPE."""

import os
import re
from collections import Counter
from pathlib import Path

import numpy as np

from isl_slt.config import Lexical, Strict
from isl_slt.data.manifests import Record, manifest_hash
from isl_slt.utils.io import fingerprint, read_json, read_jsonl, write_json

TOKEN_PATTERN = re.compile(r"\d+(?:[.,]\d+)*|[A-Za-z]+(?:'[A-Za-z]+)?")
NEGATION = {"not", "no", "never", "neither", "nor", "n't"}


class LexicalAnnotation(Strict):
    uid: str
    words: list[str]


def require_training(records: list[Record]) -> None:
    if not records or any(record.split not in ("train", "pretrain") for record in records):
        raise ValueError("Learned text resources must be fitted on training/pretraining records only")


class Lexicon:
    def __init__(self, words: list[str], settings: Lexical, training_hash: str, annotations: dict | None = None):
        self.words, self.settings, self.training_hash = words, settings, training_hash
        self.indices = {word: index for index, word in enumerate(words)}
        self.annotations = annotations or {}
        self.nlp = None
        if settings.method == "spacy":
            import spacy
            try:
                self.nlp = spacy.load(settings.spacy_model, disable=["ner", "parser"])
            except (OSError, ImportError) as error:
                raise RuntimeError("Prepare the configured spaCy model locally; no automatic download is performed") from error

    def extract(self, text: str, uid: str) -> set[str]:
        if self.settings.method == "precomputed":
            if uid not in self.annotations:
                raise ValueError(f"Missing explicit lexical annotation: {uid}")
            return {str(word).lower() for word in self.annotations[uid]}
        if self.nlp is not None:
            return {token.lemma_.lower() if token.lemma_ else token.text.lower() for token in self.nlp(text)
                    if token.pos_ in {"NOUN", "PROPN", "VERB", "ADJ", "ADV", "NUM"}
                    or token.lower_ in NEGATION or token.like_num}
        # Deliberately retain function words, especially negation; this is weak presence, not a gloss order.
        words = {match.group().lower() for match in TOKEN_PATTERN.finditer(text)}
        if any(word.endswith("n't") for word in words):
            words.add("not")
        return words

    def encode(self, text: str, uid: str) -> np.ndarray:
        result = np.zeros(len(self.words), np.float32)
        for word in self.extract(text, uid):
            if word in self.indices:
                result[self.indices[word]] = 1
        return result

    @classmethod
    def fit(cls, records: list[Record], settings: Lexical):
        require_training(records)
        annotations = {}
        if settings.method == "precomputed":
            rows = [LexicalAnnotation.model_validate(row).model_dump() for row in read_jsonl(settings.annotations)]
            if any(not word.strip() for row in rows for word in row["words"]):
                raise ValueError("Precomputed lexical words must be nonempty strings")
            annotations = {row["uid"]: row["words"] for row in rows}
            if len(annotations) != len(rows):
                raise ValueError("Duplicated lexical annotation UID")
            allowed = {record.uid for record in records}
            annotations = {uid: words for uid, words in annotations.items() if uid in allowed}
        builder = cls([], settings, manifest_hash(records), annotations)
        counts = Counter(word for record in records for word in builder.extract(record.text, record.uid))
        words = [word for word, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))
                 if count >= settings.min_frequency][:settings.max_vocabulary]
        if not words:
            raise ValueError("Training lexical vocabulary is empty; inspect annotations and extraction settings")
        return cls(words, settings, builder.training_hash, annotations)

    def payload(self) -> dict:
        return {"words": self.words, "settings": self.settings.model_dump(mode="json"),
                "training_hash": self.training_hash, "annotations": self.annotations,
                "supervision": "weak-English-lexical-presence"}

    @property
    def hash(self) -> str:
        return fingerprint(self.payload())

    @classmethod
    def from_payload(cls, payload: dict):
        return cls(payload["words"], Lexical.model_validate(payload["settings"]), payload["training_hash"],
                   payload["annotations"])


class BPETokenizer:
    pad_token_id, unk_token_id, bos_token_id, eos_token_id = 0, 1, 2, 3

    def __init__(self, tokenizer, metadata: dict):
        self.tokenizer, self.metadata = tokenizer, metadata

    @classmethod
    def fit(cls, records: list[Record], vocabulary_size: int):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        require_training(records)
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(vocab_size=vocabulary_size, special_tokens=["<pad>", "<unk>", "<bos>", "<eos>"],
                                      initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
        tokenizer.train_from_iterator((record.text for record in sorted(records, key=lambda row: row.key)), trainer)
        return cls(tokenizer, {"training_hash": manifest_hash(records), "requested_vocabulary": vocabulary_size,
                               "type": "training-only-byte-level-BPE-v1"})

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text).ids + [self.eos_token_id]

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids, skip_special_tokens=True).strip()

    def __len__(self) -> int:
        return self.tokenizer.get_vocab_size()

    def save_pretrained(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(directory / "bpe.json"))
        write_json(directory / "bpe-metadata.json", self.metadata)

    @classmethod
    def from_pretrained(cls, directory: str | Path):
        from tokenizers import Tokenizer
        directory = Path(directory)
        return cls(Tokenizer.from_file(str(directory / "bpe.json")), read_json(directory / "bpe-metadata.json"))
