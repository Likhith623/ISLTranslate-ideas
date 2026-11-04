"""Offline fixtures. No network or pretrained model assets are needed."""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import pytest

from isl_slt.data.manifests import Record


@pytest.fixture
def count_file_hash_calls(monkeypatch):
    """Start counting explicitly, after fixtures have hashed their input assets."""
    from isl_slt.utils import io
    original = io.file_hash
    def start():
        calls = []
        def counted(path):
            calls.append(str(path))
            return original(path)
        monkeypatch.setattr(io, "file_hash", counted)
        return calls
    return start


@pytest.fixture(autouse=True)
def cpu_threads():
    import torch
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def record_factory(tmp_path):
    def make(uid="sample-1", **overrides):
        values = dict(uid=uid, source_id=uid.rsplit("-", 1)[0], source_verified=True,
                      source_provenance="synthetic-test", dataset_name="synthetic",
                      dataset_revision="test-v1", text="a synthetic test sentence",
                      pose_path=str(tmp_path / f"{uid}.pose"))
        return Record(**(values | overrides))
    return make


@pytest.fixture
def pose_writer():
    def write(path, identity=0, frames=7):
        import numpy as np
        from pathlib import Path
        from pose_format import Pose
        from pose_format.numpy import NumPyPoseBody
        from pose_format.pose_header import PoseHeader, PoseHeaderComponent, PoseHeaderDimensions
        from isl_slt.features.pose import BODY, LEFT, RIGHT, UPPER, HAND
        components = [PoseHeaderComponent(name, list(points), [], [], "XYC")
                      for name, points in ((BODY, UPPER), (LEFT, HAND), (RIGHT, HAND))]
        values = np.zeros((frames, 1, 48, 2), np.float32)
        values[..., 0] = np.arange(48) + 100
        values[..., 1] = np.arange(48) + 200
        values[:, 0, 0, 0], values[:, 0, 1, 0] = 100, 200
        values[:, 0, 6:, 0] += identity * .07 + np.arange(frames)[:, None] * .1
        pose = Pose(PoseHeader(.1, PoseHeaderDimensions(640, 480), components),
                    NumPyPoseBody(25, values, np.ones((frames, 1, 48), np.float32)))
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pose.write(stream)
        return path
    return write


@pytest.fixture
def synthetic_project(tmp_path, pose_writer):
    def create(decoder="lstm"):
        import csv
        from pathlib import Path
        import yaml
        from isl_slt.config import Config
        from isl_slt.utils.io import file_hash, write_json
        root = tmp_path / decoder
        root.mkdir()
        annotations, sources, gold = [], [], []
        for source in range(10):
            for clip in range(8):
                uid = f"parent{source}-{clip}"
                text = f"a synthetic sentence number {source} {clip}"
                annotations.append((uid, text))
                sources.append((uid, f"parent{source}", "true", "synthetic fixture"))
                pose_writer(root / "poses" / f"{uid}.pose", source * 8 + clip)
                if source == 0 and clip == 0:
                    gold.append((uid, text, "an interpreter synthetic sentence"))
        for name, header, rows in (("annotations.csv", ["uid", "text"], annotations),
                ("sources.csv", ["uid", "source_id", "verified", "provenance"], sources),
                ("gold.csv", ["uid", "Transcribed Text", "Gold Translation"], gold)):
            with (root / name).open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(header)
                writer.writerows(rows)
        raw = Config().model_dump(mode="json")
        raw["run_dir"] = str(root / "run")
        raw["data"].update(manifest=str(root / "target.jsonl"), interpreter_csv=str(root / "gold.csv"),
                           split_dir=str(root / "splits"), feature_dir=str(root / "features"), workers=0)
        raw["data"]["target"].update(annotations=str(root / "annotations.csv"), pose_root=str(root / "poses"),
                                    source_mapping=str(root / "sources.csv"), dataset_revision="synthetic-v1")
        raw["model"].update(decoder=decoder, cnn_channels=[4, 8], lstm_hidden=4, dropout=.1,
                             decoder_hidden=8, decoder_embedding=8, bpe_vocab_size=300)
        raw["training"].update(max_target_epochs=2, micro_batch_size=8, gradient_accumulation_steps=2,
            max_frames_per_batch=128, retain_epoch_checkpoints=2,
            stages=[{"name": "adapter", "epochs": 1, "freeze_t5": True}, {"name": "joint", "epochs": 1}])
        raw["generation"].update(beams=[1, 2], max_new_tokens=6)
        raw["evaluation"]["metrics"] = ["bleu", "chrf", "wer"]
        raw["runtime"]["device"] = "cpu"
        if decoder == "t5":
            from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration
            from tokenizers import Tokenizer, models, pre_tokenizers, processors
            assets = root / "tiny-t5"
            vocabulary = {word: i for i, word in enumerate(["<pad>", "</s>", "<unk>", "a", "synthetic", "sentence", "number"] + list(map(str, range(10))))}
            token_backend = Tokenizer(models.WordLevel(vocabulary, unk_token="<unk>"))
            token_backend.pre_tokenizer = pre_tokenizers.Whitespace()
            token_backend.post_processor = processors.TemplateProcessing(single="$A </s>", special_tokens=[("</s>", 1)])
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=token_backend, pad_token="<pad>", eos_token="</s>", unk_token="<unk>")
            model = T5ForConditionalGeneration(T5Config(vocab_size=len(vocabulary), d_model=8, d_ff=16, d_kv=4,
                num_layers=1, num_decoder_layers=1, num_heads=2, dropout_rate=.1,
                decoder_start_token_id=0, pad_token_id=0, eos_token_id=1))
            model.save_pretrained(assets, safe_serialization=True)
            tokenizer.save_pretrained(assets)
            raw["model"].update(t5_dir=str(assets), t5_repo="synthetic/tiny-random-t5", t5_revision="fixture-v1")
            write_json(assets / "asset.json", {"repository": raw["model"]["t5_repo"], "requested_revision": "fixture-v1",
                "resolved_revision": "synthetic-local", "initializer": "random test fixture, not pretrained",
                "files": {p.name: file_hash(p) for p in assets.iterdir() if p.is_file()}})
        config = Config.model_validate(raw)
        path = root / "config.yaml"
        path.write_text(yaml.safe_dump(config.model_dump(mode="json")))
        return config, path
    return create
