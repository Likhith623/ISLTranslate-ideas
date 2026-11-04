from isl_slt.config import Config
from isl_slt.evaluation import generation
import pytest

from isl_slt.evaluation.analysis import aggregate_runs
from isl_slt.utils.io import fingerprint, write_json
from isl_slt.evaluation.metrics import score_predictions
from isl_slt.evaluation.metrics import MetricResources


def test_case_variants_keep_corpus_fourgram_policy():
    report = score_predictions(["Hello bright world today"], [["hello bright world today"]], ["bleu", "chrf"])
    assert report["scores"]["bleu4_lc"] == pytest.approx(100)
    assert report["scores"]["bleu4"] < 100
    assert report["scores"]["chrf++_lc"] == pytest.approx(100)
    assert report["signatures"]["bleu4"] != report["signatures"]["bleu4_lc"]


def test_training_beam_candidates_are_explicit(monkeypatch):
    config = Config()
    called = []
    def generated(model, tokenizer, loader, config, device, beams):
        called.append(beams)
        return []
    monkeypatch.setattr(generation, "generate_loader", generated)
    monkeypatch.setattr(generation, "score_rows", lambda *args: {"scores": {"bleu4": 0, "chrf++": 0}})
    generation.select_decoding(None, None, [], config, None, config.generation.train_beams)
    assert called == [1]
    called.clear()
    generation.select_decoding(None, None, [], config, None)
    assert called == [1, 4]


def test_effective_generation_mismatch_cannot_aggregate(tmp_path):
    paths = []
    for seed, maximum in [(13, 64), (42, 128)]:
        effective = {"generation": {"max_new_tokens": maximum}}
        report = {"partition": "test", "membership_hash": "same", "comparison_fingerprint": "same",
                  "seed": seed, "effective_evaluation": effective,
                  "evaluation_fingerprint": fingerprint(effective), "metrics": {"scores": {"bleu4": 0}}}
        path = tmp_path / f"{seed}.json"
        write_json(path, report)
        paths.append(str(path))
    with pytest.raises(ValueError, match="identical"):
        aggregate_runs(paths)


def test_gold_subset_is_paired(monkeypatch):
    monkeypatch.setattr(generation, "score_predictions",
                        lambda predictions, references, *args: {"count": len(predictions)})
    rows = [{"uid": str(i), "source_id": "source", "prediction": "generated",
             "references": ["transcript"], "interpreter_reference": "interpreter" if i < 3 else None,
             "truncated": False} for i in range(5)]
    scores = generation.score_rows(rows, Config())
    assert scores["count"] == 5
    left, right = scores["transcript_on_gold_subset"], scores["interpreter_diagnostic"]
    assert left["count"] == right["count"] == 3
    assert left["uid_hash"] == right["uid_hash"]


def test_clip_type_reports_preserve_primary_and_empty_subsets():
    config = Config()
    config.evaluation.metrics = ["wer"]
    rows = [{"uid": str(i), "source_id": "source", "clip_type": kind, "prediction": prediction,
             "references": ["correct sentence"], "truncated": False}
            for i, (kind, prediction) in enumerate((("sentence", "correct sentence"), ("def", "wrong")))]
    report = generation.score_rows(rows, config)
    assert report["by_clip_type"]["sentence"]["scores"]["wer"] == 0
    assert report["by_clip_type"]["def"]["scores"]["wer"] > 0
    assert report["by_clip_type"]["unknown"]["status"] == "empty-subset"
    assert report["by_clip_type"]["unknown"]["uid_count"] == 0
    assert report["by_clip_type"]["sentence"]["uid_hash"] != report["by_clip_type"]["def"]["uid_hash"]


def test_semantic_asset_identity_hashes_once_and_detects_mutation(tmp_path, count_file_hash_calls):
    (tmp_path / "config.json").write_text('{"model_type": "fixture"}')
    (tmp_path / "tokenizer.json").write_text('{"fixture": true}')
    weights = tmp_path / "model.safetensors"
    weights.write_bytes(b"synthetic bytes, not real model weights")
    calls = count_file_hash_calls()
    session = MetricResources()
    first = session.model_identity(tmp_path, "same-declared-revision")
    assert session.model_identity(tmp_path, "same-declared-revision") == first
    assert len(calls) == 3
    weights.write_bytes(b"changed resource bytes")
    with pytest.raises(ValueError, match="changed during verification"):
        session.model_identity(tmp_path, "same-declared-revision")
    assert MetricResources().model_identity(tmp_path, "same-declared-revision")["assets_sha256"] != first["assets_sha256"]
