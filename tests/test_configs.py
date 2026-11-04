from pathlib import Path

import pytest

from isl_slt.config import load_config


def test_overfit_is_unregularized_and_unfrozen():
    config = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    assert config.model.dropout == 0
    assert config.training.weight_decay == config.training.pose_noise_std == config.lexical.weight == 0
    assert len(config.training.stages) == 1
    assert not config.training.stages[0].freeze_t5
    assert config.generation.beams == [1]
    assert config.training.debug_only


def test_debug_training_rejects_missing_uids_before_data_loading(monkeypatch):
    from isl_slt.training.engine import run_training
    def forbidden(*args):
        raise AssertionError("Split loading must not occur")
    monkeypatch.setattr("isl_slt.training.engine.checked_splits", forbidden)
    config = load_config(Path(__file__).resolve().parents[1] / "configs/overfit.yaml")
    with pytest.raises(ValueError, match="overfit --uids"):
        run_training(config)


def test_amd_pilots_are_matched_except_precision_and_run_identity():
    root = Path(__file__).resolve().parents[1] / "configs"
    fp32 = load_config(root / "amd_fp32_pilot.yaml")
    bf16 = load_config(root / "amd_bf16_pilot.yaml")
    assert fp32.data.workers == bf16.data.workers == 8
    assert fp32.runtime.device == bf16.runtime.device == "cuda"
    assert fp32.runtime.recurrent_fp32 and bf16.runtime.recurrent_fp32
    assert fp32.runtime.determinism_mode == bf16.runtime.determinism_mode == "strict"
    assert [stage.epochs for stage in fp32.training.stages] == [1, 1]
    assert fp32.training.seed == 42
    assert fp32.training.micro_batch_size == 32
    assert fp32.training.gradient_accumulation_steps == 2
    assert fp32.generation.beams == [1]
    left, right = fp32.model_dump(), bf16.model_dump()
    for value in (left, right):
        value.pop("experiment")
        value.pop("run_dir")
        value["runtime"].pop("precision")
    assert left == right
    assert load_config(root / "baseline.yaml").data.workers == 0
