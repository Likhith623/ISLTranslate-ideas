from types import SimpleNamespace

import numpy as np

from isl_slt.features.statistics import apply_statistics, fit_statistics


def test_scaling_leaves_masks_and_invalid_values_unchanged():
    train = SimpleNamespace(values=np.array([[2., 1.], [4., 0.], [0., 1.]], dtype=np.float32),
        continuous_valid=np.array([[True, False], [True, False], [False, False]]),
        metadata={"continuous_channels": [0]})
    statistics = fit_statistics([train], "schema", "training-only")
    assert statistics["mean"] == [3., 0.]
    transformed = apply_statistics(train, statistics)
    np.testing.assert_array_equal(transformed[:, 1], train.values[:, 1])
    assert transformed[2, 0] == 0
    validation = SimpleNamespace(values=np.array([[1000000., 1.]]),
        continuous_valid=np.array([[True, False]]), metadata=train.metadata)
    apply_statistics(validation, statistics)
    assert statistics["mean"][0] == 3.


def test_prepare_stats_never_reads_validation(tmp_path, record_factory, monkeypatch):
    from isl_slt.config import Config
    from isl_slt.data import splits
    from isl_slt.features import cache
    from isl_slt.features.statistics import prepare_stats
    config = Config()
    config.features.standardize = True
    config.features.statistics_path = str(tmp_path / "stats.json")
    training = record_factory("training-1", split="train")
    validation = record_factory("validation-1", split="validation")
    monkeypatch.setattr(splits, "checked_splits", lambda data: (
        {"train": [training], "validation": [validation]}, {"hashes": {"train": "train-hash"}}))
    called = []
    def prepared(config, row, *args, **kwargs):
        called.append(row.uid)
        return SimpleNamespace(values=np.array([[2.], [4.]], dtype=np.float32),
            continuous_valid=np.ones((2, 1), bool), metadata={"continuous_channels": [0]})
    monkeypatch.setattr(cache, "read_features", prepared)
    payload = prepare_stats(config)
    assert called == ["training-1"]
    assert payload["mean"] == [3.]
