"""Train-only continuous-channel scaling; masks, invalid values and padding retain their contracts."""

import numpy as np

from isl_slt.features.cache import preprocessing_fingerprint
from isl_slt.utils.io import fingerprint, read_json, write_json


def fit_statistics(prepared_sequences, preprocessing: str, train_hash: str) -> dict:
    counts = sums = squares = channels = None
    for prepared in prepared_sequences:
        values = prepared.values.astype(np.float64)
        selected = prepared.metadata["continuous_channels"]
        if channels is None:
            channels = selected
            counts = np.zeros(values.shape[1], np.int64)
            sums = np.zeros(values.shape[1], np.float64)
            squares = np.zeros(values.shape[1], np.float64)
        if selected != channels or values.shape[1] != len(counts):
            raise ValueError("Mixed feature schemas during statistics fitting")
        valid = prepared.continuous_valid
        counts += valid.sum(axis=0)
        sums += np.where(valid, values, 0).sum(axis=0)
        squares += np.where(valid, values * values, 0).sum(axis=0)
    if counts is None:
        raise ValueError("Statistics require nonempty training records")
    mean = sums / np.maximum(counts, 1)
    variance = np.maximum(squares / np.maximum(counts, 1) - mean * mean, 0)
    std = np.where((counts > 1) & (variance > 1e-12), np.sqrt(variance), 1)
    payload = {"version": "continuous-statistics-v1", "preprocessing": preprocessing,
               "training_membership": train_hash, "input_dim": len(counts), "continuous_channels": channels,
               "counts": counts.tolist(), "mean": mean.tolist(), "std": std.tolist()}
    return {**payload, "signature": fingerprint(payload)}


def validate_statistics(payload: dict, preprocessing: str | None = None, train_hash: str | None = None) -> dict:
    unsigned = {key: value for key, value in payload.items() if key != "signature"}
    if payload.get("signature") != fingerprint(unsigned):
        raise ValueError("Feature statistics signature mismatch")
    if preprocessing is not None and payload["preprocessing"] != preprocessing:
        raise ValueError("Feature statistics preprocessing mismatch")
    if train_hash is not None and payload["training_membership"] != train_hash:
        raise ValueError("Feature statistics training-membership mismatch")
    if not np.isfinite(payload["mean"]).all() or not np.isfinite(payload["std"]).all() or min(payload["std"]) <= 0:
        raise ValueError("Feature statistics must be finite with positive scales")
    return payload


def load_statistics(config, train_hash=None):
    if not config.features.standardize:
        return None
    return validate_statistics(read_json(config.features.statistics_path), preprocessing_fingerprint(config), train_hash)


def apply_statistics(prepared, statistics):
    if statistics is None:
        return prepared.values
    validate_statistics(statistics)
    if prepared.values.shape[1] != statistics["input_dim"] or prepared.metadata["continuous_channels"] != statistics["continuous_channels"]:
        raise ValueError("Inference feature/statistics schema mismatch")
    values = prepared.values.copy()
    indices = statistics["continuous_channels"]
    scaled = (values[:, indices] - np.asarray(statistics["mean"])[indices]) / np.asarray(statistics["std"])[indices]
    values[:, indices] = np.where(prepared.continuous_valid[:, indices], scaled, 0)
    return values


def prepare_stats(config):
    from isl_slt.data.dataset import dependencies_for
    from isl_slt.data.splits import checked_splits
    from isl_slt.features.cache import read_features
    from isl_slt.features.cache import SignatureSession
    from isl_slt.data.context import SourceIndex
    if not config.features.standardize:
        raise ValueError("prepare-stats requires features.standardize=true")
    partitions, metadata = checked_splits(config.data)
    records = partitions["train"]
    index, signatures = SourceIndex(records), SignatureSession()
    prepared = (read_features(config, row, dependencies_for(row, index, config),
                              signatures=signatures) for row in records)
    payload = fit_statistics(prepared, preprocessing_fingerprint(config), metadata["hashes"]["train"])
    write_json(config.features.statistics_path, payload, immutable=True)
    return payload
