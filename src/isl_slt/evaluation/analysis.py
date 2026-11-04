"""Seed summaries, paired source bootstrap, latency and measured-throughput budget estimates."""

import math
import statistics
import time
from collections import defaultdict

import numpy as np

from isl_slt.evaluation.metrics import score_predictions
from isl_slt.utils.io import read_json
from isl_slt.utils.runtime import synchronize


def aggregate_runs(paths: list[str]) -> dict:
    reports = [read_json(path) for path in paths]
    if not reports:
        raise ValueError("No completed evaluation reports supplied")
    from isl_slt.utils.io import fingerprint
    if any("effective_evaluation" not in report or
           report.get("evaluation_fingerprint") != fingerprint(report["effective_evaluation"]) for report in reports):
        raise ValueError("Evaluation reports need valid effective-evaluation fingerprints; regenerate legacy reports")
    protocols = {(report["partition"], report["membership_hash"], report["comparison_fingerprint"],
                  report["evaluation_fingerprint"]) for report in reports}
    if len(protocols) != 1 or len({r["seed"] for r in reports}) != len(reports):
        raise ValueError("Seed aggregation requires unique seeds, identical splits/protocols and experiment settings")
    metric_names = set(reports[0]["metrics"]["scores"])
    if any(set(report["metrics"]["scores"]) != metric_names for report in reports):
        raise ValueError("Cannot summarize mismatched metric availability")
    summary = {name: {"mean": statistics.mean(values := [r["metrics"]["scores"][name] for r in reports]),
                      "std": statistics.stdev(values) if len(values) > 1 else 0.0} for name in sorted(metric_names)}
    return {"individual": [{"seed": r["seed"], "scores": r["metrics"]["scores"],
                            "unavailable": r["metrics"].get("unavailable", {}),
                            "checkpoint_bytes": r.get("checkpoint_bytes")} for r in reports],
            "summary": summary, "std_policy": "sample standard deviation; zero for a single run",
            "partition": reports[0]["partition"], "membership_hash": reports[0]["membership_hash"],
            "prediction_pooling": False}


def paired_source_bootstrap(first: list[dict], second: list[dict], samples: int, seed: int) -> dict:
    if samples < 1:
        raise ValueError("Bootstrap sample count must be positive")
    left, right = {row["uid"]: row for row in first}, {row["uid"]: row for row in second}
    if len(left) != len(first) or len(right) != len(second) or left.keys() != right.keys():
        raise ValueError("Paired bootstrap needs identical unique UID membership")
    groups = defaultdict(list)
    for uid in sorted(left):
        if (left[uid]["source_id"], left[uid]["references"]) != (right[uid]["source_id"], right[uid]["references"]):
            raise ValueError("Paired bootstrap source/reference mismatch")
        if left[uid]["source_id"] is None:
            raise ValueError("Source bootstrap requires known parent sources")
        groups[left[uid]["source_id"]].append(uid)
    sources = sorted(groups)
    if len(sources) < 2:
        raise ValueError("Source bootstrap needs at least two independent parent groups")
    rng, differences = np.random.default_rng(seed), []
    for _ in range(samples):
        uids = [uid for index in rng.integers(0, len(sources), len(sources)) for uid in groups[sources[index]]]
        scores = [score_predictions([rows[uid]["prediction"] for uid in uids], [left[uid]["references"] for uid in uids],
                                   ["bleu"])["scores"]["bleu4"] for rows in (left, right)]
        differences.append(scores[1] - scores[0])
    low, high = np.percentile(differences, [2.5, 97.5])
    return {"difference": "second minus first BLEU-4", "samples": samples, "source_groups": len(sources),
            "seed": seed, "percentile_95_interval": [float(low), float(high)]}


def benchmark(predictor, inputs, warmup: int, iterations: int) -> dict:
    if not inputs or iterations < 1:
        raise ValueError("Benchmark requires visual inputs and positive iterations")
    items = predictor.prepare(inputs)
    from isl_slt.data.dataset import Collator, LengthBuckets
    from isl_slt.utils.runtime import autocast
    import torch
    batches = LengthBuckets([len(item["features"]) for item in items], predictor.config.training.micro_batch_size,
                            predictor.config.training.max_frames_per_batch, predictor.config.training.seed, False)
    prepared_batches = [Collator(labels=False)([items[index] for index in indices]).to(predictor.device)
                        for indices in batches]
    if predictor.beams is None:
        raise ValueError("Benchmark requires validation-selected decoding")
    def model_only():
        with torch.no_grad(), autocast(predictor.config.runtime, predictor.device):
            for batch in prepared_batches:
                predictor.model.generate(**batch.visual, beams=predictor.beams,
                    max_new_tokens=predictor.config.generation.max_new_tokens,
                    length_penalty=predictor.config.generation.length_penalty)
    for _ in range(warmup):
        model_only()
    durations = {}
    for name, action in (("model_only", model_only),
                         ("end_to_end", lambda: predictor.predict(inputs))):
        measured = []
        for _ in range(iterations):
            synchronize(predictor.device)
            start = time.perf_counter()
            action()
            synchronize(predictor.device)
            measured.append(time.perf_counter() - start)
        durations[name] = {"mean_seconds": statistics.mean(measured), "median_seconds": statistics.median(measured),
                           "examples_per_second": len(inputs) / statistics.mean(measured)}
    return {"latency": durations, "batch_size": predictor.config.training.micro_batch_size,
            "examples": len(inputs), "warmup_iterations": warmup, "iterations": iterations,
            "device": str(predictor.device), "torch": torch.__version__, "hip": torch.version.hip,
            "hardware": torch.cuda.get_device_name(predictor.device) if predictor.device.type == "cuda" else "CPU",
            "parameter_counts": predictor.checkpoint["parameter_counts"],
            "context_lookahead_seconds": predictor.config.context.next_seconds if predictor.config.context.enabled else 0,
            "model_only_excludes": ["pose/video I/O", "preprocessing", "collation", "host-to-device transfer", "text decoding"],
            "causality": "offline bidirectional full-sequence model; no real-time claim"}


def estimate_budget(examples: int, epochs: int, measured_examples_per_second: float,
                    validation_seconds_per_epoch: float, hourly_price: float | None = None) -> dict:
    if (not all(math.isfinite(value) for value in (examples, epochs, measured_examples_per_second, validation_seconds_per_epoch))
        or min(examples, epochs, measured_examples_per_second) <= 0 or validation_seconds_per_epoch < 0):
        raise ValueError("Supply positive measured throughput/counts and nonnegative validation time")
    if hourly_price is not None and (not math.isfinite(hourly_price) or hourly_price < 0):
        raise ValueError("Hourly price must be finite and nonnegative")
    hours = epochs * (examples / measured_examples_per_second + validation_seconds_per_epoch) / 3600
    return {"estimated_hours": hours, "estimated_cost": hours * hourly_price if hourly_price is not None else None,
            "excludes": ["setup", "feature extraction", "failed/restarted jobs", "unmeasured experiments"],
            "basis": "user-supplied measured throughput; estimate, not a runtime or budget guarantee"}
