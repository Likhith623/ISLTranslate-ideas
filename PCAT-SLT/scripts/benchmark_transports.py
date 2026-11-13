"""Measure actual action cost and Neumann orthogonality loss; no GPU speed extrapolation."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from pcat_slt.operators import cayley_action, givens_transport


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/transport_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17)
    cases = []
    with torch.no_grad():
        for n in (16, 64, 128):
            raw = torch.randn(2, 32, n, n)
            h = raw - raw.transpose(-1, -2)
            value = torch.randn(2, 32, n, 1)
            gain = torch.tensor(.7)
            calls = {"cayley": lambda: cayley_action(h, gain, value),
                     "givens": lambda: givens_transport(h, gain, value)}
            result = {"dimensions": n, "batch": 2, "cells": 32}
            for name, call in calls.items():
                call()
                durations = []
                for _ in range(10):
                    start = time.perf_counter()
                    output = call()
                    durations.append(1000 * (time.perf_counter() - start))
                result[name + "_median_ms"] = statistics.median(durations)
                result[name + "_norm_max_error"] = float((output.norm(dim=-2) - value.norm(dim=-2)).abs().max())
            cases.append(result)
        # A valid skew generator with Frobenius norm <1, exactly the review's
        # assumption, still loses orthogonality under its proposed truncation.
        a = torch.tensor([[0., .5], [-.5, 0.]], dtype=torch.float64)
        eye = torch.eye(2, dtype=a.dtype)
        approximation = (eye + a + a @ a + a @ a @ a) @ (eye + a)
        counterexample = float((approximation.T @ approximation - eye).abs().max())
    report = {"device": "cpu", "threads": 1, "repetitions": 10, "cases": cases,
              "third_order_neumann_orthogonality_error": counterexample,
              "scope": "Actions on vectors; different orthogonal maps, not equivalent outputs. CPU timings do not establish CUDA/MPS latency."}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
