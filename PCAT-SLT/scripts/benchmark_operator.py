"""Compare exact prefix extraction with explicit clipping on the same CPU paths."""
import argparse
import json
from pathlib import Path
import statistics
import time
import torch
from pcat_slt.operators import pyramid_statistics, polygonal_statistics_reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/operator_benchmark.json")
    parser.add_argument("--frames", type=int, default=512)
    parser.add_argument("--dimensions", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=10)
    args = parser.parse_args()
    if args.frames < 2 or args.dimensions < 1 or args.repeats < 1:
        raise ValueError("positive dimensions/repeats and at least two frames required")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    scales = (1, 4, 16, 32)
    z = torch.randn(2, args.frames, args.dimensions)
    lengths = torch.tensor([args.frames, args.frames - 1])
    metric = torch.ones(args.dimensions)

    def fast():
        return pyramid_statistics(z, lengths, scales, metric)

    def reference():
        parts = [polygonal_statistics_reference(z, lengths, k, metric) for k in scales]
        return torch.cat([p.area for p in parts], 1)

    result = {"scope": "CPU integrator forward only; excludes Cayley solves, decoder, backward and data I/O",
              "batch": 2, "frames": args.frames, "dimensions": args.dimensions,
              "scales": list(scales), "repeats": args.repeats, "threads": 1}
    with torch.no_grad():
        for label, call in (("prefix", fast), ("clipped", reference)):
            call()
            call()
            times = []
            for _ in range(args.repeats):
                before = time.perf_counter()
                call()
                times.append(1000 * (time.perf_counter() - before))
            result[label + "_median_ms"] = statistics.median(times)
        result["speedup_on_this_case"] = result["clipped_median_ms"] / result["prefix_median_ms"]
        result["float32_area_max_abs_difference"] = float((fast().area - reference()).abs().max())
        z, metric = z.double(), metric.double()
        result["float64_area_max_abs_difference"] = float((fast().area - reference()).abs().max())
        for label, call in (("prefix_float64", fast), ("clipped_float64", reference)):
            times = []
            call()
            for _ in range(args.repeats):
                before = time.perf_counter()
                call()
                times.append(1000 * (time.perf_counter() - before))
            result[label + "_median_ms"] = statistics.median(times)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
