"""Paired corpus bootstrap of prediction differences (A minus B)."""
import argparse
import json
from pathlib import Path
import numpy as np
from sacrebleu.metrics import BLEU, CHRF


def load(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("prediction files need unique ids and nonempty data")
    return {r["id"]: r for r in rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a", required=True)
    parser.add_argument("--b", required=True)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--output", default="bootstrap.json")
    args = parser.parse_args()
    if args.samples < 2:
        raise ValueError("at least two bootstrap replicates required")
    a, b = load(args.a), load(args.b)
    ids = sorted(a)
    if a.keys() != b.keys() or any(a[i]["reference"] != b[i]["reference"] for i in ids):
        raise ValueError("ids and references must match exactly")
    rng = np.random.default_rng(args.seed)
    indices = rng.integers(len(ids), size=(args.samples, len(ids)))
    result = {"samples": args.samples, "seed": args.seed, "direction": "A minus B", "metrics": {}}
    for name, metric in (("bleu4", BLEU(tokenize="13a")), ("chrf++", CHRF(word_order=2))):
        def difference(selected):
            keys = [ids[int(i)] for i in selected]
            refs = [[a[i]["reference"] for i in keys]]
            return (metric.corpus_score([a[i]["hypothesis"] for i in keys], refs).score
                    - metric.corpus_score([b[i]["hypothesis"] for i in keys], refs).score)
        diffs = np.asarray([difference(index) for index in indices])
        result["metrics"][name] = {"difference": difference(range(len(ids))),
                                    "percentile_95_ci": np.quantile(diffs, [.025, .975]).tolist(),
                                    "signature": str(metric.get_signature())}
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
