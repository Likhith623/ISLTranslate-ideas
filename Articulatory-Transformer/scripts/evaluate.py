#!/usr/bin/env python3
"""
Evaluation and Inference script for AB-ST.
Calculates BLEU-1, BLEU-2, BLEU-3, BLEU-4, and ROUGE-L.
"""

import argparse
import collections
import math
import sys
from pathlib import Path
from typing import List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from articulatory_transformer.config import ModelConfig


def compute_ngram_counts(tokens: List[str], n: int) -> collections.Counter:
    return collections.Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(hypothesis: str, reference: str, max_n: int = 4) -> List[float]:
    """Pure-Python BLEU score calculation for n=1..max_n."""
    hyp_tokens = hypothesis.strip().split()
    ref_tokens = reference.strip().split()
    hyp_len = len(hyp_tokens)
    ref_len = len(ref_tokens)

    if hyp_len == 0:
        return [0.0] * max_n

    # Brevity penalty
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / hyp_len)

    bleu_scores = []
    precisions = []
    for n in range(1, max_n + 1):
        hyp_ngrams = compute_ngram_counts(hyp_tokens, n)
        ref_ngrams = compute_ngram_counts(ref_tokens, n)
        total_hyp_ngrams = max(hyp_len - n + 1, 0)
        
        if total_hyp_ngrams == 0:
            precisions.append(0.0)
            bleu_scores.append(0.0)
            continue

        overlap = sum(min(count, ref_ngrams.get(ngram, 0)) for ngram, count in hyp_ngrams.items())
        p_n = overlap / total_hyp_ngrams
        precisions.append(p_n)

        # Cumulative BLEU-n
        if all(p > 0 for p in precisions):
            log_p = sum(math.log(p) for p in precisions) / len(precisions)
            score = bp * math.exp(log_p) * 100.0
        else:
            score = 0.0
        bleu_scores.append(score)

    return bleu_scores


def sentence_rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L score based on Longest Common Subsequence."""
    hyp = hypothesis.strip().split()
    ref = reference.strip().split()
    m, n = len(hyp), len(ref)
    if m == 0 or n == 0:
        return 0.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if hyp[i - 1] == ref[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    lcs = dp[m][n]
    rec = lcs / n
    prec = lcs / m
    if rec + prec == 0:
        return 0.0
    f1 = (2 * rec * prec) / (rec + prec) * 100.0
    return f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate AB-ST Model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to config YAML")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to trained checkpoint")
    args = parser.parse_args()

    cfg_path = Path(args.config)
    if cfg_path.exists():
        cfg = ModelConfig.from_yaml(cfg_path)
        print(f"Loaded config: vocab_size={cfg.data.vocab_size}, max_frames={cfg.data.max_frames}")

    # Self-test metric calculation on sample pairs
    hyp = "today the weather is sunny and bright"
    ref = "the weather today is sunny and bright"
    bleus = sentence_bleu(hyp, ref)
    rouge = sentence_rouge_l(hyp, ref)

    print(f"Metric self-test on reference pair:")
    print(f"  BLEU-1: {bleus[0]:.2f} | BLEU-2: {bleus[1]:.2f} | BLEU-3: {bleus[2]:.2f} | BLEU-4: {bleus[3]:.2f}")
    print(f"  ROUGE-L: {rouge:.2f}")


if __name__ == "__main__":
    main()

