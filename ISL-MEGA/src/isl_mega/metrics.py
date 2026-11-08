"""Corpus-level translation metrics with explicit optional dependencies."""

from __future__ import annotations

from typing import Any


def translation_metrics(hypotheses: list[str], references: list[str]) -> dict[str, Any]:
    if len(hypotheses) != len(references) or not hypotheses:
        raise ValueError("hypotheses and references must be nonempty and equal length")
    try:
        import sacrebleu
    except ImportError as exc:
        raise ImportError("Install `pip install -e '.[eval]'` for BLEU/chrF/TER metrics") from exc
    bleu = sacrebleu.corpus_bleu(hypotheses, [references])
    chrf = sacrebleu.corpus_chrf(hypotheses, [references], word_order=2)
    ter = sacrebleu.corpus_ter(hypotheses, [references])
    result: dict[str, Any] = {
        "num_examples": len(hypotheses),
        "references_per_example": 1,
        "bleu": float(bleu.score),
        "bleu_signature": bleu.get_signature().format(),
        "chrf++": float(chrf.score),
        "chrf_signature": chrf.get_signature().format(),
        "ter": float(ter.score),
        "ter_signature": ter.get_signature().format(),
        "sacrebleu_version": getattr(sacrebleu, "__version__", "unknown"),
    }
    try:
        from rouge_score import rouge_scorer
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
        result["rougeL"] = float(
            sum(scorer.score(ref, hyp)["rougeL"].fmeasure
                for hyp, ref in zip(hypotheses, references)) / len(hypotheses)
        )
        result["rougeL_config"] = "rouge-score RougeScorer(rougeL, use_stemmer=False)"
    except ImportError:
        result["rougeL"] = None
        result["rougeL_config"] = None
    return result
