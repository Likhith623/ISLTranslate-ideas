"""Corpus metrics with reproducible SacreBLEU signatures."""
from sacrebleu.metrics import BLEU, CHRF


def rouge_l_f1(hypothesis, reference):
    h, r = hypothesis.split(), reference.split()
    if not h or not r:
        return float(h == r)
    previous = [0] * (len(r) + 1)
    for word in h:
        current = [0]
        for j, target in enumerate(r, 1):
            current.append(previous[j - 1] + 1 if word == target else max(previous[j], current[-1]))
        previous = current
    return 2 * previous[-1] / (len(h) + len(r))


def translation_metrics(hypotheses, references):
    if not hypotheses or len(hypotheses) != len(references):
        raise ValueError("equal, nonempty hypothesis/reference corpora required")
    result = {"samples": len(hypotheses), "signatures": {}}
    for n in range(1, 5):
        metric = BLEU(tokenize="13a", max_ngram_order=n, effective_order=False)
        result[f"bleu{n}"] = metric.corpus_score(hypotheses, [references]).score
        result["signatures"][f"bleu{n}"] = str(metric.get_signature())
    metric = CHRF(word_order=2)
    result["chrf++"] = metric.corpus_score(hypotheses, [references]).score
    result["signatures"]["chrf++"] = str(metric.get_signature())
    result["rouge_l_f1"] = 100 * sum(rouge_l_f1(h, r) for h, r in zip(hypotheses, references)) / len(hypotheses)
    result["exact_match"] = 100 * sum(h == r for h, r in zip(hypotheses, references)) / len(hypotheses)
    result["rouge_definition"] = "macro whitespace-token LCS F1, beta=1, case-sensitive, no stemming"
    return result
