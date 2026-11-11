"""Free generation never supplies references, labels, or decoder prompts to a model."""

import torch

from isl_slt.config import Config
from isl_slt.data.lexical import BPETokenizer
from isl_slt.evaluation.metrics import MetricResources, score_predictions
from isl_slt.utils.runtime import autocast
from isl_slt.utils.io import fingerprint


def decode_tokens(tokenizer, tokens: torch.Tensor, limit: int) -> list[dict]:
    rows = []
    for sequence in tokens.cpu().tolist():
        eos = tokenizer.eos_token_id
        ended = eos in sequence
        usable = sequence[:sequence.index(eos)] if ended else sequence
        text = tokenizer.decode(usable) if isinstance(tokenizer, BPETokenizer) else tokenizer.decode(
            usable, skip_special_tokens=True, clean_up_tokenization_spaces=False).strip()
        rows.append({"prediction": text, "generated_tokens": len(usable), "ended_with_eos": ended,
                     "truncated": not ended and len(sequence) >= limit})
    return rows


@torch.no_grad()
def generate_loader(model, tokenizer, loader, config: Config, device: torch.device, beams: int) -> list[dict]:
    model.eval()
    rows = []
    for batch in loader:
        moved = batch.to(device)
        with autocast(config.runtime, device):
            tokens = model.generate(**moved.visual, beams=beams, max_new_tokens=config.generation.max_new_tokens,
                                    length_penalty=config.generation.length_penalty)
        decoded = decode_tokens(tokenizer, tokens, config.generation.max_new_tokens)
        for record, prediction, length in zip(batch.records, decoded, batch.visual["lengths"].tolist()):
            rows.append({"uid": record.uid, "source_id": record.source_id, "dataset": record.dataset_name,
                         "clip_type": record.clip_type,
                         "references": [record.text], "interpreter_reference": record.interpreter_text,
                         "input_frames": length, "split": record.split, **prediction})
    return rows


def score_rows(rows: list[dict], config: Config, metrics: list[str] | None = None, resources=None) -> dict:
    resources = resources or MetricResources()
    result = score_predictions([row["prediction"] for row in rows], [row["references"] for row in rows],
                               metrics or config.evaluation.metrics, config.evaluation.bertscore_model,
                               config.evaluation.bertscore_revision, resources)
    result["truncated_count"] = sum(row["truncated"] for row in rows)
    result["by_clip_type"] = {}
    for clip_type in ("sentence", "def", "unknown"):
        subset = [row for row in rows if row.get("clip_type", "unknown") == clip_type]
        identity = {"uid_count": len(subset), "uid_hash": fingerprint(sorted(row["uid"] for row in subset)),
                    "source_count": len({row["source_id"] for row in subset})}
        scores = score_predictions([row["prediction"] for row in subset], [row["references"] for row in subset],
            metrics or config.evaluation.metrics, config.evaluation.bertscore_model,
            config.evaluation.bertscore_revision, resources) if subset else {"scores": {}, "unavailable": {}, "status": "empty-subset"}
        result["by_clip_type"][clip_type] = {**scores, **identity}
    gold = [row for row in rows if row.get("interpreter_reference") is not None]
    if gold:
        uids = sorted(row["uid"] for row in gold)
        if len(set(uids)) != len(uids):
            raise ValueError("Gold scoring requires unique paired UIDs")
        identity = {"uid_count": len(uids), "uid_hash": fingerprint(uids),
                    "source_count": len({row["source_id"] for row in gold})}
        result["transcript_on_gold_subset"] = score_predictions([row["prediction"] for row in gold],
            [row["references"] for row in gold], metrics or config.evaluation.metrics,
            config.evaluation.bertscore_model, config.evaluation.bertscore_revision, resources)
        result["transcript_on_gold_subset"].update(identity)
        result["interpreter_diagnostic"] = score_predictions([row["prediction"] for row in gold],
            [[row["interpreter_reference"]] for row in gold], metrics or config.evaluation.metrics,
            config.evaluation.bertscore_model, config.evaluation.bertscore_revision, resources)
        result["interpreter_diagnostic"].update(identity)
        result["gold_subset_identity"] = identity
    return result


def select_decoding(model, tokenizer, loader, config: Config, device: torch.device, candidate_beams=None):
    candidates = []
    for beams in sorted(set(config.generation.beams if candidate_beams is None else candidate_beams)):
        rows = generate_loader(model, tokenizer, loader, config, device, beams)
        score = score_rows(rows, config, ["bleu", "chrf"])
        candidates.append((beams, score, rows))
    best = max(candidates, key=lambda candidate: (candidate[1]["scores"]["bleu4"],
                                                candidate[1]["scores"]["chrf++"], -candidate[0]))
    return best, [{"beams": beams, "metrics": score} for beams, score, _ in candidates]
