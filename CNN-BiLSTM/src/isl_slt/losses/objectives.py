"""Accumulate sums with window-wide denominators rather than averaging microbatch means."""

import torch
import torch.nn.functional as F


def translation_sum(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1),
                           ignore_index=-100, reduction="sum")


def lexical_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="sum")


def window_denominators(batches) -> tuple[int, int]:
    tokens = sum(int((batch.labels != -100).sum()) for batch in batches)
    lexical = sum(batch.lexical.numel() for batch in batches if batch.lexical is not None)
    if tokens == 0:
        raise ValueError("Accumulation window contains no valid target tokens")
    return tokens, lexical


def loss_for_window(output: dict, batch, token_count: int, lexical_count: int, weight: float):
    ce = translation_sum(output["logits"], batch.labels)
    auxiliary = ce.new_zeros(())
    if weight:
        if output["lexical_logits"] is None or batch.lexical is None or lexical_count == 0:
            raise ValueError("Enabled lexical objective lacks head or labels")
        auxiliary = lexical_sum(output["lexical_logits"], batch.lexical)
    loss = ce / token_count + weight * auxiliary / max(lexical_count, 1)
    return loss, {"ce_sum": float(ce.detach()), "lexical_sum": float(auxiliary.detach())}
