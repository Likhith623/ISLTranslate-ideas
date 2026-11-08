"""Training losses and annotation utilities."""

from __future__ import annotations

import torch
from .fingerspelling import encode_characters, ctc_loss


def prepare_ctc_targets(texts: list[str], device: torch.device, max_input_length: int):
    sequences = [encode_characters(x) for x in texts]
    keep = [i for i, seq in enumerate(sequences) if seq and len(seq) <= max_input_length]
    if not keep:
        return None
    target_lengths = torch.tensor([len(sequences[i]) for i in keep], dtype=torch.long, device=device)
    targets = torch.tensor([token for i in keep for token in sequences[i]], dtype=torch.long, device=device)
    indices = torch.tensor(keep, dtype=torch.long, device=device)
    return indices, targets, target_lengths


def compute_ctc_batch(logits: torch.Tensor, input_lengths: torch.Tensor, texts: list[str],
                      availability: torch.Tensor | None = None) -> torch.Tensor:
    sequences = [encode_characters(x) for x in texts]
    selected_logits, selected_targets, selected_lengths = [], [], []
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        length = int(input_lengths[i])
        valid = (availability[i, :length].bool() if availability is not None
                 else torch.ones(length, dtype=torch.bool, device=logits.device))
        row = logits[i, :length][valid]
        repeated = sum(a == b for a, b in zip(seq, seq[1:]))
        minimum_steps = len(seq) + repeated
        if len(row) < minimum_steps:
            continue
        selected_logits.append(row)
        selected_targets.extend(seq)
        selected_lengths.append(len(seq))
    if not selected_logits:
        return logits.sum() * 0.0
    max_t = max(x.shape[0] for x in selected_logits)
    padded = logits.new_zeros((len(selected_logits), max_t, logits.shape[-1]))
    in_lengths = []
    for i, row in enumerate(selected_logits):
        padded[i, :len(row)] = row
        in_lengths.append(len(row))
    target_lengths = torch.tensor(selected_lengths, device=logits.device, dtype=torch.long)
    targets = torch.tensor(selected_targets, device=logits.device, dtype=torch.long)
    return ctc_loss(padded, torch.tensor(in_lengths, device=logits.device), targets, target_lengths)


def compute_cross_modal_alignment_loss(
    interactions: torch.Tensor,
    availability: torch.Tensor,
    source_ids: list[str],
    temperature: float = 0.07,
) -> torch.Tensor:
    """Symmetric RGB-to-pose InfoNCE with source-aware negative masking.

    Positives are same-frame, same-articulator RGB/pose pairs for hands, face
    and upper body. Other frames from the same source clip are excluded from
    the negative set, avoiding an especially harmful false-negative pattern.
    Batches with fewer than two distinct source IDs return a differentiable
    zero; they do not manufacture negatives from neighbouring frames.

    Args:
      interactions: [B,T,10,D] projected per-stream features before cross-stream interaction.
      availability: [B,T,10] valid stream masks.
      source_ids: one original source-video identifier per batch element.
      temperature: positive InfoNCE temperature.
    """
    if interactions.ndim != 4 or interactions.shape[2] != 10:
        raise ValueError("interactions must have shape [B,T,10,D]")
    if availability.shape != interactions.shape[:3]:
        raise ValueError("availability must have shape [B,T,10]")
    b, _, _, _ = interactions.shape
    if len(source_ids) != b or any(not isinstance(x, str) or not x for x in source_ids):
        raise ValueError("source_ids must contain one nonempty string per batch item")
    if temperature <= 0:
        raise ValueError("temperature must be positive")

    # RGB indices 0..3 pair with the matching pose indices 4..7. Motion and
    # fingerspelling do not have a one-to-one appearance/pose counterpart.
    pair_features_a, pair_features_b, pair_groups = [], [], []
    for rgb_index, pose_index in ((0, 4), (1, 5), (2, 7), (3, 6)):
        valid = availability[:, :, rgb_index].bool() & availability[:, :, pose_index].bool()
        batch_index, time_index = valid.nonzero(as_tuple=True)
        if batch_index.numel() == 0:
            continue
        pair_features_a.append(interactions[batch_index, time_index, rgb_index])
        pair_features_b.append(interactions[batch_index, time_index, pose_index])
        pair_groups.extend(source_ids[int(i)] for i in batch_index.tolist())

    if not pair_features_a:
        return interactions.sum() * 0.0
    a = torch.nn.functional.normalize(torch.cat(pair_features_a).float(), dim=-1)
    p = torch.nn.functional.normalize(torch.cat(pair_features_b).float(), dim=-1)
    if len(set(pair_groups)) < 2:
        return (a.sum() + p.sum()) * 0.0

    # Encode group IDs deterministically within this batch. Only pairs from a
    # different source may act as negatives; the diagonal remains the positive.
    group_lookup = {name: i for i, name in enumerate(dict.fromkeys(pair_groups))}
    group_tensor = torch.tensor([group_lookup[name] for name in pair_groups],
                                device=interactions.device, dtype=torch.long)
    allowed = group_tensor[:, None] != group_tensor[None, :]
    allowed.fill_diagonal_(True)
    logits = (a @ p.transpose(0, 1)) / temperature
    logits = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min)
    target = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (torch.nn.functional.cross_entropy(logits, target) +
                  torch.nn.functional.cross_entropy(logits.transpose(0, 1), target))
