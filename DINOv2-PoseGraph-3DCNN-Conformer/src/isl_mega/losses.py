"""Training losses and annotation utilities."""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F
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


def compute_ctc_batch(
    logits: torch.Tensor,
    input_lengths: torch.Tensor,
    texts: list[str],
    availability: torch.Tensor | None = None,
    span_targets: torch.Tensor | None = None,
    span_known: torch.Tensor | None = None,
) -> torch.Tensor:
    sequences = [encode_characters(x) for x in texts]
    selected_logits, selected_targets, selected_lengths = [], [], []
    for i, seq in enumerate(sequences):
        if not seq:
            continue
        length = int(input_lengths[i])
        valid = (availability[i, :length].bool() if availability is not None
                 else torch.ones(length, dtype=torch.bool, device=logits.device))
        repeated = sum(a == b for a, b in zip(seq, seq[1:]))
        minimum_steps = len(seq) + repeated
        if span_targets is not None and span_known is not None and bool(span_known[i]):
            spanned = valid & span_targets[i, :length].bool()
            if int(spanned.sum()) >= minimum_steps:
                valid = spanned
        row = logits[i, :length][valid]
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


def compute_fingerspelling_spotting_loss(
    spot_logits: torch.Tensor | None,
    availability: torch.Tensor | None,
    input_lengths: torch.Tensor,
    texts: list[str],
    span_targets: torch.Tensor | None = None,
    span_known: torch.Tensor | None = None,
) -> torch.Tensor:
    """Supervise frame-level fingerspelling activity logits via spans or clip-level MIL."""
    if spot_logits is None or availability is None:
        return torch.tensor(0.0, device=input_lengths.device)
    if spot_logits.ndim != 2 or availability.shape != spot_logits.shape:
        raise ValueError("spot_logits and availability must have matching [B,T] shapes")
    if input_lengths.shape != (spot_logits.shape[0],) or len(texts) != spot_logits.shape[0]:
        raise ValueError("input_lengths and texts must match spot_logits batch size")
    losses = []
    for i, text in enumerate(texts):
        length = int(input_lengths[i])
        valid = availability[i, :length].bool()
        if not valid.any():
            continue
        logits_i = spot_logits[i, :length][valid].float()
        if span_targets is not None and span_known is not None and bool(span_known[i]):
            tgt = span_targets[i, :length][valid].float()
            losses.append(F.binary_cross_entropy_with_logits(logits_i, tgt))
        elif text.strip():
            # Weak clip-level positive bag: peak frame should activate, while
            # sparsity regularizes toward localized fingerspelling density.
            peak_loss = F.binary_cross_entropy_with_logits(logits_i.amax().unsqueeze(0), logits_i.new_ones((1,)))
            mean_prob = torch.sigmoid(logits_i).mean()
            target_frac = min(0.65, max(0.10, float(len(text.strip())) / max(float(len(logits_i)), 1.0)))
            losses.append(peak_loss + 0.5 * (mean_prob - target_frac).square())
        else:
            # Explicit non-fingerspelling clip suppresses false character spotting.
            losses.append(0.25 * F.binary_cross_entropy_with_logits(logits_i, torch.zeros_like(logits_i)))
    return torch.stack(losses).mean() if losses else spot_logits.sum() * 0.0


def compute_fusion_entropy_loss(
    weights: torch.Tensor,
    stream_available: torch.Tensor,
    frame_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalized stream-entropy regularizer preventing single-modality collapse."""
    if weights.ndim != 3 or stream_available.shape != weights.shape:
        raise ValueError("weights and stream_available must have matching [B,T,S] shapes")
    avail = stream_available.bool()
    if frame_mask is not None:
        avail = avail & frame_mask.bool()[..., None]
    count = avail.sum(dim=-1)
    multi = count >= 2
    if not multi.any():
        return weights.sum() * 0.0
    w = weights.float().clamp_min(1e-8)
    entropy = -(torch.where(avail, w * w.log(), torch.zeros_like(w))).sum(dim=-1)
    max_ent = torch.log(count.float().clamp_min(2.0))
    norm_ent = entropy / max_ent
    # Normalized loss in [0, 1]; 0 when uniformly balancing available streams.
    return (1.0 - norm_ent[multi]).clamp(0.0, 1.0).mean()


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
