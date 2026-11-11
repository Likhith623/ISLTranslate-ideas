"""Dedicated character-level fingerspelling expert with CTC supervision."""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


CHARACTERS = "abcdefghijklmnopqrstuvwxyz0123456789-' "
BLANK_ID = 0


def encode_characters(text: str) -> list[int]:
    lookup = {ch: i + 1 for i, ch in enumerate(CHARACTERS)}
    normalized = text.lower()
    unknown = sorted(set(normalized) - set(lookup))
    if unknown:
        raise ValueError(f"Unsupported fingerspelling character(s): {unknown}")
    return [lookup[ch] for ch in normalized]


def decode_characters(ids: list[int]) -> str:
    chars = ["<blank>"] + list(CHARACTERS)
    return "".join(chars[i] for i in ids if 0 < i < len(chars))


def greedy_ctc_decode(logits: torch.Tensor, lengths: torch.Tensor) -> list[str]:
    if logits.ndim != 3 or lengths.shape != (logits.shape[0],):
        raise ValueError("logits must be [B,T,C] and lengths [B]")
    if torch.any(lengths < 0) or torch.any(lengths > logits.shape[1]):
        raise ValueError("CTC decode lengths must be within [0,T]")
    paths = logits.argmax(-1)
    outputs = []
    for row, n in zip(paths, lengths.tolist()):
        collapsed, previous = [], None
        for token in row[:n].tolist():
            if token != 0 and token != previous:
                collapsed.append(token - 1)
            previous = token
        outputs.append("".join(CHARACTERS[i] for i in collapsed if 0 <= i < len(CHARACTERS)))
    return outputs


class FingerspellingExpert(nn.Module):
    """BiGRU/CTC specialist over paired hand appearance and pose embeddings.

    The character posterior is fed back as a softly gated feature. The gate is
    initialized near closed so an untrained specialist does not dominate SLT.
    Train CTC only on explicitly annotated fingerspelling segments.
    """

    def __init__(self, visual_dim: int, pose_dim: int, hidden_dim: int = 256, layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.input = nn.Sequential(nn.Linear(2 * visual_dim + 2 * pose_dim, hidden_dim),
                                   nn.LayerNorm(hidden_dim), nn.GELU())
        self.rnn = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=layers, batch_first=True,
                          bidirectional=True, dropout=dropout if layers > 1 else 0.0)
        self.ctc = nn.Linear(hidden_dim, len(CHARACTERS) + 1)
        self.char_embedding = nn.Parameter(torch.randn(len(CHARACTERS) + 1, hidden_dim) * 0.02)
        self.feedback_gate = nn.Linear(hidden_dim * 2, 1)
        nn.init.constant_(self.feedback_gate.bias, -3.5)
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_dim = hidden_dim

    def forward(self, left_rgb, right_rgb, left_pose, right_pose, lengths, left_valid, right_valid):
        if not (left_rgb.shape[:2] == right_rgb.shape[:2] == left_pose.shape[:2] == right_pose.shape[:2]):
            raise ValueError("Fingerspelling streams must share batch and time dimensions")
        b, t = left_rgb.shape[:2]
        if lengths.shape != (b,) or torch.any(lengths < 1) or torch.any(lengths > t):
            raise ValueError("lengths must contain one valid value in [1,T] per sample")
        if left_valid.shape != (b, t) or right_valid.shape != (b, t):
            raise ValueError("left_valid/right_valid must be [B,T]")
        x = self.input(torch.cat([left_rgb, right_rgb, left_pose, right_pose], dim=-1))
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.rnn(packed)
        hidden, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=t)
        logits = self.ctc(hidden)
        posterior = torch.softmax(logits.float(), dim=-1).to(hidden.dtype)
        char_context = posterior @ self.char_embedding
        gate = torch.sigmoid(self.feedback_gate(torch.cat([hidden, char_context], dim=-1)))
        # A one-handed spelling can still be informative; absent-hand features
        # are zeroed upstream and must not invalidate the visible hand stream.
        available = left_valid.bool() | right_valid.bool()
        fused = self.norm(hidden + gate * char_context)
        fused = torch.where(available[..., None], fused, torch.zeros_like(fused))
        logits = torch.where(available[..., None], logits, torch.zeros_like(logits))
        return {"features": fused, "logits": logits, "available": available, "gate": gate.squeeze(-1)}


def ctc_loss(logits: torch.Tensor, lengths: torch.Tensor, targets: torch.Tensor,
             target_lengths: torch.Tensor) -> torch.Tensor:
    """CTC loss with explicit concatenated targets and per-sequence lengths."""
    if targets.ndim != 1 or target_lengths.shape != lengths.shape:
        raise ValueError("targets must be concatenated [sum(target_lengths)] and lengths [B]")
    if int(target_lengths.sum()) != targets.numel():
        raise ValueError("sum(target_lengths) must equal targets.numel()")
    if torch.any(lengths < 1) or torch.any(lengths > logits.shape[1]):
        raise ValueError("CTC input lengths must be in [1,T]")
    if torch.any(target_lengths < 0) or torch.any(target_lengths > lengths):
        raise ValueError("CTC target length cannot exceed input sequence length")
    log_probs = logits.float().log_softmax(-1).transpose(0, 1)
    return F.ctc_loss(log_probs, targets.long(), lengths.long(), target_lengths.long(),
                      blank=BLANK_ID, reduction="mean", zero_infinity=True)
