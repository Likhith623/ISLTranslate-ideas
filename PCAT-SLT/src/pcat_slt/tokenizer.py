"""Lossless UTF-8 byte targets: fixed vocabulary, no pretrained downloads."""
import torch


class ByteTokenizer:
    pad_id, bos_id, eos_id, vocab_size = 0, 1, 2, 259

    def encode(self, text: str) -> list[int]:
        return [self.bos_id] + [b + 3 for b in text.encode("utf-8")] + [self.eos_id]

    def decode(self, ids, errors="replace") -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.detach().cpu().tolist()
        result = []
        for token in ids:
            if token == self.eos_id:
                break
            if 3 <= token < self.vocab_size:
                result.append(token - 3)
        return bytes(result).decode("utf-8", errors=errors)


class Utf8Constraint:
    """Finite-state UTF-8 validity with budget-aware completion (no surrogates).

    State encodes remaining continuation bytes plus the special E0/ED/F0/F4
    bounds that exclude overlong encodings and out-of-range code points.
    """

    def __init__(self, batch_size, device):
        self.transitions = torch.full((8, ByteTokenizer.vocab_size), -1, dtype=torch.long, device=device)
        table = self.transitions
        table[0, ByteTokenizer.eos_id] = 0
        table[0, 3:131] = 0                 # 00..7F
        table[0, 0xC2 + 3:0xDF + 4] = 1
        table[0, 0xE1 + 3:0xEC + 4] = 2
        table[0, 0xEE + 3:0xEF + 4] = 2
        table[0, 0xF1 + 3:0xF3 + 4] = 3
        for lead, state in ((0xE0, 4), (0xED, 5), (0xF0, 6), (0xF4, 7)):
            table[0, lead + 3] = state
        for state in (1, 2, 3):
            table[state, 0x80 + 3:0xBF + 4] = state - 1
        table[4, 0xA0 + 3:0xBF + 4] = 1
        table[5, 0x80 + 3:0x9F + 4] = 1
        table[6, 0x90 + 3:0xBF + 4] = 2
        table[7, 0x80 + 3:0x8F + 4] = 2
        self.remaining = torch.tensor([0, 1, 2, 3, 2, 2, 3, 3], device=device)
        self.state = torch.zeros(batch_size, dtype=torch.long, device=device)

    def mask(self, logits, slots):
        if slots < 1:
            raise ValueError("at least one output slot required")
        next_state = self.transitions[self.state]
        allowed = (next_state >= 0) & (self.remaining[next_state.clamp_min(0)] < slots)
        return logits.masked_fill(~allowed, -torch.inf)

    def advance(self, tokens, finished):
        state = self.transitions[self.state, tokens]
        if ((state < 0) & ~finished).any():
            raise ValueError("invalid UTF-8 transition")
        self.state = torch.where(finished, 0, state)
