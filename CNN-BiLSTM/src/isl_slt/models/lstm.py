"""Complete additive-attention LSTM decoder with training teacher forcing and free beam generation."""

import torch
from torch import nn

from isl_slt.models.encoders import VisualEncoder, masked_mean


class AttentionLSTM(nn.Module):
    def __init__(self, encoder: VisualEncoder, vocabulary: int, hidden: int, embedding: int,
                 dropout: float, lexical_size: int = 0):
        super().__init__()
        self.visual = encoder
        self.embedding = nn.Embedding(vocabulary, embedding, padding_idx=0)
        self.initial = nn.Linear(encoder.output_dim, hidden)
        self.key = nn.Linear(encoder.output_dim, hidden, bias=False)
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.energy = nn.Linear(hidden, 1, bias=False)
        self.cell = nn.LSTMCell(embedding + encoder.output_dim, hidden)
        self.output = nn.Linear(hidden + encoder.output_dim, vocabulary)
        self.dropout = nn.Dropout(dropout)
        self.lexical_head = nn.Linear(encoder.output_dim, lexical_size) if lexical_size else None

    def step(self, token, state, memory, keys, mask):
        hidden, cell = state
        scores = self.energy(torch.tanh(keys + self.query(hidden)[:, None])).squeeze(-1).float()
        weights = torch.softmax(scores.masked_fill(~mask, -torch.inf), dim=-1).to(memory.dtype)
        context = torch.bmm(weights[:, None], memory).squeeze(1)
        # Recurrent arithmetic stays FP32; projections can still use outer autocast.
        with torch.autocast(memory.device.type, enabled=False):
            hidden, cell = self.cell(torch.cat([self.dropout(self.embedding(token)), context], -1).float(),
                                     (hidden.float(), cell.float()))
        logits = self.output(torch.cat([self.dropout(hidden), context], -1))
        return logits, (hidden, cell)

    def initialize(self, memory, mask):
        hidden = torch.tanh(self.initial(masked_mean(memory, mask)))
        return hidden.float(), torch.zeros_like(hidden).float()

    def forward(self, *, labels: torch.Tensor, **visual):
        memory, mask = self.visual(**visual)
        keys, state = self.key(memory), self.initialize(memory, mask)
        previous = torch.full((len(memory),), 2, dtype=torch.long, device=memory.device)
        logits = []
        for index in range(labels.shape[1]):
            output, state = self.step(previous, state, memory, keys, mask)
            logits.append(output)
            previous = labels[:, index].masked_fill(labels[:, index] == -100, 0)
        lexical = self.lexical_head(masked_mean(memory, mask)) if self.lexical_head is not None else None
        return {"logits": torch.stack(logits, 1), "lexical_logits": lexical}

    @torch.no_grad()
    def generate(self, *, beams: int, max_new_tokens: int, length_penalty: float, **visual):
        memory, mask = self.visual(**visual)
        predictions = []
        for index in range(len(memory)):
            mem, valid = memory[index:index + 1], mask[index:index + 1]
            keys = self.key(mem)
            # Each candidate keeps its own recurrent state; completed EOS candidates remain eligible.
            candidates = [([], 0.0, self.initialize(mem, valid), False)]
            for _ in range(max_new_tokens):
                expanded = []
                for tokens, score, state, complete in candidates:
                    if complete:
                        expanded.append((tokens, score, state, True))
                        continue
                    previous = torch.tensor([tokens[-1] if tokens else 2], device=memory.device)
                    logits, next_state = self.step(previous, state, mem, keys, valid)
                    log_probs = logits.float().log_softmax(-1)[0]
                    log_probs[0], log_probs[2] = -torch.inf, -torch.inf
                    values, indices = log_probs.topk(min(beams, len(log_probs) - 2))
                    for value, token in zip(values.tolist(), indices.tolist()):
                        expanded.append((tokens + [token], score + value, next_state, token == 3))
                candidates = sorted(expanded, key=lambda c: c[1] / max(len(c[0]), 1) ** length_penalty,
                                    reverse=True)[:beams]
                if all(candidate[3] for candidate in candidates):
                    break
            predictions.append(candidates[0][0])
        output = torch.zeros(len(predictions), max(map(len, predictions)), dtype=torch.long, device=memory.device)
        for index, prediction in enumerate(predictions):
            output[index, :len(prediction)] = torch.tensor(prediction, device=memory.device)
        return output
