"""Pretrained T5/FLAN-T5 bridge accepting visual and latent-plan embeddings."""

from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class T5VisualDecoder(nn.Module):
    """Adapter around Hugging Face T5ForConditionalGeneration.

    The full T5 encoder-decoder is retained. Visual frame embeddings and latent
    plan embeddings are supplied to T5's encoder through ``inputs_embeds``.
    """

    def __init__(self, model: nn.Module, visual_dim: int, freeze: bool = False):
        super().__init__()
        self.model = model
        self.t5_dim = int(model.config.d_model)
        self.bridge = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, self.t5_dim))
        self.plan_bridge = nn.Sequential(nn.LayerNorm(visual_dim), nn.Linear(visual_dim, self.t5_dim))
        if freeze:
            for p in self.model.parameters():
                p.requires_grad_(False)

    def encode_inputs(self, visual: torch.Tensor, visual_mask: torch.Tensor,
                      plan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        v = self.bridge(visual)
        p = self.plan_bridge(plan)
        embeddings = torch.cat([v, p], dim=1)
        plan_mask = torch.ones(plan.shape[:2], dtype=visual_mask.dtype, device=visual_mask.device)
        mask = torch.cat([visual_mask.to(plan_mask.dtype), plan_mask], dim=1)
        return embeddings, mask

    def forward(self, visual: torch.Tensor, visual_mask: torch.Tensor, plan: torch.Tensor,
                labels: torch.Tensor, semantic_ot: bool = False, ot_epsilon: float = 0.08,
                ot_iterations: int = 30, label_smoothing: float = 0.0) -> dict:
        if labels.ndim != 2 or labels.shape[0] != visual.shape[0]:
            raise ValueError("labels must have shape [B,L] matching visual batch size")
        if not 0.0 <= label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0,1)")
        embeddings, mask = self.encode_inputs(visual, visual_mask, plan)
        if label_smoothing == 0.0:
            output = self.model(inputs_embeds=embeddings, attention_mask=mask.long(), labels=labels,
                                use_cache=False, return_dict=True)
            loss = output.loss
        else:
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=labels)
            # T5 commonly uses pad_token_id as decoder_start_token_id. The first
            # shifted token is therefore numerically equal to PAD but is a real
            # start symbol and must remain visible to decoder self-attention.
            decoder_mask = torch.cat([
                torch.ones((labels.shape[0], 1), dtype=torch.long, device=labels.device),
                labels[:, :-1].ne(-100).long(),
            ], dim=1)
            output = self.model(inputs_embeds=embeddings, attention_mask=mask.long(),
                                decoder_input_ids=decoder_input_ids,
                                decoder_attention_mask=decoder_mask, use_cache=False, return_dict=True)
            log_probs = F.log_softmax(output.logits.float(), dim=-1)
            valid = labels.ne(-100)
            safe_labels = labels.masked_fill(~valid, 0)
            nll = -log_probs.gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
            smooth = -log_probs.mean(dim=-1)
            per_token = (1.0 - label_smoothing) * nll + label_smoothing * smooth
            loss = (per_token * valid).sum() / valid.sum().clamp_min(1)
        result = {"loss": loss, "logits": output.logits,
                  "encoder_last_hidden_state": output.encoder_last_hidden_state}
        if semantic_ot:
            text_ids = labels.masked_fill(labels < 0, self.model.config.pad_token_id)
            text_mask = text_ids.ne(self.model.config.pad_token_id)
            text_output = self.model.encoder(input_ids=text_ids, attention_mask=text_mask.long(),
                                             return_dict=True)
            result["semantic_ot_loss"] = entropic_semantic_ot(
                output.encoder_last_hidden_state[:, :visual.shape[1]], visual_mask,
                text_output.last_hidden_state, text_mask, epsilon=ot_epsilon, iterations=ot_iterations,
            )
        return result

    @torch.no_grad()
    def generate(self, visual: torch.Tensor, visual_mask: torch.Tensor, plan: torch.Tensor,
                 num_beams: int = 4, max_new_tokens: int = 256, length_penalty: float = 1.0):
        embeddings, mask = self.encode_inputs(visual, visual_mask, plan)
        # Supplying explicit encoder_outputs is more portable across Transformers
        # versions than relying on GenerationMixin's inputs_embeds handling.
        encoder_outputs = self.model.encoder(
            inputs_embeds=embeddings, attention_mask=mask.long(), return_dict=True
        )
        return self.model.generate(encoder_outputs=encoder_outputs, attention_mask=mask.long(),
                                   num_beams=num_beams, max_new_tokens=max_new_tokens,
                                   length_penalty=length_penalty, do_sample=False)


def entropic_semantic_ot(x: torch.Tensor, x_mask: torch.Tensor, y: torch.Tensor, y_mask: torch.Tensor,
                         epsilon: float = 0.08, iterations: int = 30) -> torch.Tensor:
    """Unconstrained entropic OT over visual/time and target-token embeddings.

    This is an optional regularizer, not a claim that sign and English word
    order are monotonic or that frame-to-word alignments are ground truth.
    """
    if x.ndim != 3 or y.ndim != 3 or x.shape[0] != y.shape[0]:
        raise ValueError("x and y must be [B,T,D] and [B,L,D]")
    if x_mask.shape != x.shape[:2] or y_mask.shape != y.shape[:2]:
        raise ValueError("OT masks must match their sequence shapes")
    if epsilon <= 0 or iterations <= 0:
        raise ValueError("epsilon and iterations must be positive")
    losses = []
    for i in range(x.shape[0]):
        a = x[i, x_mask[i].bool()].float()
        b = y[i, y_mask[i].bool()].float()
        if not len(a) or not len(b):
            continue
        a = F.normalize(a, dim=-1)
        b = F.normalize(b, dim=-1)
        cost = (1.0 - a @ b.T).clamp_min(0)
        n, m = cost.shape
        log_mu = cost.new_full((n,), -math.log(float(n)))
        log_nu = cost.new_full((m,), -math.log(float(m)))
        log_k = -cost / epsilon
        u = torch.zeros_like(log_mu)
        v = torch.zeros_like(log_nu)
        for _ in range(iterations):
            u = log_mu - torch.logsumexp(log_k + v[None, :], dim=1)
            v = log_nu - torch.logsumexp(log_k + u[:, None], dim=0)
        log_plan = log_k + u[:, None] + v[None, :]
        plan = torch.exp(log_plan)
        # Full KL-regularized transport objective (not merely the raw cost of
        # the smoothed coupling): <P,C> + eps * KL(P || mu tensor nu).
        log_reference = log_mu[:, None] + log_nu[None, :]
        kl = (plan * (log_plan - log_reference)).sum()
        losses.append((plan * cost).sum() + epsilon * kl)
    return torch.stack(losses).mean() if losses else x.sum() * 0.0
