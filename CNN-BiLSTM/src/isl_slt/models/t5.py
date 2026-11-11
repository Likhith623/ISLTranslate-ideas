"""Visual bridge into the full pretrained T5 encoder-decoder; no reference-dependent prediction API."""

import torch
from torch import nn

from isl_slt.models.encoders import VisualEncoder, masked_mean


class T5Translator(nn.Module):
    def __init__(self, encoder: VisualEncoder, t5, lexical_size: int = 0):
        super().__init__()
        self.visual = encoder
        self.t5 = t5
        self.bridge = nn.Sequential(nn.LayerNorm(encoder.output_dim), nn.Linear(encoder.output_dim, t5.config.d_model))
        self.lexical_head = nn.Linear(encoder.output_dim, lexical_size) if lexical_size else None

    def encode_visual(self, **visual):
        states, mask = self.visual(**visual)
        return states, self.bridge(states), mask

    def forward(self, *, labels: torch.Tensor, **visual):
        states, embeddings, mask = self.encode_visual(**visual)
        # Frozen T5 must remain in autograd: gradients flow through its operations into bridge/visual.
        output = self.t5(inputs_embeds=embeddings, attention_mask=mask.long(), labels=labels,
                         use_cache=False, return_dict=True)
        lexical = self.lexical_head(masked_mean(states, mask)) if self.lexical_head is not None else None
        return {"logits": output.logits, "lexical_logits": lexical}

    @torch.no_grad()
    def generate(self, *, beams: int, max_new_tokens: int, length_penalty: float, **visual):
        _, embeddings, mask = self.encode_visual(**visual)
        tokens = self.t5.generate(inputs_embeds=embeddings, attention_mask=mask.long(), num_beams=beams,
                                  max_new_tokens=max_new_tokens, length_penalty=length_penalty, do_sample=False,
                                  decoder_start_token_id=self.t5.config.decoder_start_token_id,
                                  eos_token_id=self.t5.config.eos_token_id, pad_token_id=self.t5.config.pad_token_id)
        return tokens[:, 1:]  # Exclude T5's decoder-start PAD, retain generated EOS for truncation checks.
