"""
Decoder Module for Continuous Sign Language Translation.

Interfaces prompt representations from the Articulatory Q-Former with
pre-trained sequence-to-sequence models (e.g. FLAN-T5) or causal decoders.
Enforces rigorous teacher-forcing target shifting and causal attention masking.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


if _HAS_TORCH:
    class TranslationDecoderAdapter(nn.Module):
        """
        Decoder wrapper supporting encoder-decoder models (like T5) where
        visual Q-Former tokens serve as the encoder output for cross-attention.
        """
        def __init__(
            self,
            pretrained_model_name: str = "google/flan-t5-base",
            d_model: int = 768,
            vocab_size: int = 32128,
            max_target_len: int = 128,
            label_smoothing: float = 0.1,
            pad_token_id: int = 0,
            decoder_start_token_id: int = 0,
        ):
            super().__init__()
            self.pretrained_model_name = pretrained_model_name
            self.d_model = d_model
            self.vocab_size = vocab_size
            self.max_target_len = max_target_len
            self.label_smoothing = label_smoothing
            self.pad_token_id = pad_token_id
            self.decoder_start_token_id = decoder_start_token_id

            self.hf_model = None
            try:
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                self.hf_model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name)
                # Freeze lower layers if needed, or leave trainable
            except Exception:
                # Standalone lightweight TransformerDecoder fallback when weights unavailable
                self.hf_model = None

            if self.hf_model is None:
                # Fallback PyTorch standard TransformerDecoder
                decoder_layer = nn.TransformerDecoderLayer(
                    d_model=d_model,
                    nhead=8,
                    dim_feedforward=d_model * 4,
                    dropout=0.1,
                    batch_first=True,
                )
                self.fallback_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
                self.token_embeddings = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
                self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
            else:
                self.fallback_decoder = None
                self.token_embeddings = None
                self.lm_head = None

        def shift_tokens_right(self, input_ids: torch.Tensor) -> torch.Tensor:
            """Shift input ids one token to the right and prepend decoder_start_token_id."""
            shifted = input_ids.new_zeros(input_ids.shape)
            shifted[:, 1:] = input_ids[:, :-1].clone()
            shifted[:, 0] = self.decoder_start_token_id
            # Replace possible -100 values with pad_token_id
            shifted.masked_fill_(shifted == -100, self.pad_token_id)
            return shifted

        def forward(
            self,
            encoder_hidden_states: torch.Tensor,
            labels: Optional[torch.Tensor] = None,
            decoder_input_ids: Optional[torch.Tensor] = None,
        ) -> Dict[str, torch.Tensor]:
            """
            Args:
                encoder_hidden_states: (B, M, D) from Articulatory Q-Former.
                labels: (B, L) target token IDs (with -100 for ignored positions).
                decoder_input_ids: Optional (B, L) shifted decoder inputs.
            Returns:
                dict containing 'loss' (if labels provided) and 'logits'.
            """
            B, M, D = encoder_hidden_states.shape

            if labels is not None and decoder_input_ids is None:
                decoder_input_ids = self.shift_tokens_right(labels)

            if self.hf_model is not None:
                # Use HuggingFace Seq2Seq model cross-attention directly
                # Pass encoder_hidden_states as the encoder outputs
                encoder_outputs = (encoder_hidden_states,)
                outputs = self.hf_model(
                    decoder_input_ids=decoder_input_ids,
                    encoder_outputs=encoder_outputs,
                    labels=labels,
                    return_dict=True,
                )
                return {
                    "loss": outputs.loss,
                    "logits": outputs.logits,
                }
            else:
                # Fallback TransformerDecoder execution
                L = decoder_input_ids.shape[1] if decoder_input_ids is not None else 1
                dec_embeds = self.token_embeddings(decoder_input_ids)  # (B, L, D)
                
                # Causal self-attention mask for decoder
                causal_mask = nn.Transformer.generate_square_subsequent_mask(
                    L, device=encoder_hidden_states.device
                )
                
                tgt_key_padding_mask = (decoder_input_ids == self.pad_token_id)
                
                hidden = self.fallback_decoder(
                    tgt=dec_embeds,
                    memory=encoder_hidden_states,
                    tgt_mask=causal_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                )
                logits = self.lm_head(hidden)  # (B, L, vocab_size)
                
                loss = None
                if labels is not None:
                    loss = F.cross_entropy(
                        logits.view(-1, self.vocab_size),
                        labels.view(-1),
                        ignore_index=-100,
                        label_smoothing=self.label_smoothing,
                    )
                return {"loss": loss, "logits": logits}

        @torch.no_grad()
        def generate(
            self,
            encoder_hidden_states: torch.Tensor,
            max_length: int = 64,
            beam_size: int = 1,
        ) -> torch.Tensor:
            """
            Autoregressive text generation.
            
            Args:
                encoder_hidden_states: (B, M, D)
                max_length: Maximum target sequence length.
                beam_size: Number of beams (1 for greedy).
            Returns:
                generated_ids: (B, seq_len)
            """
            B, M, D = encoder_hidden_states.shape
            device = encoder_hidden_states.device

            if self.hf_model is not None:
                encoder_outputs = (encoder_hidden_states,)
                return self.hf_model.generate(
                    encoder_outputs=encoder_outputs,
                    max_length=max_length,
                    num_beams=beam_size,
                )
            else:
                # Fallback greedy generation
                generated = torch.full(
                    (B, 1),
                    self.decoder_start_token_id,
                    dtype=torch.long,
                    device=device,
                )
                for _ in range(max_length - 1):
                    dec_embeds = self.token_embeddings(generated)
                    L = generated.shape[1]
                    causal_mask = nn.Transformer.generate_square_subsequent_mask(L, device=device)
                    hidden = self.fallback_decoder(
                        tgt=dec_embeds,
                        memory=encoder_hidden_states,
                        tgt_mask=causal_mask,
                    )
                    next_token_logits = self.lm_head(hidden[:, -1, :])  # (B, vocab_size)
                    next_tokens = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (B, 1)
                    generated = torch.cat([generated, next_tokens], dim=1)
                    
                    # Stop if all sequences generated pad/eos
                    if (next_tokens == self.pad_token_id).all():
                        break
                        
                return generated
else:
    class TranslationDecoderAdapter:  # type: ignore
        def __init__(self, *args, **kwargs): pass

