"""RA-MAGC visual encoder and FLAN-T5 translation wrapper."""

from __future__ import annotations

import torch
import torch.nn as nn

from .config import ModelConfig
from .graph_layers import HandGraphEncoder, UpperBodyGraphEncoder, FaceSetEncoder
from .fusion import RegionInteraction, ReliabilityGate
from .temporal import (
    MultiScaleTemporalBlock,
    PairwiseAggregator,
    RelativeElapsedTimeBias,
    ConformerBlock,
    FineDetailRetrieval,
    length_mask,
)


def compute_quality_descriptors(nodes: torch.Tensor, conf_idx: int) -> torch.Tensor:
    """Compute six provenance-aware quality descriptors for each frame.

    The six node channels starting at ``conf_idx`` are confidence, confidence
    availability, observed, imputed, position-valid and velocity-valid.
    """
    if nodes.ndim != 4 or nodes.shape[-1] < conf_idx + 6:
        raise ValueError("nodes must have shape [B,T,N,F] and contain six quality channels")
    conf = nodes[..., conf_idx]
    conf_available = nodes[..., conf_idx + 1].clamp(0.0, 1.0)
    observed = nodes[..., conf_idx + 2].clamp(0.0, 1.0)
    imputed = nodes[..., conf_idx + 3].clamp(0.0, 1.0)
    position_valid = nodes[..., conf_idx + 4].clamp(0.0, 1.0)
    velocity_valid = nodes[..., conf_idx + 5].clamp(0.0, 1.0)

    confidence_count = conf_available.sum(dim=-1).clamp_min(1.0)
    has_confidence = conf_available.sum(dim=-1) > 0
    mean_confidence = (conf * conf_available).sum(dim=-1) / confidence_count
    mean_confidence = torch.where(has_confidence, mean_confidence, torch.zeros_like(mean_confidence))
    return torch.stack(
        [
            mean_confidence,
            conf_available.mean(dim=-1),
            observed.mean(dim=-1),
            imputed.mean(dim=-1),
            position_valid.mean(dim=-1),
            velocity_valid.mean(dim=-1),
        ],
        dim=-1,
    )


class RAMAGCVisualEncoder(nn.Module):
    """Convert structured anatomical landmark sequences into T5-width embeddings."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.hand_encoder = HandGraphEncoder(config.hand)
        self.body_encoder = UpperBodyGraphEncoder(config.body)
        self.face_encoder = FaceSetEncoder(config.face) if config.face.enabled else None

        region_count = 4 if config.face.enabled else 3
        self.region_interaction = RegionInteraction(config.fusion, max_regions=region_count)
        self.reliability_gate = ReliabilityGate(config.fusion, max_regions=region_count)

        self.temporal_blocks = nn.ModuleList(
            [
                MultiScaleTemporalBlock(
                    in_channels=config.multi_scale.in_channels,
                    out_channels=config.multi_scale.out_channels,
                    branch_dim=config.multi_scale.branch_dim,
                    dropout=config.multi_scale.dropout,
                    kernel_sizes=config.multi_scale.kernel_sizes,
                    dilations=config.multi_scale.dilations,
                )
                for _ in range(config.multi_scale.num_blocks)
            ]
        )
        self.pairwise_aggregator = PairwiseAggregator(
            in_channels=config.multi_scale.out_channels,
            out_channels=config.conformer.d_model,
        )
        self.time_bias = RelativeElapsedTimeBias(
            num_heads=config.conformer.num_heads,
            num_buckets=config.conformer.num_time_buckets,
            max_seconds=config.conformer.max_elapsed_seconds,
        )
        self.conformer_blocks = nn.ModuleList(
            [ConformerBlock(config.conformer) for _ in range(config.conformer.num_blocks)]
        )
        self.detail_retrieval = FineDetailRetrieval(
            d_model=config.conformer.d_model,
            num_heads=4,
            local_window_radius=2,
            dropout=config.dropout,
        )
        self.bridge = nn.Sequential(
            nn.LayerNorm(config.conformer.d_model),
            nn.Linear(config.conformer.d_model, config.t5_d_model),
            nn.Dropout(config.dropout),
        )
        self.output_dim = config.t5_d_model

    def forward(
        self,
        left_hand_nodes: torch.Tensor,
        left_hand_edges: torch.Tensor,
        left_hand_adj: torch.Tensor,
        left_hand_valid: torch.Tensor,
        right_hand_nodes: torch.Tensor,
        right_hand_edges: torch.Tensor,
        right_hand_adj: torch.Tensor,
        right_hand_valid: torch.Tensor,
        body_nodes: torch.Tensor,
        body_edges: torch.Tensor,
        body_adj: torch.Tensor,
        body_valid: torch.Tensor,
        timestamps: torch.Tensor,
        lengths: torch.Tensor,
        face_nodes: torch.Tensor | None = None,
        face_valid: torch.Tensor | None = None,
        **unused_batch_fields,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if left_hand_nodes.ndim != 4:
            raise ValueError("left_hand_nodes must have shape [B,T,21,13]")
        batch_size, time_steps = left_hand_nodes.shape[:2]
        if time_steps < 1 or lengths.shape != (batch_size,):
            raise ValueError("Invalid temporal dimensions or lengths shape")
        if torch.any(lengths < 1) or torch.any(lengths > time_steps):
            raise ValueError("Every sequence length must be in [1,T]")
        if timestamps.shape != (batch_size, time_steps):
            raise ValueError("timestamps must have shape [B,T]")

        frame_mask = length_mask(lengths, time_steps, left_hand_nodes.device)

        # Region encoders. Their graph masks prevent invalid nodes from sending
        # messages to valid nodes; masked pooling returns zero for absent regions.
        left = self.hand_encoder(
            left_hand_nodes, left_hand_edges, left_hand_adj, left_hand_valid, side_idx=0
        )
        right = self.hand_encoder(
            right_hand_nodes, right_hand_edges, right_hand_adj, right_hand_valid, side_idx=1
        )
        body = self.body_encoder(body_nodes, body_edges, body_adj, body_valid)

        region_tokens = [left, right, body]
        region_valid = [
            left_hand_valid.bool().any(dim=-1) & frame_mask,
            right_hand_valid.bool().any(dim=-1) & frame_mask,
            body_valid.bool().any(dim=-1) & frame_mask,
        ]
        quality = [
            compute_quality_descriptors(left_hand_nodes, 6),
            compute_quality_descriptors(right_hand_nodes, 6),
            compute_quality_descriptors(body_nodes, 4),
        ]

        if self.face_encoder is not None:
            # Keep the four-stream architecture fixed across batches. Missing face
            # data is represented by zero nodes and an all-false validity mask.
            if face_nodes is None:
                face_nodes = left_hand_nodes.new_zeros((batch_size, time_steps, 18, self.config.face.in_node_dim))
            if face_valid is None:
                face_valid = torch.zeros(
                    (batch_size, time_steps, 18), dtype=torch.bool, device=left_hand_nodes.device
                )
            if face_nodes.shape != (batch_size, time_steps, 18, self.config.face.in_node_dim):
                raise ValueError("face_nodes must have shape [B,T,18,13]")
            if face_valid.shape != (batch_size, time_steps, 18):
                raise ValueError("face_valid must have shape [B,T,18]")
            face = self.face_encoder(face_nodes, face_valid)
            region_tokens.append(face)
            region_valid.append(face_valid.bool().any(dim=-1) & frame_mask)
            quality.append(compute_quality_descriptors(face_nodes, 6))

        tokens = torch.stack(region_tokens, dim=2)
        valid_regions = torch.stack(region_valid, dim=-1)
        quality_features = torch.stack(quality, dim=2)

        interacted = self.region_interaction(tokens, valid_regions)
        fused = self.reliability_gate(tokens, interacted, quality_features, valid_regions)
        # Preserve the learned missing-observation token on real frames, but never
        # allow it to represent batch padding.
        fused = torch.where(frame_mask[..., None], fused, torch.zeros_like(fused))

        fine_memory = fused
        for block in self.temporal_blocks:
            fine_memory = block(fine_memory, frame_mask)

        coarse, coarse_lengths, coarse_elapsed_times = self.pairwise_aggregator(
            fine_memory, lengths, timestamps
        )
        coarse_mask = length_mask(coarse_lengths, coarse.shape[1], coarse.device)
        relative_bias = self.time_bias(coarse_elapsed_times)
        contextual = coarse
        for block in self.conformer_blocks:
            contextual = block(contextual, coarse_mask, relative_bias)

        refined = self.detail_retrieval(contextual, fine_memory, coarse_mask, frame_mask)
        embeddings = self.bridge(refined)
        embeddings = torch.where(coarse_mask[..., None], embeddings, torch.zeros_like(embeddings))
        return embeddings, coarse_mask


class RAMAGCTranslator(nn.Module):
    """RA-MAGC visual encoder coupled to an instantiated T5 encoder-decoder model."""

    def __init__(self, visual_encoder: RAMAGCVisualEncoder, t5_model: nn.Module):
        super().__init__()
        self.visual = visual_encoder
        self.t5 = t5_model
        t5_config = getattr(t5_model, "config", None)
        t5_width = getattr(t5_config, "d_model", None)
        if t5_width is not None and int(t5_width) != visual_encoder.output_dim:
            raise ValueError(
                f"T5 d_model ({t5_width}) does not match visual output_dim ({visual_encoder.output_dim})"
            )

    def forward(
        self,
        labels: torch.Tensor | None = None,
        decoder_attention_mask: torch.Tensor | None = None,
        **visual_inputs,
    ) -> dict:
        embeddings, visual_mask = self.visual(**visual_inputs)
        if labels is not None and decoder_attention_mask is None:
            decoder_attention_mask = labels.ne(-100).long()
        if decoder_attention_mask is not None and labels is not None:
            if decoder_attention_mask.shape != labels.shape:
                raise ValueError("decoder_attention_mask must have the same shape as labels")
        output = self.t5(
            inputs_embeds=embeddings,
            attention_mask=visual_mask.long(),
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return {"loss": output.loss, "logits": output.logits, "visual_mask": visual_mask}

    @torch.no_grad()
    def generate(
        self,
        beams: int = 4,
        max_new_tokens: int = 128,
        length_penalty: float = 1.0,
        **visual_inputs,
    ) -> torch.Tensor:
        if beams < 1 or max_new_tokens < 1:
            raise ValueError("beams and max_new_tokens must be positive")
        embeddings, visual_mask = self.visual(**visual_inputs)
        start_id = self.t5.config.decoder_start_token_id
        tokens = self.t5.generate(
            inputs_embeds=embeddings,
            attention_mask=visual_mask.long(),
            num_beams=beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            do_sample=False,
            decoder_start_token_id=start_id,
            eos_token_id=self.t5.config.eos_token_id,
            pad_token_id=self.t5.config.pad_token_id,
        )
        # Encoder-decoder generate normally returns the decoder-start token first.
        if tokens.ndim == 2 and tokens.shape[1] > 0 and torch.all(tokens[:, 0] == start_id):
            return tokens[:, 1:]
        return tokens
