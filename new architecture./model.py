"""RA-MAGC (Reliability-Aware Multi-Rate Articulatory Graph-Conformer) Model.

Complete end-to-end Sign Language Translation architecture connecting the
anatomical graph streams and Conformer backbone to pretrained FLAN-T5-small.
"""

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
    length_mask
)


class RAMAGCVisualEncoder(nn.Module):
    """
    Complete Visual Feature Extractor:
    Anatomical Graphs -> Reliability Gate -> Multi-Scale Temporal CNN ->
    Pairwise Aggregation -> Conformer Backbone -> Fine Detail Retrieval -> Visual Bridge.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # 1. Spatial Anatomical Encoders
        self.hand_encoder = HandGraphEncoder(config.hand)
        self.body_encoder = UpperBodyGraphEncoder(config.body)
        self.face_encoder = FaceSetEncoder(config.face) if config.face.enabled else None
        
        # 2. Region Interaction & Reliability Gate
        max_regions = 4 if config.face.enabled else 3
        self.region_interaction = RegionInteraction(config.fusion, max_regions=max_regions)
        self.reliability_gate = ReliabilityGate(config.fusion, max_regions=max_regions)
        
        # 3. Full-Resolution Multi-Scale Temporal Blocks (operating on length T)
        self.temporal_blocks = nn.ModuleList([
            MultiScaleTemporalBlock(
                in_channels=config.multi_scale.in_channels,
                out_channels=config.multi_scale.out_channels,
                branch_dim=config.multi_scale.branch_dim,
                dropout=config.multi_scale.dropout
            )
            for _ in range(config.multi_scale.num_blocks)
        ])
        
        # 4. Pairwise Aggregation (length T -> ceil(T/2))
        self.pairwise_aggregator = PairwiseAggregator(
            in_channels=config.multi_scale.out_channels,
            out_channels=config.conformer.d_model
        )
        
        # 5. Global Temporal Conformer Backbone with Relative Elapsed-Time Bias
        self.time_bias = RelativeElapsedTimeBias(
            num_heads=config.conformer.num_heads,
            num_buckets=config.conformer.num_time_buckets,
            max_seconds=config.conformer.max_elapsed_seconds
        )
        self.conformer_blocks = nn.ModuleList([
            ConformerBlock(config.conformer)
            for _ in range(config.conformer.num_blocks)
        ])
        
        # 6. Fine Detail Retrieval (Cross-attention from coarse tokens into fine memory Z)
        self.detail_retrieval = FineDetailRetrieval(
            d_model=config.conformer.d_model,
            num_heads=4,
            dropout=config.dropout
        )
        
        # 7. Visual-to-Text Bridge into FLAN-T5-small d_model
        self.bridge = nn.Sequential(
            nn.LayerNorm(config.conformer.d_model),
            nn.Linear(config.conformer.d_model, config.t5_d_model),
            nn.Dropout(config.dropout)
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
        **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Processes multi-stream anatomical inputs into contextualized visual embeddings.
        Returns:
            visual_embeddings: [B, T_coarse, t5_d_model]
            coarse_mask: [B, T_coarse]
        """
        B, T = left_hand_nodes.shape[:2]
        frame_mask = length_mask(lengths, T, left_hand_nodes.device)
        
        # 1. Anatomical Graph Encoding
        h_left = self.hand_encoder(left_hand_nodes, left_hand_edges, left_hand_adj, left_hand_valid, side_idx=0)
        h_right = self.hand_encoder(right_hand_nodes, right_hand_edges, right_hand_adj, right_hand_valid, side_idx=1)
        h_body = self.body_encoder(body_nodes, body_edges, body_adj, body_valid)
        
        region_list = [h_left, h_right, h_body]
        left_val = left_hand_valid.any(dim=-1) & frame_mask
        right_val = right_hand_valid.any(dim=-1) & frame_mask
        body_val = body_valid.any(dim=-1) & frame_mask
        valid_list = [left_val, right_val, body_val]
        
        # Compute quality descriptors for reliability gating (ideas.md Section 3.5):
        # 1. fraction observed, 2. fraction imputed, 3. fraction pos-valid, 4. fraction vel-valid
        q_left = left_hand_nodes[..., 6:10].mean(dim=-2)    # [B, T, 4]
        q_right = right_hand_nodes[..., 6:10].mean(dim=-2)  # [B, T, 4]
        q_body = body_nodes[..., 4:8].mean(dim=-2)          # [B, T, 4]
        quality_list = [q_left, q_right, q_body]
        
        if self.face_encoder is not None and face_nodes is not None and face_valid is not None:
            h_face = self.face_encoder(face_nodes, face_valid)
            face_val = face_valid.any(dim=-1) & frame_mask
            q_face = face_nodes[..., 6:10].mean(dim=-2)
            region_list.append(h_face)
            valid_list.append(face_val)
            quality_list.append(q_face)
            
        region_tokens = torch.stack(region_list, dim=2)        # [B, T, K, 128]
        region_valid_mask = torch.stack(valid_list, dim=-1)     # [B, T, K]
        quality_desc = torch.stack(quality_list, dim=2)         # [B, T, K, 4]
        
        # 2. Region Interaction & Reliability Gated Fusion
        interacted = self.region_interaction(region_tokens, region_valid_mask)
        fused = self.reliability_gate(region_tokens, interacted, quality_desc, region_valid_mask)  # [B, T, 256]
        
        # 3. Stacked Multi-Scale Temporal Blocks (Fine Path at full resolution T)
        fine_memory = fused
        for block in self.temporal_blocks:
            fine_memory = block(fine_memory, frame_mask)
            
        # 4. Pairwise Aggregation (length T -> ceil(T/2))
        coarse_tokens, coarse_lengths, coarse_times = self.pairwise_aggregator(
            fine_memory, lengths, timestamps
        )
        Tc = coarse_tokens.shape[1]
        coarse_mask = length_mask(coarse_lengths, Tc, coarse_tokens.device)
        
        # 5. Conformer Backbone with Relative Elapsed-Time Bias
        time_bias = self.time_bias(coarse_times)  # [B, H, Tc, Tc]
        conformer_states = coarse_tokens
        for c_block in self.conformer_blocks:
            conformer_states = c_block(conformer_states, coarse_mask, time_bias)
            
        # 6. Fine Detail Retrieval (Coarse tokens cross-attend into Fine Memory)
        refined_states = self.detail_retrieval(conformer_states, fine_memory, coarse_mask, frame_mask)
        
        # 7. Bridge into FLAN-T5 d_model
        visual_embeddings = self.bridge(refined_states)
        visual_embeddings = torch.where(coarse_mask[..., None], visual_embeddings, torch.zeros_like(visual_embeddings))
        return visual_embeddings, coarse_mask


class RAMAGCTranslator(nn.Module):
    """
    Complete Sign Language Translation Model:
    RA-MAGC Visual Encoder + Pretrained FLAN-T5-small Encoder-Decoder.
    """
    def __init__(self, visual_encoder: RAMAGCVisualEncoder, t5_model: nn.Module):
        super().__init__()
        self.visual = visual_encoder
        self.t5 = t5_model

    def forward(self, labels: torch.Tensor | None = None, **visual_inputs) -> dict:
        """
        Forward pass with teacher-forced text generation.
        """
        embeddings, mask = self.visual(**visual_inputs)
        output = self.t5(
            inputs_embeds=embeddings,
            attention_mask=mask.long(),
            labels=labels,
            use_cache=False,
            return_dict=True
        )
        return {
            "loss": output.loss,
            "logits": output.logits,
            "visual_mask": mask
        }

    @torch.no_grad()
    def generate(
        self,
        beams: int = 4,
        max_new_tokens: int = 128,
        length_penalty: float = 1.0,
        **visual_inputs
    ) -> torch.Tensor:
        """
        Reference-free autoregressive beam-search generation.
        """
        embeddings, mask = self.visual(**visual_inputs)
        tokens = self.t5.generate(
            inputs_embeds=embeddings,
            attention_mask=mask.long(),
            num_beams=beams,
            max_new_tokens=max_new_tokens,
            length_penalty=length_penalty,
            do_sample=False,
            decoder_start_token_id=self.t5.config.decoder_start_token_id,
            eos_token_id=self.t5.config.eos_token_id,
            pad_token_id=self.t5.config.pad_token_id
        )
        return tokens[:, 1:]  # Exclude initial decoder-start token

