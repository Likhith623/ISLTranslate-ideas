"""
Articulatory-Bound Spatiotemporal Transformer (AB-ST).

Unified master model class integrating:
1. Dynamic Torso-Midline Normalization
2. Utterance-Level Kinematic Dominance Routing
3. Deformable Keypoint-Guided Patch Binding
4. Factored 4-Articulator Aggregation
5. Axial Spatio-Temporal Conformer
6. Multi-Task Fingerspelling CTC Branch
7. Articulatory Q-Former Resampler
8. Sequence-to-Sequence Translation Decoder
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

from .config import ModelConfig
from .dominance import KinematicDominanceRouter, TorsoMidlineNormalizer
from .vision import DINOv2SpatialExtractor
from .binding import DeformablePatchBinder
from .articulators import FactoredArticulatorGrouper
from .temporal import AxialSpatioTemporalConformer
from .resampler import ArticulatoryQFormer
from .decoder import TranslationDecoderAdapter


if _HAS_TORCH:
    class ArticulatoryTransformer(nn.Module):
        """
        Complete end-to-end Articulatory-Bound Spatiotemporal Transformer for ISLT.
        """
        def __init__(self, config: Optional[ModelConfig] = None):
            super().__init__()
            self.config = config or ModelConfig()
            cfg = self.config

            # 1. Torso-Midline Normalization
            self.torso_norm = TorsoMidlineNormalizer(
                left_shoulder_idx=cfg.pose.left_shoulder_idx,
                right_shoulder_idx=cfg.pose.right_shoulder_idx,
                ema_momentum=cfg.pose.ema_momentum,
                eps=cfg.pose.eps,
            )

            # 2. Kinematic Hand Dominance Router
            self.dominance_router = KinematicDominanceRouter(
                embed_dim=cfg.binding.hidden_dim,
                left_hand_slice=cfg.pose.left_hand_slice,
                right_hand_slice=cfg.pose.right_hand_slice,
                use_role_embeddings=cfg.pose.use_role_embeddings,
            )

            # 3. Vision Backbone (optional if using cached patch features)
            self.vision_extractor = DINOv2SpatialExtractor(
                backbone_name=cfg.vision.backbone_name,
                pretrained=cfg.vision.pretrained,
                freeze=cfg.vision.freeze_backbone,
                image_size=cfg.vision.image_size,
                patch_size=cfg.vision.patch_size,
            )

            # 4. Deformable Keypoint-to-Patch Binder
            self.binder = DeformablePatchBinder(
                visual_dim=cfg.binding.visual_dim,
                pose_dim=cfg.binding.pose_dim,
                hidden_dim=cfg.binding.hidden_dim,
                num_joints=cfg.pose.num_joints,
                max_offset=cfg.binding.max_offset,
                dropout=cfg.binding.dropout,
                modality_dropout_prob=cfg.loss.modality_dropout_prob,
            )

            # 5. Factored 4-Articulator Grouper
            self.articulator_grouper = FactoredArticulatorGrouper(
                embed_dim=cfg.articulator.embed_dim,
                aggregation_heads=cfg.articulator.aggregation_heads,
                dropout=cfg.articulator.dropout,
                left_hand_slice=cfg.pose.left_hand_slice,
                right_hand_slice=cfg.pose.right_hand_slice,
                body_slice=cfg.pose.body_slice,
                face_slice=cfg.pose.face_slice,
            )

            # 6. Axial Spatio-Temporal Conformer with CTC head
            self.temporal_conformer = AxialSpatioTemporalConformer(
                num_layers=cfg.temporal.num_layers,
                dim=cfg.temporal.hidden_dim,
                num_spatial_heads=cfg.temporal.num_spatial_heads,
                num_temporal_heads=cfg.temporal.num_temporal_heads,
                conv_kernel_size=cfg.temporal.conv_kernel_size,
                ffn_expansion_factor=cfg.temporal.ffn_expansion_factor,
                dropout=cfg.temporal.dropout,
                enable_ctc=cfg.temporal.enable_fingerspelling_ctc,
                ctc_vocab_size=cfg.temporal.ctc_vocab_size,
            )

            # 7. Articulatory Q-Former Resampler
            self.resampler = ArticulatoryQFormer(
                num_queries=cfg.resampler.num_queries,
                input_dim=cfg.temporal.hidden_dim,
                query_dim=cfg.resampler.hidden_dim,
                output_dim=cfg.resampler.output_dim,
                num_layers=cfg.resampler.num_layers,
                num_heads=cfg.resampler.num_heads,
                dropout=cfg.resampler.dropout,
            )

            # 8. Translation Decoder
            self.decoder = TranslationDecoderAdapter(
                pretrained_model_name=cfg.decoder.pretrained_model_name,
                d_model=cfg.decoder.d_model,
                vocab_size=cfg.data.vocab_size,
                max_target_len=cfg.decoder.max_target_len,
                label_smoothing=cfg.decoder.label_smoothing,
            )

        def encode(
            self,
            landmarks_66: torch.Tensor,
            visual_input: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
            landmarks_raw_xy: Optional[torch.Tensor] = None,
        ) -> Dict[str, Any]:
            """
            Forward pass through all encoder stages.
            
            Args:
                landmarks_66: (B, T, 66, 3) raw or pre-normalized coordinates.
                visual_input: Either (B, T, 3, H, W) raw frames OR
                              (B, T, Hg, Wg, C) pre-extracted DINOv2 patches.
                temporal_mask: Optional (B, T) boolean mask (True=valid frame).
                landmarks_raw_xy: Optional (B, T, 66, 2) unnormalized image coordinates.
            Returns:
                dict containing:
                    - 'prompt_tokens': (B, num_queries, d_model)
                    - 'ctc_logits': (T, B, vocab_size) or None
                    - 'articulators': (B, T, 4, D)
                    - 'offsets': (B, T, 66, 2)
            """
            # Extract patches if raw frames are passed
            if visual_input.dim() == 5 and visual_input.shape[2] == 3:
                patch_grid = self.vision_extractor(visual_input)
            else:
                patch_grid = visual_input

            # Stage 1: Torso-Midline Normalization
            norm_landmarks, _ = self.torso_norm(landmarks_66)

            # Stage 2: Kinematic Dominance Decision
            e_left, e_right = self.dominance_router.compute_energy(norm_landmarks)
            is_right_dominant = (e_right >= e_left)

            # Stage 3: Deformable Patch-Keypoint Binding
            joint_tokens, offsets = self.binder(
                patch_grid=patch_grid,
                landmarks_norm=norm_landmarks,
                landmarks_raw_xy=landmarks_raw_xy,
            )  # (B, T, 66, D)

            # Stage 4: Factored 4-Articulator Grouping
            articulators = self.articulator_grouper(
                joint_tokens=joint_tokens,
                is_right_dominant=is_right_dominant,
            )  # (B, T, 4, D)

            # Stage 5: Axial Spatio-Temporal Conformer
            conformer_out, ctc_logits = self.temporal_conformer(
                x=articulators,
                temporal_mask=temporal_mask,
            )  # (B, T, 4, D), (T, B, V_ctc)

            # Stage 6: Articulatory Q-Former Resampler
            prompt_tokens = self.resampler(
                articulatory_tokens=conformer_out,
                temporal_mask=temporal_mask,
            )  # (B, num_queries, d_model)

            return {
                "prompt_tokens": prompt_tokens,
                "ctc_logits": ctc_logits,
                "articulators": conformer_out,
                "offsets": offsets,
                "is_right_dominant": is_right_dominant,
            }

        def forward(
            self,
            landmarks_66: torch.Tensor,
            visual_input: torch.Tensor,
            target_ids: Optional[torch.Tensor] = None,
            temporal_mask: Optional[torch.Tensor] = None,
            ctc_targets: Optional[torch.Tensor] = None,
            ctc_target_lengths: Optional[torch.Tensor] = None,
            ctc_input_lengths: Optional[torch.Tensor] = None,
        ) -> Dict[str, Any]:
            """
            Full end-to-end forward step for training and loss calculation.
            """
            encoded = self.encode(
                landmarks_66=landmarks_66,
                visual_input=visual_input,
                temporal_mask=temporal_mask,
            )

            prompt_tokens = encoded["prompt_tokens"]
            decoder_outputs = self.decoder(
                encoder_hidden_states=prompt_tokens,
                labels=target_ids,
            )

            loss_dict = {
                "translation_loss": decoder_outputs.get("loss"),
                "logits": decoder_outputs.get("logits"),
                "ctc_logits": encoded["ctc_logits"],
                "offsets": encoded["offsets"],
            }

            # Optional CTC loss computation if targets provided
            if (
                encoded["ctc_logits"] is not None
                and ctc_targets is not None
                and ctc_target_lengths is not None
                and ctc_input_lengths is not None
            ):
                ctc_log_probs = F.log_softmax(encoded["ctc_logits"], dim=-1)
                ctc_loss = F.ctc_loss(
                    ctc_log_probs,
                    ctc_targets,
                    ctc_input_lengths,
                    ctc_target_lengths,
                    blank=0,
                    zero_infinity=True,
                )
                loss_dict["ctc_loss"] = ctc_loss

            return loss_dict

        @torch.no_grad()
        def generate(
            self,
            landmarks_66: torch.Tensor,
            visual_input: torch.Tensor,
            temporal_mask: Optional[torch.Tensor] = None,
            max_length: int = 64,
            beam_size: int = 1,
        ) -> torch.Tensor:
            """
            Autoregressive translation inference.
            """
            encoded = self.encode(
                landmarks_66=landmarks_66,
                visual_input=visual_input,
                temporal_mask=temporal_mask,
            )
            return self.decoder.generate(
                encoder_hidden_states=encoded["prompt_tokens"],
                max_length=max_length,
                beam_size=beam_size,
            )

        def count_parameters(self) -> Dict[str, int]:
            """Return parameter breakdown."""
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            return {"total": total, "trainable": trainable, "frozen": total - trainable}
else:
    class ArticulatoryTransformer:  # type: ignore
        def __init__(self, *args, **kwargs): pass

