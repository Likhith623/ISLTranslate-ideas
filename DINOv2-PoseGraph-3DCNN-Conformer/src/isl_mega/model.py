"""Complete ISL-MEGA multimodal architecture."""

from __future__ import annotations

import torch
from torch import nn

from .config import ISLMEGAConfig
from .vision import AppearanceExpert, crop_regions, roi_quality
from .pose import PoseExpert
from .motion import MotionExpert
from .fingerspelling import FingerspellingExpert
from .fusion import ReliabilityAwareFusion, STREAM_NAMES
from .temporal import HierarchicalTemporalEncoder
from .planning import LatentPlanGrounder
from .decoder import T5VisualDecoder


class ISLMEGA(nn.Module):
    """Multi-Expert, Evidence-Grounded Architecture for ISL Translation.

    Inputs retain the complete original video timeline. Each frame is processed
    by RGB appearance, pose, motion and fingerspelling experts. A reliability-
    aware fusion block feeds a hierarchical temporal encoder and latent plan
    grounder. When a T5 decoder is attached, frame and plan tokens condition its
    full encoder-decoder stack.
    """

    def __init__(self, config: ISLMEGAConfig, decoder: T5VisualDecoder | None = None):
        super().__init__()
        config.validate()
        self.config = config
        self.appearance = AppearanceExpert(config.vision)
        self.pose = PoseExpert(config.pose)
        self.motion = MotionExpert(config.motion)
        self.fingerspelling = FingerspellingExpert(
            config.vision.dim, config.pose.hidden_dim, config.fingerspelling.hidden_dim,
            config.fingerspelling.layers, config.fingerspelling.dropout,
            span_routing=config.fingerspelling.span_routing,
        ) if config.fingerspelling.enabled else None
        d_vis, d_pose, d_motion = config.vision.dim, config.pose.hidden_dim, config.motion.hidden_dim
        d_fs = config.fingerspelling.hidden_dim if config.fingerspelling.enabled else d_vis
        dims = {
            "left_hand_rgb": d_vis, "right_hand_rgb": d_vis, "face_rgb": d_vis, "body_rgb": d_vis,
            "left_hand_pose": d_pose, "right_hand_pose": d_pose, "body_pose": d_pose, "face_pose": d_pose,
            "motion": d_motion, "fingerspelling": d_fs,
        }
        self.fusion = ReliabilityAwareFusion(dims, config.fusion)
        self.temporal = HierarchicalTemporalEncoder(config.temporal)
        self.planner = LatentPlanGrounder(
            config.temporal.dim,
            config.temporal.planner_queries,
            config.temporal.planner_layers,
            config.temporal.heads,
            config.temporal.dropout,
            diverse_ordered=config.temporal.diverse_ordered_planning,
            dynamic_routing=config.temporal.dynamic_plan_routing,
            min_queries=config.temporal.min_plan_queries,
            order_margin=config.temporal.plan_order_margin,
            target_stride=config.temporal.coarse_stride * 2,
        )
        self.decoder = decoder
        self.output_dim = config.temporal.dim

    def encode(self, *, frames: torch.Tensor, timestamps: torch.Tensor, lengths: torch.Tensor,
               pose_xy: torch.Tensor, pose_confidence: torch.Tensor, pose_valid: torch.Tensor,
               confidence_known: torch.Tensor, boxes: torch.Tensor, box_valid: torch.Tensor,
               box_scores: torch.Tensor | None = None) -> dict:
        if frames.ndim != 5 or frames.shape[2] != 3:
            raise ValueError("frames must have shape [B,T,3,H,W]")
        b, t = frames.shape[:2]
        if timestamps.shape != (b, t) or lengths.shape != (b,):
            raise ValueError("timestamps/lengths have incompatible shape")
        if boxes.shape != (b, t, 4, 4) or box_valid.shape != (b, t, 4):
            raise ValueError("boxes must be [B,T,4,4] and box_valid [B,T,4]")
        frame_mask = torch.arange(t, device=frames.device)[None] < lengths[:, None]
        effective_box_valid = box_valid.bool() & frame_mask[..., None]
        crops = crop_regions(frames, boxes, effective_box_valid, self.config.vision.crop_size)
        visual = self.appearance(crops, effective_box_valid)
        pose = self.pose(pose_xy, pose_confidence, pose_valid, confidence_known,
                         timestamps, frame_mask)
        body_available = box_valid[:, :, 3].bool() & frame_mask
        motion = self.motion(crops[:, :, 3:4], frame_mask, body_available)
        fs = None
        if self.fingerspelling is not None:
            left_fs_available = effective_box_valid[:, :, 1] | pose["available"]["left_hand"]
            right_fs_available = effective_box_valid[:, :, 2] | pose["available"]["right_hand"]
            fs = self.fingerspelling(visual["left_hand"], visual["right_hand"],
                                     pose["features"]["left_hand"], pose["features"]["right_hand"],
                                     lengths, left_fs_available, right_fs_available)

        streams = {
            "left_hand_rgb": visual["left_hand"], "right_hand_rgb": visual["right_hand"],
            "face_rgb": visual["face"], "body_rgb": visual["body"],
            "left_hand_pose": pose["features"]["left_hand"],
            "right_hand_pose": pose["features"]["right_hand"],
            "body_pose": pose["features"]["body"], "face_pose": pose["features"]["face"],
            "motion": motion,
            "fingerspelling": fs["features"] if fs is not None else visual["left_hand"].new_zeros(
                (b, t, self.config.vision.dim)),
        }
        roi_q = roi_quality(crops, boxes, effective_box_valid, box_scores, frame_mask)
        quality = {
            "left_hand_rgb": roi_q[:, :, 1], "right_hand_rgb": roi_q[:, :, 2],
            "face_rgb": roi_q[:, :, 0], "body_rgb": roi_q[:, :, 3],
            "left_hand_pose": pose["quality"]["left_hand"],
            "right_hand_pose": pose["quality"]["right_hand"],
            "body_pose": pose["quality"]["body"], "face_pose": pose["quality"]["face"],
        }
        diff_energy = torch.zeros((b, t), dtype=frames.dtype, device=frames.device)
        body = crops[:, :, 3]
        if t > 1:
            pair_valid = body_available[:, 1:] & body_available[:, :-1]
            raw_diff = (body[:, 1:] - body[:, :-1]).abs().mean(dim=(2, 3, 4))
            diff_energy[:, 1:] = torch.where(pair_valid, raw_diff, torch.zeros_like(raw_diff))
        motion_avail = body_available
        quality["motion"] = torch.stack([motion_avail.float(), diff_energy,
                                          roi_q[:, :, 3, 3], motion_avail.float(), frame_mask.float()], -1)
        if fs is not None:
            left_ok = effective_box_valid[:, :, 1] | pose["available"]["left_hand"]
            right_ok = effective_box_valid[:, :, 2] | pose["available"]["right_hand"]
            fs_avail = fs["available"] & frame_mask
            fs_conf_signal = (fs["spot_prob"] if self.config.fingerspelling.span_routing else fs["gate"]).float()
            quality["fingerspelling"] = torch.stack([
                fs_avail.float(), (left_ok | right_ok).float(), fs_conf_signal,
                fs["logits"].float().softmax(-1).amax(-1), frame_mask.float()
            ], -1)
        else:
            quality["fingerspelling"] = torch.zeros((b, t, 5), device=frames.device, dtype=frames.dtype)
        available = {
            "left_hand_rgb": effective_box_valid[:, :, 1],
            "right_hand_rgb": effective_box_valid[:, :, 2],
            "face_rgb": effective_box_valid[:, :, 0],
            "body_rgb": effective_box_valid[:, :, 3],
            "left_hand_pose": pose["available"]["left_hand"],
            "right_hand_pose": pose["available"]["right_hand"],
            "body_pose": pose["available"]["body"], "face_pose": pose["available"]["face"],
            "motion": motion_avail,
            "fingerspelling": fs["available"] if fs is not None else torch.zeros_like(frame_mask),
        }
        fused, modality_weights, interacted, modality_embeddings, fusion_aux = self.fusion.forward_with_aux(
            streams, available, quality, timestamps=timestamps
        )
        temporal, memory_mask, temporal_aux = self.temporal(fused, lengths, timestamps)
        plan, grounding, plan_aux = self.planner.forward_with_aux(temporal, memory_mask, timestamps=timestamps)
        result = {
            "memory": temporal, "memory_mask": memory_mask, "plan_tokens": plan,
            "plan_mask": plan_aux["plan_mask"],
            "plan_diversity_loss": plan_aux["diversity_loss"],
            "plan_order_loss": plan_aux["order_loss"],
            "grounding_attention": grounding, "modality_weights": modality_weights,
            "modality_interactions": interacted,
            "modality_embeddings": modality_embeddings,
            "stream_available": torch.stack([available[name] for name in STREAM_NAMES], dim=2),
            "ctc_logits": fs["logits"] if fs is not None else None,
            "ctc_available": fs["available"] if fs is not None else None,
            "fingerspelling_gate": fs["gate"] if fs is not None else None,
            "spot_logits": fs["spot_logits"] if fs is not None else None,
            "spot_prob": fs["spot_prob"] if fs is not None else None,
            "fusion_aux": fusion_aux,
            "temporal_aux": temporal_aux,
        }
        return result

    def forward(self, *, labels: torch.Tensor | None = None, semantic_ot: bool = False,
                counterfactual_grounding: bool = False, **inputs) -> dict:
        encoded = self.encode(**inputs)
        if labels is not None:
            if self.decoder is None:
                raise RuntimeError("No T5 decoder is attached. Build with build_model(..., with_decoder=True).")
            decoded = self.decoder(
                encoded["memory"],
                encoded["memory_mask"],
                encoded["plan_tokens"],
                labels,
                semantic_ot=semantic_ot,
                ot_epsilon=self.config.decoder.ot_epsilon,
                ot_iterations=self.config.decoder.ot_iterations,
                label_smoothing=self.config.train.label_smoothing,
                plan_mask=encoded["plan_mask"],
                plan_semantic_ot=self.config.decoder.plan_semantic_ot,
                plan_ot_alpha=self.config.decoder.plan_ot_alpha,
                counterfactual_grounding=counterfactual_grounding or (
                    self.config.decoder.use_counterfactual_grounding and self.config.train.counterfactual_weight > 0
                ),
                counterfactual_margin=self.config.decoder.counterfactual_margin,
                counterfactual_kl_margin=self.config.decoder.counterfactual_kl_margin,
                counterfactual_mode=self.config.decoder.counterfactual_mode,
            )
            encoded.update({f"decoder_{k}": v for k, v in decoded.items()})
        return encoded

    @torch.no_grad()
    def generate(self, num_beams: int = 4, max_new_tokens: int | None = None,
                 length_penalty: float = 1.0, **inputs) -> torch.Tensor:
        if self.decoder is None:
            raise RuntimeError("No T5 decoder is attached")
        encoded = self.encode(**inputs)
        return self.decoder.generate(
            encoded["memory"], encoded["memory_mask"], encoded["plan_tokens"],
            num_beams=num_beams,
            max_new_tokens=max_new_tokens or self.config.decoder.max_target_tokens,
            length_penalty=length_penalty,
            plan_mask=encoded["plan_mask"],
        )


def build_model(config: ISLMEGAConfig, with_decoder: bool = False) -> ISLMEGA:
    config.validate()
    decoder = None
    if with_decoder:
        try:
            from transformers import AutoModelForSeq2SeqLM
        except ImportError as exc:
            raise ImportError("Install `pip install -e '.[text]'` to load FLAN-T5") from exc
        hf = AutoModelForSeq2SeqLM.from_pretrained(
            config.decoder.model_name, local_files_only=config.decoder.local_files_only
        )
        decoder = T5VisualDecoder(hf, config.temporal.dim, freeze=config.decoder.freeze_decoder)
    return ISLMEGA(config, decoder=decoder)
