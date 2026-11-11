"""Validated configuration for ISL-MEGA.

The defaults are intentionally modest for smoke tests. ``configs/research.yaml``
contains the high-capacity research candidate; it is not a claim that those
hyperparameters are empirically optimal.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json
import math


@dataclass
class VisionConfig:
    backend: str = "native"  # native or timm
    timm_model_name: str = "vit_large_patch14_dinov2.lvd142m"
    pretrained: bool = False
    image_size: int = 224
    crop_size: int = 112
    patch_size: int = 16
    dim: int = 192
    depth: int = 4
    heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    separate_hand_face_body: bool = True
    freeze_backbone: bool = False

    def validate(self) -> None:
        if self.backend not in {"native", "timm"}:
            raise ValueError("vision.backend must be 'native' or 'timm'")
        if min(self.image_size, self.crop_size, self.patch_size, self.dim, self.depth, self.heads) <= 0:
            raise ValueError("Vision dimensions must be positive")
        if self.image_size < 2 or self.crop_size < 2:
            raise ValueError("Vision image/crop sizes must be at least 2 pixels")
        if self.backend == "native" and self.crop_size % self.patch_size != 0:
            raise ValueError("crop_size must be divisible by patch_size for the native ViT")
        if self.backend == "timm" and not self.timm_model_name:
            raise ValueError("vision.timm_model_name is required for timm backend")
        if self.dim % self.heads:
            raise ValueError("vision.dim must be divisible by vision.heads")
        _dropout(self.dropout, "vision.dropout")
        if not math.isfinite(self.mlp_ratio) or self.mlp_ratio <= 0:
            raise ValueError("vision.mlp_ratio must be positive and finite")


@dataclass
class PoseConfig:
    num_landmarks: int = 66
    node_dim: int = 11
    hidden_dim: int = 256
    layers: int = 2
    heads: int = 8
    dropout: float = 0.1
    min_confidence: float = 0.0
    max_interpolation_gap_seconds: float = 0.20
    normalize_by_shoulder_span: bool = False
    share_hand_encoder: bool = False
    cross_articulator_graph: bool = False
    cross_articulator_layers: int = 1
    cross_articulator_gate_init: float = 0.1

    def validate(self) -> None:
        if self.num_landmarks != 66:
            raise ValueError("Current pose schema requires 66 points: hands 42, body 6, face 18")
        if self.node_dim != 11:
            raise ValueError("Pose node feature contract is 11 channels")
        if self.hidden_dim <= 0 or self.heads <= 0 or self.hidden_dim % self.heads:
            raise ValueError("pose.hidden_dim must be positive and divisible by pose.heads")
        if self.layers <= 0 or self.cross_articulator_layers <= 0:
            raise ValueError("pose.layers and cross_articulator_layers must be positive")
        _dropout(self.dropout, "pose.dropout")
        if not _finite_in_bounds(self.max_interpolation_gap_seconds, 0.0, None):
            raise ValueError("max_interpolation_gap_seconds cannot be negative")
        if not _finite_in_bounds(self.min_confidence, 0.0, 1.0):
            raise ValueError("pose.min_confidence must be in [0,1]")
        if not _finite_in_bounds(self.cross_articulator_gate_init, -5.0, 5.0):
            raise ValueError("pose.cross_articulator_gate_init must be finite in [-5, 5]")


@dataclass
class MotionConfig:
    enabled: bool = True
    hidden_dim: int = 256
    temporal_kernel: int = 5
    dropout: float = 0.1

    def validate(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("motion.enabled must be a boolean")
        if self.hidden_dim < 2 or self.hidden_dim % 2 or self.temporal_kernel < 1 or self.temporal_kernel % 2 == 0:
            raise ValueError("motion.hidden_dim must be positive/even and temporal_kernel must be positive/odd")
        _dropout(self.dropout, "motion.dropout")


@dataclass
class FusionConfig:
    dim: int = 512
    num_heads: int = 8
    layers: int = 2
    quality_dim: int = 5
    dropout: float = 0.1
    continuous_time_lag: bool = False
    lag_window_seconds: float = 0.20
    lag_radius_frames: int = 4
    reliability_uncertainty_gate: bool = False

    def validate(self) -> None:
        if self.dim < 2 or self.num_heads <= 0 or self.dim % self.num_heads:
            raise ValueError("fusion.dim must be >=2 and divisible by fusion.num_heads")
        if self.layers <= 0 or self.quality_dim != 5:
            raise ValueError("fusion.layers must be positive and quality_dim must equal 5")
        if not _finite_in_bounds(self.lag_window_seconds, 0.0, None, lower_inclusive=False) or self.lag_radius_frames < 0:
            raise ValueError("fusion.lag_window_seconds must be positive/finite and lag_radius_frames >= 0")
        _dropout(self.dropout, "fusion.dropout")


@dataclass
class FingerspellingConfig:
    enabled: bool = True
    hidden_dim: int = 256
    layers: int = 2
    dropout: float = 0.1
    vocabulary: str = "abcdefghijklmnopqrstuvwxyz0123456789-' "
    span_routing: bool = False

    def validate(self) -> None:
        if self.hidden_dim < 2 or self.hidden_dim % 2 or self.layers <= 0:
            raise ValueError("fingerspelling.hidden_dim must be positive/even and layers positive")
        if self.vocabulary != "abcdefghijklmnopqrstuvwxyz0123456789-' ":
            raise ValueError("Current CTC codec supports lowercase Latin letters, digits, hyphen, apostrophe and space only")
        _dropout(self.dropout, "fingerspelling.dropout")


@dataclass
class TemporalConfig:
    dim: int = 512
    local_kernels: tuple[int, ...] = (3, 5, 9, 15)
    local_dilations: tuple[int, ...] = (1, 1, 2, 4)
    global_layers: int = 12
    heads: int = 8
    ffn_dim: int = 2048
    dropout: float = 0.1
    relative_time_buckets: int = 64
    max_elapsed_seconds: float = 30.0
    coarse_stride: int = 2
    detail_window: int = 7
    planner_queries: int = 16
    planner_layers: int = 2
    adaptive_consolidation: bool = False
    diverse_ordered_planning: bool = False
    dynamic_plan_routing: bool = False
    min_plan_queries: int = 2
    plan_order_margin: float = 0.0

    def validate(self) -> None:
        if self.dim <= 0 or self.heads <= 0 or self.dim % self.heads:
            raise ValueError("temporal.dim must be positive and divisible by temporal.heads")
        if self.global_layers <= 0 or self.ffn_dim <= 0:
            raise ValueError("temporal layer and FFN sizes must be positive")
        if not self.local_kernels or len(self.local_kernels) != len(self.local_dilations):
            raise ValueError("local_kernels and local_dilations must be nonempty and equal length")
        if self.dim % len(self.local_kernels):
            raise ValueError("temporal.dim must be divisible by len(temporal.local_kernels)")
        if any(k <= 0 or k % 2 == 0 for k in self.local_kernels):
            raise ValueError("All local temporal kernels must be positive odd integers")
        if any(d <= 0 for d in self.local_dilations):
            raise ValueError("All local temporal dilations must be positive")
        if self.coarse_stride < 2 or self.detail_window < 1 or self.detail_window % 2 == 0:
            raise ValueError("coarse_stride >= 2 and positive odd detail_window required")
        if self.planner_queries <= 0 or self.planner_layers <= 0:
            raise ValueError("planner settings must be positive")
        if not 1 <= self.min_plan_queries <= self.planner_queries:
            raise ValueError("min_plan_queries must be in [1, planner_queries]")
        if not _finite_in_bounds(self.plan_order_margin, 0.0, None):
            raise ValueError("plan_order_margin cannot be negative")
        if self.relative_time_buckets < 4 or self.relative_time_buckets % 2:
            raise ValueError("relative_time_buckets must be even and >= 4")
        if not _finite_in_bounds(self.max_elapsed_seconds, 1e-5, None, lower_inclusive=False):
            raise ValueError("max_elapsed_seconds must be greater than 1e-5 seconds")
        _dropout(self.dropout, "temporal.dropout")


@dataclass
class DecoderConfig:
    model_name: str = "google/flan-t5-xl"
    local_files_only: bool = False
    tokenizer_name: str | None = None
    max_target_tokens: int = 256
    freeze_decoder: bool = False
    use_semantic_ot: bool = True
    ot_epsilon: float = 0.08
    ot_iterations: int = 30
    plan_semantic_ot: bool = False
    plan_ot_alpha: float = 0.5
    use_counterfactual_grounding: bool = False
    counterfactual_margin: float = 0.5
    counterfactual_kl_margin: float = 0.2
    counterfactual_mode: str = "batch_swap_or_zero"

    def validate(self) -> None:
        if not self.model_name:
            raise ValueError("decoder.model_name is required")
        if (self.max_target_tokens <= 0 or self.ot_iterations <= 0
                or not _finite_in_bounds(self.ot_epsilon, 0.0, None, lower_inclusive=False)):
            raise ValueError("Decoder token/OT settings must be positive and finite")
        if not _finite_in_bounds(self.plan_ot_alpha, 0.0, 1.0):
            raise ValueError("decoder.plan_ot_alpha must be in [0,1]")
        if (not _finite_in_bounds(self.counterfactual_margin, 0.0, None)
                or not _finite_in_bounds(self.counterfactual_kl_margin, 0.0, None)):
            raise ValueError("counterfactual margins cannot be negative")
        if self.counterfactual_mode not in {"batch_swap_or_zero", "zero"}:
            raise ValueError("decoder.counterfactual_mode must be 'batch_swap_or_zero' or 'zero'")


@dataclass
class DataConfig:
    train_manifest: str = "data/manifests/train.jsonl"
    valid_manifest: str = "data/manifests/valid.jsonl"
    test_manifest: str = "data/manifests/test.jsonl"
    image_size: int = 224
    crop_size: int = 112
    num_workers: int = 4
    pin_memory: bool = True
    max_target_tokens: int = 256
    strict_source_disjoint: bool = True
    require_signer_disjoint: bool = False

    def validate(self) -> None:
        if self.image_size < 2 or self.crop_size < 2 or self.num_workers < 0:
            raise ValueError("Invalid data dimensions/worker count")
        if self.max_target_tokens <= 0:
            raise ValueError("max_target_tokens must be positive")


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 4
    visual_lr: float = 2e-4
    decoder_lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    grad_clip_norm: float = 1.0
    amp_dtype: str = "bf16"  # bf16, fp16, or none
    ctc_weight: float = 0.10
    ot_weight: float = 0.05
    cross_modal_weight: float = 0.05
    cross_modal_temperature: float = 0.07
    plan_diversity_weight: float = 0.0
    plan_order_weight: float = 0.0
    fs_spot_weight: float = 0.0
    fusion_entropy_weight: float = 0.0
    counterfactual_weight: float = 0.0
    label_smoothing: float = 0.0
    seed: int = 2026
    output_dir: str = "runs/isl-mega"
    log_every: int = 20
    save_every_epochs: int = 1

    def validate(self) -> None:
        if self.epochs <= 0 or self.batch_size <= 0:
            raise ValueError("epochs and batch_size must be positive")
        if (not _finite_in_bounds(self.visual_lr, 0.0, None, lower_inclusive=False)
                or not _finite_in_bounds(self.decoder_lr, 0.0, None, lower_inclusive=False)
                or not _finite_in_bounds(self.grad_clip_norm, 0.0, None, lower_inclusive=False)
                or not _finite_in_bounds(self.weight_decay, 0.0, None)):
            raise ValueError("Invalid optimizer settings")
        if not _finite_in_bounds(self.warmup_ratio, 0.0, 1.0, upper_inclusive=False):
            raise ValueError("warmup_ratio must be in [0,1)")
        if self.amp_dtype not in {"bf16", "fp16", "none"}:
            raise ValueError("amp_dtype must be bf16, fp16, or none")
        if self.log_every <= 0 or self.save_every_epochs <= 0:
            raise ValueError("log_every and save_every_epochs must be positive")
        for name in (
            "ctc_weight", "ot_weight", "cross_modal_weight", "label_smoothing",
            "plan_diversity_weight", "plan_order_weight", "fs_spot_weight",
            "fusion_entropy_weight", "counterfactual_weight",
        ):
            if not _finite_in_bounds(getattr(self, name), 0.0, None):
                raise ValueError(f"{name} cannot be negative or nonfinite")
        if not _finite_in_bounds(self.cross_modal_temperature, 0.0, None, lower_inclusive=False):
            raise ValueError("cross_modal_temperature must be positive and finite")
        if self.label_smoothing >= 1.0:
            raise ValueError("label_smoothing must be < 1")


@dataclass
class ISLMEGAConfig:
    vision: VisionConfig = field(default_factory=VisionConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    motion: MotionConfig = field(default_factory=MotionConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    fingerspelling: FingerspellingConfig = field(default_factory=FingerspellingConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        for section in (self.vision, self.pose, self.motion, self.fusion, self.fingerspelling,
                        self.temporal, self.decoder, self.data, self.train):
            section.validate()
        # Expert projections map into fusion.dim, so input dimensions may differ.
        if self.temporal.dim != self.fusion.dim:
            raise ValueError("temporal.dim must equal fusion.dim")
        if self.temporal.dim % self.temporal.heads:
            raise ValueError("temporal dimension/head mismatch")
        if self.vision.crop_size != self.data.crop_size:
            raise ValueError("vision.crop_size and data.crop_size must match")
        if self.vision.image_size != self.data.image_size:
            raise ValueError("vision.image_size and data.image_size must match")
        if self.data.max_target_tokens != self.decoder.max_target_tokens:
            raise ValueError("data.max_target_tokens and decoder.max_target_tokens must match")
        if not isinstance(self.data.strict_source_disjoint, bool):
            raise ValueError("data.strict_source_disjoint must be boolean")
        if not isinstance(self.data.require_signer_disjoint, bool):
            raise ValueError("data.require_signer_disjoint must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


def _finite_in_bounds(
    value: float,
    low: float | None,
    high: float | None,
    lower_inclusive: bool = True,
    upper_inclusive: bool = True,
) -> bool:
    if value is None or isinstance(value, bool) or not math.isfinite(value):
        return False
    if low is not None and (value < low if lower_inclusive else value <= low):
        return False
    if high is not None and (value > high if upper_inclusive else value >= high):
        return False
    return True


def _dropout(value: float, name: str) -> None:
    if not _finite_in_bounds(value, 0.0, 1.0, upper_inclusive=False):
        raise ValueError(f"{name} must be in [0,1)")


def _merge_dataclass(cls, values: dict[str, Any]):
    if not isinstance(values, dict):
        raise ValueError(f"Configuration section for {cls.__name__} must be a mapping")
    valid = set(cls.__dataclass_fields__)
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} configuration keys: {sorted(unknown)}")
    return cls(**values)


def load_config(path: str | Path) -> ISLMEGAConfig:
    import yaml
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a YAML mapping")
    valid_sections = {"vision", "pose", "motion", "fusion", "fingerspelling", "temporal",
                      "decoder", "data", "train"}
    unknown_sections = set(raw) - valid_sections
    if unknown_sections:
        raise ValueError(f"Unknown top-level configuration sections: {sorted(unknown_sections)}")
    config = ISLMEGAConfig(
        vision=_merge_dataclass(VisionConfig, raw.get("vision", {})),
        pose=_merge_dataclass(PoseConfig, raw.get("pose", {})),
        motion=_merge_dataclass(MotionConfig, raw.get("motion", {})),
        fusion=_merge_dataclass(FusionConfig, raw.get("fusion", {})),
        fingerspelling=_merge_dataclass(FingerspellingConfig, raw.get("fingerspelling", {})),
        temporal=_merge_dataclass(TemporalConfig, raw.get("temporal", {})),
        decoder=_merge_dataclass(DecoderConfig, raw.get("decoder", {})),
        data=_merge_dataclass(DataConfig, raw.get("data", {})),
        train=_merge_dataclass(TrainConfig, raw.get("train", {})),
    )
    config.validate()
    return config
