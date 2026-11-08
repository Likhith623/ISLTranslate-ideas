"""Typed configuration for the RA-MAGC research prototype."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    """Path conventions for integration with the parent ISLTranslate project.

    This package does not read these paths or generate data splits. They are
    configuration metadata for the integration layer.
    """

    annotations: str = "data/raw/ISLTranslate.csv"
    dataset_name: str = "isltranslate"
    dataset_revision: str = "original"
    pose_root: str = "data/raw/poses"
    video_root: str | None = None
    pose_pattern: str = "{uid}.pose"
    video_pattern: str = "{uid}.mp4"
    source_mapping: str = "data/metadata/source_mapping.csv"
    manifest: str = "data/manifests/target.jsonl"
    split_dir: str = "splits/source-v1"
    feature_dir: str = "artifacts/features"
    stats_path: str = "artifacts/stats/pose.json"
    t5_dir: str = "artifacts/models/flan-t5-small"
    t5_repo: str = "google/flan-t5-small"
    uid_col: str = "uid"
    text_col: str = "text"
    interpreter_csv: str = "data/raw/ISL-signer_validation.csv"
    split_protocol: str = "source-held-out"
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 2026

    def __post_init__(self) -> None:
        if len(self.split_ratios) != 3 or any(r <= 0 for r in self.split_ratios):
            raise ValueError("split_ratios must contain three positive values")
        if abs(sum(self.split_ratios) - 1.0) > 1e-6:
            raise ValueError("split_ratios must sum to 1.0")


@dataclass
class HandGraphConfig:
    """Shared anatomical encoder configuration for each 21-node hand."""

    num_joints: int = 21
    in_node_dim: int = 13
    edge_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_joints != 21:
            raise ValueError("The supplied hand feature extractor requires exactly 21 joints")
        if self.in_node_dim != 13 or self.edge_dim != 4:
            raise ValueError("Hand input contract is 13 node features and 4 edge features")
        _validate_encoder_dims(self.hidden_dim, self.num_heads, self.num_layers, self.dropout, "hand")


@dataclass
class BodyGraphConfig:
    """Upper-body skeleton encoder configuration."""

    num_joints: int = 6
    in_node_dim: int = 10
    edge_dim: int = 4
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_joints != 6:
            raise ValueError("The supplied upper-body feature extractor requires exactly 6 joints")
        if self.in_node_dim != 10 or self.edge_dim != 4:
            raise ValueError("Body input contract is 10 node features and 4 edge features")
        _validate_encoder_dims(self.hidden_dim, self.num_heads, self.num_layers, self.dropout, "body")


@dataclass
class FaceSetConfig:
    """Optional sparse facial landmark encoder configuration."""

    enabled: bool = False
    num_points: int = 18
    in_node_dim: int = 13
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.num_points != 18 or self.in_node_dim != 13:
            raise ValueError("The supplied face extractor requires 18 points and 13 node features")
        _validate_encoder_dims(self.hidden_dim, self.num_heads, self.num_layers, self.dropout, "face")


@dataclass
class RegionFusionConfig:
    """Cross-region interaction and quality-conditioned gated fusion."""

    region_dim: int = 128
    fused_dim: int = 256
    num_interaction_layers: int = 2
    num_heads: int = 4
    quality_feature_dim: int = 6
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.region_dim <= 0 or self.fused_dim <= 0:
            raise ValueError("Fusion dimensions must be positive")
        if self.num_heads <= 0 or self.region_dim % self.num_heads:
            raise ValueError("fusion.region_dim must be divisible by fusion.num_heads")
        if self.num_interaction_layers <= 0:
            raise ValueError("fusion.num_interaction_layers must be positive")
        if self.quality_feature_dim != 6:
            raise ValueError("The reliability gate expects exactly six quality descriptors")
        _validate_dropout(self.dropout, "fusion")


@dataclass
class MultiScaleTemporalConfig:
    """Stride-one multiscale temporal front-end."""

    in_channels: int = 256
    out_channels: int = 256
    num_blocks: int = 2
    kernel_sizes: tuple[int, int, int, int] = (3, 5, 5, 9)
    dilations: tuple[int, int, int, int] = (1, 1, 2, 2)
    branch_dim: int = 64
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if self.in_channels != self.out_channels:
            raise ValueError("Residual multiscale blocks require in_channels == out_channels")
        if self.in_channels <= 0 or self.num_blocks <= 0 or self.branch_dim <= 0:
            raise ValueError("Temporal channel counts and block counts must be positive")
        if self.branch_dim * 4 != self.out_channels:
            raise ValueError("branch_dim * 4 must equal out_channels")
        if len(self.kernel_sizes) != 4 or len(self.dilations) != 4:
            raise ValueError("Exactly four kernel sizes and four dilation factors are required")
        if any(k <= 0 or k % 2 == 0 for k in self.kernel_sizes):
            raise ValueError("Temporal kernel sizes must be positive odd integers")
        if any(d <= 0 for d in self.dilations):
            raise ValueError("Temporal dilations must be positive integers")
        _validate_dropout(self.dropout, "multi_scale")


@dataclass
class ConformerConfig:
    """Global temporal Conformer configuration."""

    d_model: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    dim_feedforward: int = 1024
    depthwise_kernel_size: int = 15
    dropout: float = 0.1
    max_elapsed_seconds: float = 20.0
    num_time_buckets: int = 64

    def __post_init__(self) -> None:
        if self.d_model <= 0 or self.num_blocks <= 0 or self.dim_feedforward <= 0:
            raise ValueError("Conformer dimensions and number of blocks must be positive")
        if self.num_heads <= 0 or self.d_model % self.num_heads:
            raise ValueError("conformer.d_model must be divisible by conformer.num_heads")
        if self.depthwise_kernel_size <= 0 or self.depthwise_kernel_size % 2 == 0:
            raise ValueError("Conformer depthwise kernel size must be a positive odd integer")
        if self.max_elapsed_seconds <= 1e-4:
            raise ValueError("max_elapsed_seconds must be greater than 1e-4 seconds")
        if self.num_time_buckets < 4 or self.num_time_buckets % 2:
            raise ValueError("num_time_buckets must be an even integer >= 4")
        _validate_dropout(self.dropout, "conformer")


@dataclass
class ModelConfig:
    """Top-level RA-MAGC configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    hand: HandGraphConfig = field(default_factory=HandGraphConfig)
    body: BodyGraphConfig = field(default_factory=BodyGraphConfig)
    face: FaceSetConfig = field(default_factory=FaceSetConfig)
    fusion: RegionFusionConfig = field(default_factory=RegionFusionConfig)
    multi_scale: MultiScaleTemporalConfig = field(default_factory=MultiScaleTemporalConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    decoder_type: Literal["t5", "lstm"] = "t5"
    t5_d_model: int = 512
    dropout: float = 0.1
    lr_visual: float = 3e-4
    lr_t5: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_epochs: int = 20
    batch_size: int = 16
    max_frames_per_batch: int = 16000
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hand.hidden_dim != self.fusion.region_dim:
            raise ValueError("fusion.region_dim must equal hand.hidden_dim")
        if self.body.hidden_dim != self.fusion.region_dim:
            raise ValueError("fusion.region_dim must equal body.hidden_dim")
        if self.face.enabled and self.face.hidden_dim != self.fusion.region_dim:
            raise ValueError("When enabled, face.hidden_dim must equal fusion.region_dim")
        if self.fusion.fused_dim != self.multi_scale.in_channels:
            raise ValueError("multi_scale.in_channels must equal fusion.fused_dim")
        if self.multi_scale.out_channels != self.conformer.d_model:
            raise ValueError("multi_scale.out_channels must equal conformer.d_model")
        if self.t5_d_model <= 0:
            raise ValueError("t5_d_model must be positive")
        if self.decoder_type not in ("t5", "lstm"):
            raise ValueError("decoder_type must be 't5' or 'lstm'")
        if self.batch_size <= 0 or self.max_frames_per_batch <= 0 or self.max_epochs <= 0:
            raise ValueError("Training metadata values must be positive")
        if self.lr_visual <= 0 or self.lr_t5 <= 0 or self.weight_decay < 0:
            raise ValueError("Learning rates must be positive and weight_decay non-negative")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise ValueError("warmup_ratio must be in [0, 1)")
        _validate_dropout(self.dropout, "model")


def _validate_encoder_dims(hidden_dim: int, num_heads: int, num_layers: int, dropout: float, name: str) -> None:
    if hidden_dim <= 0 or num_heads <= 0 or hidden_dim % num_heads:
        raise ValueError(f"{name}.hidden_dim must be positive and divisible by num_heads")
    if num_layers <= 0:
        raise ValueError(f"{name}.num_layers must be positive")
    _validate_dropout(dropout, name)


def _validate_dropout(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name}.dropout must be in [0, 1)")
