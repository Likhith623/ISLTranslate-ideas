"""Configuration definitions for RA-MAGC (Reliability-Aware Multi-Rate Articulatory Graph-Conformer).

Preserves all canonical dataset paths and splits from the ISLTranslate repository.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal


@dataclass
class DataConfig:
    """Exact dataset paths matching the ISLTranslate repository contracts."""
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
    
    # Target and interpreter column definitions
    uid_col: str = "uid"
    text_col: str = "text"
    interpreter_csv: str = "data/raw/ISL-signer_validation.csv"
    
    # Split configuration: source-held-out protocol (NEVER random or stratified k-fold)
    split_protocol: str = "source-held-out"
    split_ratios: tuple[float, float, float] = (0.8, 0.1, 0.1)
    split_seed: int = 2026


@dataclass
class HandGraphConfig:
    """Configuration for anatomical hand graph encoder (shared across left and right hands)."""
    num_joints: int = 21
    in_node_dim: int = 11  # (x, y, vx, vy, x_local, y_local, obs, imp, pos_val, vel_val, loc_val)
    edge_dim: int = 4      # (dx, dy, length, edge_valid)
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class BodyGraphConfig:
    """Configuration for upper body skeleton graph encoder."""
    num_joints: int = 6   # Shoulders, elbows, wrists
    in_node_dim: int = 8  # (x, y, vx, vy, obs, imp, pos_val, vel_val)
    edge_dim: int = 4     # (dx, dy, length, edge_valid)
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class FaceSetConfig:
    """Configuration for sparse facial landmark set encoder."""
    enabled: bool = False
    num_points: int = 18
    in_node_dim: int = 11  # (x, y, vx, vy, x_local, y_local, obs, imp, pos_val, vel_val, loc_val)
    hidden_dim: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.1


@dataclass
class RegionFusionConfig:
    """Cross-region interaction and reliability-aware gated fusion."""
    region_dim: int = 128
    fused_dim: int = 256
    num_interaction_layers: int = 2
    num_heads: int = 4
    quality_feature_dim: int = 4  # (ratio_obs, ratio_imp, ratio_pos_val, ratio_vel_val)
    dropout: float = 0.1


@dataclass
class MultiScaleTemporalConfig:
    """Full-resolution multi-scale temporal detail encoder (running at stride 1)."""
    in_channels: int = 256
    out_channels: int = 256
    num_blocks: int = 2
    kernel_sizes: tuple[int, int, int, int] = (3, 5, 5, 9)
    dilations: tuple[int, int, int, int] = (1, 1, 2, 2)
    branch_dim: int = 64
    dropout: float = 0.1


@dataclass
class ConformerConfig:
    """Global temporal Conformer backbone operating on coarse pairwise sequence."""
    d_model: int = 256
    num_blocks: int = 4
    num_heads: int = 8
    dim_feedforward: int = 1024
    depthwise_kernel_size: int = 15
    dropout: float = 0.1
    max_elapsed_seconds: float = 20.0
    num_time_buckets: int = 64


@dataclass
class ModelConfig:
    """Top-level model architecture configuration."""
    data: DataConfig = field(default_factory=DataConfig)
    hand: HandGraphConfig = field(default_factory=HandGraphConfig)
    body: BodyGraphConfig = field(default_factory=BodyGraphConfig)
    face: FaceSetConfig = field(default_factory=FaceSetConfig)
    fusion: RegionFusionConfig = field(default_factory=RegionFusionConfig)
    multi_scale: MultiScaleTemporalConfig = field(default_factory=MultiScaleTemporalConfig)
    conformer: ConformerConfig = field(default_factory=ConformerConfig)
    
    # Text generation model
    decoder_type: Literal["t5", "lstm"] = "t5"
    t5_d_model: int = 512
    dropout: float = 0.1
    
    # Optimization
    lr_visual: float = 3e-4
    lr_t5: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_epochs: int = 20
    batch_size: int = 16
    max_frames_per_batch: int = 16000
    seed: int = 42

