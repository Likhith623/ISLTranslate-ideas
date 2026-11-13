"""
Configuration schema for the Articulatory-Bound Spatiotemporal Transformer (AB-ST).
Strongly typed with dataclasses, validation, and JSON/YAML serialization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except ImportError:
    _HAS_YAML = False


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    """
    Lightweight, zero-dependency YAML parser for standard key-value configs,
    nested sections, lists, and inline arrays.
    """
    lines = text.splitlines()

    def parse_val(v: str) -> Any:
        v = v.strip()
        if v.startswith("[") and v.endswith("]"):
            inner = v[1:-1].strip()
            if not inner:
                return []
            return [parse_val(x) for x in inner.split(",")]
        if v.lower() in ("true", "yes"):
            return True
        if v.lower() in ("false", "no"):
            return False
        if v.lower() in ("null", "none", "~"):
            return None
        if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
            return v[1:-1]
        try:
            if "." in v or "e" in v.lower():
                return float(v)
            return int(v)
        except ValueError:
            return v

    parsed_lines = []
    for line in lines:
        raw = line.split("#")[0].rstrip()
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip())
        stripped = raw.strip()
        parsed_lines.append((indent, stripped))

    def parse_block(idx: int, current_indent: int) -> Tuple[Any, int]:
        res: Dict[str, Any] = {}
        is_list = False
        list_res: List[Any] = []

        while idx < len(parsed_lines):
            indent, line = parsed_lines[idx]
            if indent < current_indent:
                break

            if line.startswith("- "):
                is_list = True
                val = parse_val(line[2:].strip())
                list_res.append(val)
                idx += 1
            elif ":" in line:
                parts = line.split(":", 1)
                k = parts[0].strip()
                v_str = parts[1].strip()
                if v_str != "":
                    res[k] = parse_val(v_str)
                    idx += 1
                else:
                    if idx + 1 < len(parsed_lines) and parsed_lines[idx + 1][0] > indent:
                        child_res, next_idx = parse_block(idx + 1, parsed_lines[idx + 1][0])
                        res[k] = child_res
                        idx = next_idx
                    else:
                        res[k] = None
                        idx += 1
            else:
                idx += 1

        return (list_res if is_list else res), idx

    result, _ = parse_block(0, 0)
    return result if isinstance(result, dict) else {}


def _dump_simple_yaml(data: Dict[str, Any], indent_level: int = 0) -> str:
    """Format dictionary into standard YAML string."""
    lines = []
    indent = "  " * indent_level
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{indent}{k}:")
            lines.append(_dump_simple_yaml(v, indent_level + 1))
        elif isinstance(v, list):
            if all(isinstance(x, (int, float, str)) for x in v) and len(v) <= 4:
                # Format as inline list
                inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in v)
                lines.append(f"{indent}{k}: [{inner}]")
            else:
                lines.append(f"{indent}{k}:")
                for item in v:
                    lines.append(f"{indent}  - {item}")
        elif isinstance(v, bool):
            lines.append(f"{indent}{k}: {'true' if v else 'false'}")
        elif v is None:
            lines.append(f"{indent}{k}: null")
        elif isinstance(v, (int, float)):
            lines.append(f"{indent}{k}: {v}")
        else:
            lines.append(f'{indent}{k}: "{v}"')
    return "\n".join(lines)


@dataclass
class DataConfig:
    """Dataset and preprocessing parameters."""
    num_landmarks: int = 66
    raw_landmarks: int = 543
    in_channels: int = 3
    max_frames: int = 256
    target_fps: int = 25
    image_size: Tuple[int, int] = (224, 224)
    patch_size: int = 14
    vocab_size: int = 32128  # Default T5 tokenizer vocab size
    max_text_len: int = 128
    dataset_name: str = "ISLTranslate"
    data_dir: str = "data"
    splits_file: Optional[str] = None
    num_workers: int = 4
    batch_size: int = 4


@dataclass
class PoseConfig:
    """Canonical 66-point pose & kinematic dominance settings."""
    num_joints: int = 66
    in_channels: int = 3
    has_confidence: bool = False
    
    # Canonical landmark index slices in 66-point format:
    # 0..20: Left Hand (21)
    # 21..41: Right Hand (21)
    # 42..47: Upper Body Pose (6: left_shoulder, right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist)
    # 48..65: Expressive Face (18 key face points)
    left_hand_slice: Tuple[int, int] = (0, 21)
    right_hand_slice: Tuple[int, int] = (21, 42)
    body_slice: Tuple[int, int] = (42, 48)
    face_slice: Tuple[int, int] = (48, 66)
    
    # Normalization anchors
    left_shoulder_idx: int = 42
    right_shoulder_idx: int = 43
    ema_momentum: float = 0.95
    eps: float = 1e-6
    
    # Kinematic dominance parameters
    velocity_window: int = 5
    energy_threshold: float = 1e-4
    use_role_embeddings: bool = True


@dataclass
class VisionConfig:
    """DINOv2 patch backbone configuration."""
    backbone_name: str = "dinov2_vits14"
    image_size: int = 224
    patch_size: int = 14
    embed_dim: int = 384
    pretrained: bool = True
    freeze_backbone: bool = True
    grid_size: Tuple[int, int] = (16, 16)  # 224 // 14 = 16


@dataclass
class BindingConfig:
    """Deformable Keypoint-to-Patch Binding module settings."""
    visual_dim: int = 384
    pose_dim: int = 3
    hidden_dim: int = 256
    num_heads: int = 4
    max_offset: float = 0.15
    dropout: float = 0.1
    use_deformable_sampling: bool = True


@dataclass
class ArticulatorConfig:
    """4-Articulator factorization settings."""
    num_articulators: int = 4
    articulator_names: List[str] = field(
        default_factory=lambda: ["dominant_hand", "base_hand", "face", "torso"]
    )
    embed_dim: int = 256
    aggregation_heads: int = 4
    dropout: float = 0.1


@dataclass
class TemporalConfig:
    """Axial Spatio-Temporal Conformer configuration."""
    hidden_dim: int = 256
    num_spatial_heads: int = 4
    num_temporal_heads: int = 8
    num_layers: int = 6
    ffn_expansion_factor: int = 4
    conv_kernel_size: int = 31
    dropout: float = 0.1
    max_seq_len: int = 512
    enable_fingerspelling_ctc: bool = True
    ctc_vocab_size: int = 64  # Characters for ISL fingerspelling / gloss


@dataclass
class ResamplerConfig:
    """Articulatory Q-Former resampler configuration."""
    num_queries: int = 32
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    output_dim: int = 768  # Target decoder dimension (e.g. FLAN-T5-base d_model)


@dataclass
class DecoderConfig:
    """Text generation decoder configuration."""
    model_type: str = "seq2seq"  # "seq2seq" (T5) or "causal" (LLaMA/Gemma)
    pretrained_model_name: str = "google/flan-t5-base"
    d_model: int = 768
    max_target_len: int = 128
    label_smoothing: float = 0.1
    beam_size: int = 4
    length_penalty: float = 1.0


@dataclass
class LossConfig:
    """Multi-task loss weights and regularization."""
    translation_weight: float = 1.0
    ctc_weight: float = 0.2
    articulatory_alignment_weight: float = 0.1
    modality_dropout_prob: float = 0.15
    label_smoothing: float = 0.1


@dataclass
class TrainConfig:
    """Optimization and training runtime parameters."""
    learning_rate: float = 1e-4
    min_learning_rate: float = 1e-6
    weight_decay: float = 1e-2
    warmup_steps: int = 1000
    max_steps: int = 50000
    gradient_accumulation_steps: int = 2
    gradient_clip_val: float = 1.0
    mixed_precision: str = "fp16"  # "no", "fp16", "bf16"
    seed: int = 42
    save_dir: str = "checkpoints"
    log_interval: int = 50
    eval_interval: int = 500
    save_top_k: int = 3


@dataclass
class ModelConfig:
    """Unified master configuration for AB-ST."""
    data: DataConfig = field(default_factory=DataConfig)
    pose: PoseConfig = field(default_factory=PoseConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)
    binding: BindingConfig = field(default_factory=BindingConfig)
    articulator: ArticulatorConfig = field(default_factory=ArticulatorConfig)
    temporal: TemporalConfig = field(default_factory=TemporalConfig)
    resampler: ResamplerConfig = field(default_factory=ResamplerConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration tree to a nested dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> ModelConfig:
        """Construct ModelConfig from dictionary."""
        cfg = cls()
        for section_name, section_class in [
            ("data", DataConfig),
            ("pose", PoseConfig),
            ("vision", VisionConfig),
            ("binding", BindingConfig),
            ("articulator", ArticulatorConfig),
            ("temporal", TemporalConfig),
            ("resampler", ResamplerConfig),
            ("decoder", DecoderConfig),
            ("loss", LossConfig),
            ("train", TrainConfig),
        ]:
            if section_name in data:
                sub_dict = data[section_name]
                if isinstance(sub_dict, dict):
                    # Convert list tuples if needed
                    for k, v in list(sub_dict.items()):
                        if isinstance(v, list) and len(v) == 2 and ("slice" in k or "grid_size" in k or "image_size" in k):
                            sub_dict[k] = tuple(v)
                    setattr(cfg, section_name, section_class(**sub_dict))
        cfg.validate()
        return cfg

    def validate(self) -> None:
        """Validate consistency of sub-module parameters."""
        assert self.pose.num_joints == 66, f"Expected 66 canonical joints, got {self.pose.num_joints}"
        assert self.pose.left_shoulder_idx == 42
        assert self.pose.right_shoulder_idx == 43
        img_size = self.vision.image_size if isinstance(self.vision.image_size, int) else self.vision.image_size[0]
        assert img_size % self.vision.patch_size == 0
        expected_grid = (
            img_size // self.vision.patch_size,
            img_size // self.vision.patch_size,
        )
        assert tuple(self.vision.grid_size) == expected_grid
        assert self.binding.visual_dim == self.vision.embed_dim
        assert self.binding.hidden_dim == self.articulator.embed_dim
        assert self.articulator.embed_dim == self.temporal.hidden_dim
        assert self.resampler.output_dim == self.decoder.d_model
        assert self.resampler.num_queries > 0

    def to_yaml(self, path: str | Path) -> None:
        """Save configuration to YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        d = self.to_dict()
        if _HAS_YAML:
            with open(p, "w", encoding="utf-8") as f:
                yaml.dump(d, f, sort_keys=False, default_flow_style=False)
        else:
            with open(p, "w", encoding="utf-8") as f:
                f.write(_dump_simple_yaml(d))

    @classmethod
    def from_yaml(cls, path: str | Path) -> ModelConfig:
        """Load configuration from YAML or JSON file."""
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            if _HAS_YAML:
                d = yaml.safe_load(f)
            else:
                text = f.read()
                d = _parse_simple_yaml(text)
        return cls.from_dict(d)
