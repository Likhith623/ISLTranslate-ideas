"""Strict configuration, shared by all entry points."""
from dataclasses import asdict, dataclass, field
from pathlib import Path
import math
import yaml


def default_regions():
    return [list(range(3, 24)), list(range(24, 45)), list(range(45, 63)), [0, 1, 2, 63, 64, 65]]


@dataclass
class ModelConfig:
    joints: int = 66
    anchors: list[int] = field(default_factory=lambda: [0, 1, 2])
    regions: list[list[int]] = field(default_factory=default_regions)
    rank: int = 4
    bins: int = 32
    width: int = 256
    heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 4
    ff_multiplier: int = 4
    dropout: float = 0.1
    max_target_length: int = 512
    operator: str = "pcat"
    clock: str = "learned"
    timing: bool = False
    metric_floor: float = 0.05
    anchor_min_confidence: float = 0.5
    anchor_policy: str = "first_reliable"
    progress_scales: list[int] = field(default_factory=list)
    progress_augmentation: bool = False
    detail_memory: bool = False
    detail_position_dim: int = 32
    metric_condition_limit: float = 16.0
    integration: str = "prefix"
    integration_precision: str = "float32"
    transport: str = "cayley"
    area_sketch_dim: int = 0
    area_sketch_heads: int = 1
    adaptive_step: float = 0.0
    adaptive_min_bins: int = 2
    local_clocks: bool = False
    local_bins: int = 8
    body_frame: str = "fixed"
    body_time_constant: float = .25
    depth_policy: str = "use"
    visual_backend: str = "none"
    visual_regions: int = 4
    visual_to_articulator: list[int] = field(default_factory=lambda: [0, 1, 2, 2])
    visual_dim: int = 64
    visual_grid: int = 1
    visual_memory_norm: bool = False
    allow_unanchored_rgb: bool = False
    visual_chunk_size: int = 64
    visual_frozen: bool = False
    visual_model: str = "facebook/dinov2-small"
    visual_revision: str = "main"
    visual_config: dict = field(default_factory=dict)
    text_backend: str = "byte"
    text_model: str = "google/flan-t5-base"
    text_revision: str = "main"
    text_target_lang: str = ""
    text_frozen: bool = False
    text_config: dict = field(default_factory=dict)
    generation_beams: int = 1

    def validate(self):
        for name in ("joints", "rank", "bins", "width", "heads", "encoder_layers",
                     "decoder_layers", "ff_multiplier", "max_target_length"):
            value = getattr(self, name)
            if type(value) is not int or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.max_target_length < 2 or self.width % self.heads:
            raise ValueError("target length must be >=2 and width divisible by heads")
        if len(self.anchors) != 3 or len(set(self.anchors)) != 3:
            raise ValueError("anchors must be three distinct joints: left shoulder, right shoulder, neck")
        if not self.regions or any(not r for r in self.regions):
            raise ValueError("regions must contain nonempty joint lists")
        if any(type(j) is not int or j < 0 or j >= self.joints
               for j in self.anchors + [j for r in self.regions for j in r]):
            raise ValueError("joint index outside layout")
        if self.operator not in {"pcat", "signature", "first_order", "no_transport", "temporal"}:
            raise ValueError("unknown operator ablation")
        if self.clock not in {"learned", "euclidean"}:
            raise ValueError("clock must be learned or euclidean")
        if not math.isfinite(self.metric_floor) or self.metric_floor <= 0:
            raise ValueError("metric_floor must be finite and positive")
        if not math.isfinite(self.metric_condition_limit) or self.metric_condition_limit < 1:
            raise ValueError("metric_condition_limit must be finite and >=1")
        if self.integration not in {"prefix", "clipped"} or self.integration_precision not in {"float32", "float64"}:
            raise ValueError("invalid integration backend or precision")
        if any(type(k) is not int or k < 1 for k in self.progress_scales):
            raise ValueError("progress_scales must contain positive integers")
        if len(set(self.progress_scales)) != len(self.progress_scales):
            raise ValueError("progress_scales must not contain duplicates")
        if self.progress_scales and (sorted(self.progress_scales) != self.progress_scales
                                     or self.progress_scales[-1] != self.bins):
            raise ValueError("progress_scales must increase and end at bins")
        if type(self.detail_position_dim) is not int or self.detail_position_dim < 2 or self.detail_position_dim % 2:
            raise ValueError("detail_position_dim must be a positive even integer >=2")
        if self.detail_memory and self.width < 4 * self.joints + self.detail_position_dim + 2:
            raise ValueError("detail_memory requires width >= 4*joints + detail_position_dim + 2")
        if any(type(x) is not bool for x in (self.timing, self.detail_memory, self.progress_augmentation,
                                            self.local_clocks, self.visual_frozen, self.text_frozen,
                                            self.visual_memory_norm, self.allow_unanchored_rgb)):
            raise ValueError("timing, detail_memory and progress_augmentation must be booleans")
        if not 0 <= self.dropout < 1 or not 0 < self.anchor_min_confidence <= 1:
            raise ValueError("invalid dropout or anchor confidence threshold")
        if self.anchor_policy not in {"first", "first_reliable"}:
            raise ValueError("unknown anchor policy")
        if self.transport not in {"cayley", "givens"} or self.body_frame not in {"fixed", "dual"}:
            raise ValueError("unknown transport or body frame")
        if self.depth_policy not in {"use", "ignore"}:
            raise ValueError("depth_policy must be use or ignore")
        if self.body_frame == "dual" and not self.detail_memory:
            raise ValueError("dual body frames require global detail memory to retain torso motion")
        if not math.isfinite(self.body_time_constant) or self.body_time_constant <= 0:
            raise ValueError("body_time_constant must be finite and positive")
        for name in ("area_sketch_heads", "adaptive_min_bins", "local_bins", "visual_regions", "visual_dim",
                     "visual_chunk_size", "visual_grid", "generation_beams"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        n = self.rank * len(self.regions)
        if type(self.area_sketch_dim) is not int or not 0 <= self.area_sketch_dim <= n:
            raise ValueError("area_sketch_dim must be between zero and the embedded dimension")
        if self.area_sketch_heads > 1 and not self.area_sketch_dim:
            raise ValueError("multiple area heads require a sketch dimension")
        if not math.isfinite(self.adaptive_step) or self.adaptive_step < 0:
            raise ValueError("adaptive_step must be finite and nonnegative")
        if self.adaptive_step and self.adaptive_min_bins > self.bins:
            raise ValueError("adaptive_min_bins cannot exceed bins")
        if self.visual_backend not in {"none", "cnn", "dinov2"} or self.text_backend not in {"byte", "hf"}:
            raise ValueError("unknown vision or text backend")
        if self.allow_unanchored_rgb and self.visual_backend == "none":
            raise ValueError("unanchored fallback needs actual visual observations")
        if self.visual_backend != "none" and (len(self.visual_to_articulator) != self.visual_regions or any(
                type(a) is not int or not 0 <= a < len(self.regions) for a in self.visual_to_articulator)):
            raise ValueError("visual_to_articulator must map each crop to an articulator")


@dataclass
class DataConfig:
    train: str = "data/train.jsonl"
    val: str = "data/val.jsonl"
    test: str = "data/test.jsonl"
    batch_size: int = 8
    workers: int = 0
    max_frames: int = 1500
    max_bytes: int = 510


@dataclass
class TrainConfig:
    epochs: int = 50
    lr: float = 0.0003
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    accumulation: int = 1
    warmup_steps: int = 100
    seed: int = 17
    device: str = "auto"
    amp: bool = False
    deterministic: bool = True
    output_dir: str = "runs/pcat"


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self):
        self.model.validate()
        d, t = self.data, self.train
        if any(type(x) is not int or x < 1 for x in
               (d.batch_size, d.max_frames, d.max_bytes, t.epochs, t.accumulation)):
            raise ValueError("batch, frame, byte, epoch and accumulation limits must be positive integers")
        if d.max_frames < 2 or (self.model.text_backend == "byte" and d.max_bytes + 1 > self.model.max_target_length):
            raise ValueError("max_frames must be >=2; max_bytes+1 must fit max_target_length")
        if any(type(x) is not int or x < 0 for x in (d.workers, t.warmup_steps, t.seed)):
            raise ValueError("workers, warmup_steps and seed must be nonnegative integers")
        if t.seed >= 2**32:
            raise ValueError("seed must fit NumPy's uint32 seed range")
        if any(not math.isfinite(x) for x in (t.lr, t.weight_decay, t.grad_clip, t.label_smoothing)):
            raise ValueError("training scalars must be finite")
        if t.lr <= 0 or t.weight_decay < 0 or t.grad_clip <= 0 or not 0 <= t.label_smoothing < 1:
            raise ValueError("invalid training scalars")
        if t.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("supported devices: auto, cpu, cuda, mps")
        if t.device == "mps" and self.model.integration_precision != "float32":
            raise ValueError("MPS configuration must use float32; integration explicitly runs on CPU")

    def to_dict(self):
        return asdict(self)


def from_dict(raw: dict) -> Config:
    if not isinstance(raw, dict) or set(raw) - {"model", "data", "train"}:
        raise ValueError("configuration must have only model, data, train sections")
    cfg = Config(ModelConfig(**raw.get("model", {})), DataConfig(**raw.get("data", {})),
                 TrainConfig(**raw.get("train", {})))
    cfg.validate()
    return cfg


def load_config(path: str | Path) -> Config:
    with Path(path).open(encoding="utf-8") as f:
        return from_dict(yaml.safe_load(f))
