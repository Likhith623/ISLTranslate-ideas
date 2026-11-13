"""
Articulatory-Bound Spatiotemporal Transformer (AB-ST) Package.
"""

from .config import (
    ModelConfig,
    DataConfig,
    PoseConfig,
    VisionConfig,
    BindingConfig,
    ArticulatorConfig,
    TemporalConfig,
    ResamplerConfig,
    DecoderConfig,
    LossConfig,
    TrainConfig,
)
from .adapter_543_to_66 import (
    map_543_to_66,
    map_holistic_543_to_canonical_66_list,
    validate_canonical_66_structure,
    CANONICAL_66_INDICES_IN_543,
    LEFT_HAND_RANGE,
    RIGHT_HAND_RANGE,
    BODY_RANGE,
    FACE_RANGE,
)
from .dominance import (
    TorsoMidlineNormalizer,
    KinematicDominanceRouter,
    compute_torso_midline_pure,
    compute_hand_energies_pure,
)
from .vision import DINOv2SpatialExtractor, FeatureCacheManager
from .binding import DeformablePatchBinder
from .articulators import FactoredArticulatorGrouper
from .temporal import AxialSpatioTemporalConformer
from .resampler import ArticulatoryQFormer
from .decoder import TranslationDecoderAdapter
from .model import ArticulatoryTransformer
from .losses import CompositeABSTLoss
from .data import SignTranslationDataset, collate_abst_batch

__version__ = "1.0.0"
__all__ = [
    "ModelConfig",
    "DataConfig",
    "PoseConfig",
    "VisionConfig",
    "BindingConfig",
    "ArticulatorConfig",
    "TemporalConfig",
    "ResamplerConfig",
    "DecoderConfig",
    "LossConfig",
    "TrainConfig",
    "map_543_to_66",
    "map_holistic_543_to_canonical_66_list",
    "validate_canonical_66_structure",
    "CANONICAL_66_INDICES_IN_543",
    "LEFT_HAND_RANGE",
    "RIGHT_HAND_RANGE",
    "BODY_RANGE",
    "FACE_RANGE",
    "TorsoMidlineNormalizer",
    "KinematicDominanceRouter",
    "compute_torso_midline_pure",
    "compute_hand_energies_pure",
    "DINOv2SpatialExtractor",
    "FeatureCacheManager",
    "DeformablePatchBinder",
    "FactoredArticulatorGrouper",
    "AxialSpatioTemporalConformer",
    "ArticulatoryQFormer",
    "TranslationDecoderAdapter",
    "ArticulatoryTransformer",
    "CompositeABSTLoss",
    "SignTranslationDataset",
    "collate_abst_batch",
]

