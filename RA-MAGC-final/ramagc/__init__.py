"""RA-MAGC: Reliability-Aware Multi-Rate Articulatory Graph-Conformer.

Research prototype for continuous, gloss-free Indian Sign Language translation.
"""

from .config import (
    ModelConfig,
    DataConfig,
    HandGraphConfig,
    BodyGraphConfig,
    FaceSetConfig,
    RegionFusionConfig,
    MultiScaleTemporalConfig,
    ConformerConfig,
)
from .dataset import (
    StructuredPoseSample,
    RAMAGCDataset,
    RAMAGCCollator,
    extract_structured_sample,
    compute_torso_reference,
    normalize_with_torso,
    compute_velocity,
    HAND_EDGES,
    BODY_EDGES,
)
from .graph_layers import (
    EdgeAwareGraphAttention,
    HandGraphEncoder,
    UpperBodyGraphEncoder,
    FaceSetEncoder,
)
from .fusion import RegionInteraction, ReliabilityGate
from .temporal import (
    length_mask,
    MultiScaleTemporalBlock,
    PairwiseAggregator,
    RelativeElapsedTimeBias,
    ConformerBlock,
    FineDetailRetrieval,
)
from .model import RAMAGCVisualEncoder, RAMAGCTranslator, compute_quality_descriptors

__all__ = [
    "ModelConfig",
    "DataConfig",
    "HandGraphConfig",
    "BodyGraphConfig",
    "FaceSetConfig",
    "RegionFusionConfig",
    "MultiScaleTemporalConfig",
    "ConformerConfig",
    "StructuredPoseSample",
    "RAMAGCDataset",
    "RAMAGCCollator",
    "extract_structured_sample",
    "compute_torso_reference",
    "normalize_with_torso",
    "compute_velocity",
    "HAND_EDGES",
    "BODY_EDGES",
    "EdgeAwareGraphAttention",
    "HandGraphEncoder",
    "UpperBodyGraphEncoder",
    "FaceSetEncoder",
    "RegionInteraction",
    "ReliabilityGate",
    "length_mask",
    "MultiScaleTemporalBlock",
    "PairwiseAggregator",
    "RelativeElapsedTimeBias",
    "ConformerBlock",
    "FineDetailRetrieval",
    "RAMAGCVisualEncoder",
    "RAMAGCTranslator",
    "compute_quality_descriptors",
]
