"""RA-MAGC (Reliability-Aware Multi-Rate Articulatory Graph-Conformer) package.

Research architecture prototype for continuous Indian Sign Language Translation.
"""

try:
    from .config import (
        ModelConfig,
        DataConfig,
        HandGraphConfig,
        BodyGraphConfig,
        FaceSetConfig,
        RegionFusionConfig,
        MultiScaleTemporalConfig,
        ConformerConfig
    )
    from .graph_layers import HandGraphEncoder, UpperBodyGraphEncoder, FaceSetEncoder
    from .fusion import RegionInteraction, ReliabilityGate
    from .temporal import MultiScaleTemporalBlock, PairwiseAggregator, ConformerBlock, FineDetailRetrieval
    from .model import RAMAGCVisualEncoder, RAMAGCTranslator
    from .dataset import RAMAGCCollator, RAMAGCDataset, StructuredPoseSample, HAND_EDGES, BODY_EDGES
except (ImportError, ValueError):
    from config import (
        ModelConfig,
        DataConfig,
        HandGraphConfig,
        BodyGraphConfig,
        FaceSetConfig,
        RegionFusionConfig,
        MultiScaleTemporalConfig,
        ConformerConfig
    )
    from graph_layers import HandGraphEncoder, UpperBodyGraphEncoder, FaceSetEncoder
    from fusion import RegionInteraction, ReliabilityGate
    from temporal import MultiScaleTemporalBlock, PairwiseAggregator, ConformerBlock, FineDetailRetrieval
    from model import RAMAGCVisualEncoder, RAMAGCTranslator
    from dataset import RAMAGCCollator, RAMAGCDataset, StructuredPoseSample, HAND_EDGES, BODY_EDGES

__all__ = [
    "ModelConfig",
    "DataConfig",
    "HandGraphConfig",
    "BodyGraphConfig",
    "FaceSetConfig",
    "RegionFusionConfig",
    "MultiScaleTemporalConfig",
    "ConformerConfig",
    "HandGraphEncoder",
    "UpperBodyGraphEncoder",
    "FaceSetEncoder",
    "RegionInteraction",
    "ReliabilityGate",
    "MultiScaleTemporalBlock",
    "PairwiseAggregator",
    "ConformerBlock",
    "FineDetailRetrieval",
    "RAMAGCVisualEncoder",
    "RAMAGCTranslator",
    "RAMAGCCollator",
    "RAMAGCDataset",
    "StructuredPoseSample",
    "HAND_EDGES",
    "BODY_EDGES",
]

