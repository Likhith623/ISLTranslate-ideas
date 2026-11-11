"""ISL-MEGA: multimodal, multi-expert ISL-to-English research architecture."""

from .config import ISLMEGAConfig, load_config
from .model import ISLMEGA, build_model

__all__ = ["ISLMEGA", "ISLMEGAConfig", "build_model", "load_config"]
__version__ = "0.2.0"
