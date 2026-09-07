"""Market-conditioned lead-lag alignment for stock prediction."""

from .model import MarketGateLLAPureMLP, ModelConfig, build_model

__all__ = ["MarketGateLLAPureMLP", "ModelConfig", "build_model"]
__version__ = "0.2.0"
