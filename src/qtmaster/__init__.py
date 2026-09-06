"""Portable MarketGate + LLA + PureMLP stock-prediction model."""

from .model import MarketGateLLAPureMLP, ModelConfig, build_model

__all__ = ["MarketGateLLAPureMLP", "ModelConfig", "build_model"]
__version__ = "0.1.0"
