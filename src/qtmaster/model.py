"""The current mainline: MarketGate -> LLA -> PureMLP -> scalar readout.

The model accepts only the 221 input features, never the label column.
Checkpoint names and construction order preserve the original mainline.
"""

from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn

from .lead_lag import LeadLagAlignedStarCore
from .vendor.tqnet import Model as TQNetCore


LOOKBACK = 8
STOCK_FEATURES = 158
MARKET_FEATURES = 63
MODEL_INPUT_FEATURES = STOCK_FEATURES + MARKET_FEATURES
EXPECTED_PARAMETER_COUNT = 206_017


@dataclass(frozen=True)
class ModelConfig:
    """Frozen mainline defaults; cycle is 7 for CSI300 and 3 for CSI800.

    Cycle remains a compatibility field: temporal queries and channel
    attention are disabled, so it does not participate in this forward pass.
    """

    seq_len: int = LOOKBACK
    pred_len: int = 1
    enc_in: int = STOCK_FEATURES
    cycle: int = 7
    model_type: str = "mlp"
    d_model: int = 256
    dropout: float = 0.0
    use_revin: int = 0


class MarketGateLLAPureMLP(nn.Module):
    """Predict one future-return score per stock from an observed window."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        if (config.seq_len, config.pred_len, config.enc_in, config.d_model) != (8, 1, 158, 256):
            raise ValueError("mainline requires seq_len=8, pred_len=1, enc_in=158, d_model=256")
        if config.model_type != "mlp" or config.dropout != 0.0 or config.use_revin != 0:
            raise ValueError("mainline requires model_type='mlp', dropout=0, use_revin=0")
        if not isinstance(config.cycle, int) or config.cycle <= 0:
            raise ValueError("cycle must be a positive integer")
        self.config = config

        # Construct the complete upstream core before removing disabled parts.
        # Its MHA initialization consumes RNG; skipping it would silently change
        # every subsequent PureMLP/readout initial tensor for the same seed.
        self.core = TQNetCore(SimpleNamespace(**asdict(config)))
        self.core.channelAggregator.num_heads = 1
        self.core.channelAggregator.head_dim = LOOKBACK
        self.readout = nn.Linear(STOCK_FEATURES, 1, bias=False)
        with torch.no_grad():
            self.readout.weight.fill_(1.0 / STOCK_FEATURES)
        self.core.use_tq = False
        self.core.channel_aggre = False
        del self.core.temporalQuery
        del self.core.channelAggregator

        # Zero-initialized direct Parameter: exact identity gate and no RNG use.
        self.market_gate_weight = nn.Parameter(torch.zeros(MARKET_FEATURES, STOCK_FEATURES))
        with torch.random.fork_rng(devices=[]):
            self.lla_block = LeadLagAlignedStarCore(
                channels=STOCK_FEATURES,
                seq_len=LOOKBACK,
                d_core=32,
                max_lag=3.0,
                fft_len=16,
                residual_init=0.1,
            )

    def compute_gate(self, market: torch.Tensor) -> torch.Tensor:
        if market.ndim != 3 or tuple(market.shape[1:]) != (LOOKBACK, MARKET_FEATURES):
            raise ValueError("market input must have shape [B,8,63]")
        return 2.0 * torch.sigmoid(torch.matmul(market, self.market_gate_weight))

    def forward(
        self, features: torch.Tensor, cycle_index: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Return [B,1,1] scores from [B,8,221] historical features.

        The optional cycle_index is retained for old training-loop compatibility;
        the PureMLP arm does not use its values.
        """
        if features.ndim != 3 or tuple(features.shape[1:]) != (LOOKBACK, MODEL_INPUT_FEATURES):
            raise ValueError("model input must have shape [B,8,221] (158 stock + 63 market; no label)")
        if cycle_index is None:
            cycle_index = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        stock = features[:, :, :STOCK_FEATURES]
        market = features[:, :, STOCK_FEATURES:MODEL_INPUT_FEATURES]
        gated_stock = stock * self.compute_gate(market)
        refined_stock = self.lla_block(gated_stock)
        core_output = self.core(refined_stock, cycle_index.to(dtype=torch.long))
        if tuple(core_output.shape) != (features.shape[0], 1, STOCK_FEATURES):
            raise RuntimeError("PureMLP output violated [B,1,158] shape contract")
        output = self.readout(core_output)
        if tuple(output.shape) != (features.shape[0], 1, 1):
            raise RuntimeError("scalar readout violated [B,1,1] shape contract")
        return output


def build_model(config: Optional[ModelConfig] = None) -> MarketGateLLAPureMLP:
    """Build the mainline on CPU; callers choose seed, dtype and device."""
    model = MarketGateLLAPureMLP(config if config is not None else ModelConfig())
    count = sum(parameter.numel() for parameter in model.parameters())
    if count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError("mainline parameter count changed: %d != %d" % (count, EXPECTED_PARAMETER_COUNT))
    return model
