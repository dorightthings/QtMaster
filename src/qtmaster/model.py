"""Market-conditioned factor gating and factor-level lead-lag alignment.

One model only: the last observed market state conditions the lag correction.
Inputs contain 158 stock factors and 63 market features, never the label.
"""

from dataclasses import asdict, dataclass
import math
from types import SimpleNamespace
from typing import Any, Dict, Optional

import torch
from torch import nn

from .lead_lag import LeadLagAlignedStarCore
from .vendor.tqnet import Model as TQNetCore


LOOKBACK = 8
STOCK_FEATURES = 158
MARKET_FEATURES = 63
MODEL_INPUT_FEATURES = STOCK_FEATURES + MARKET_FEATURES
EXPECTED_PARAMETER_COUNT = 213_279


@dataclass(frozen=True)
class ModelConfig:
    """Current model defaults; cycle is 7 for CSI300 and 3 for CSI800.

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

        self.residual_scale = 0.1
        # Preserve the training RNG and start with zero conditional correction.
        with torch.random.fork_rng(devices=[]):
            self.tau_head = nn.Sequential(
                nn.Linear(MARKET_FEATURES, 32),
                nn.GELU(),
                nn.Linear(32, STOCK_FEATURES),
            )
            nn.init.zeros_(self.tau_head[-1].weight)
            nn.init.zeros_(self.tau_head[-1].bias)

    def compute_gate(self, market: torch.Tensor) -> torch.Tensor:
        if market.ndim != 3 or tuple(market.shape[1:]) != (LOOKBACK, MARKET_FEATURES):
            raise ValueError("market input must have shape [B,8,63]")
        return 2.0 * torch.sigmoid(torch.matmul(market, self.market_gate_weight))

    @staticmethod
    def _check_features(features: torch.Tensor) -> None:
        if features.ndim != 3 or tuple(features.shape[1:]) != (LOOKBACK, MODEL_INPUT_FEATURES):
            raise ValueError("model input must have shape [B,8,221] (158 stock + 63 market; no label)")

    def compute_tau(self, features: torch.Tensor) -> torch.Tensor:
        """Return [B,158] relative offsets using the final observed market day."""
        self._check_features(features)
        condition = features[:, -1, STOCK_FEATURES:MODEL_INPUT_FEATURES]
        correction = self.tau_head(condition)
        return self.lla_block.max_lag * torch.tanh(
            self.lla_block.raw_tau.unsqueeze(0) + self.residual_scale * correction
        )

    def shift_history(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        """Original padded FFT/crop operator, supporting [C] or [B,C] lags.

        No zero-lag shortcut is used: it would change both numerical behavior
        and gradients of the alignment operator. Inverse shifting reuses
        this same operator and therefore retains its original crop behavior.
        """
        if x.ndim != 3 or tuple(x.shape[1:]) != (LOOKBACK, STOCK_FEATURES):
            raise ValueError("LLA input must have shape [B,8,158]")
        if tau.ndim == 1 and tuple(tau.shape) == (STOCK_FEATURES,):
            tau_for_phase = tau.unsqueeze(0).unsqueeze(0)
        elif tau.ndim == 2 and tuple(tau.shape) == (x.shape[0], STOCK_FEATURES):
            tau_for_phase = tau.unsqueeze(1)
        else:
            raise ValueError("tau must have shape [158] or [B,158]")
        tau_for_phase = tau_for_phase.to(device=x.device, dtype=x.dtype)
        spectrum = torch.fft.rfft(x, n=self.lla_block.fft_len, dim=1)
        frequency = self.lla_block.frequency_index.to(
            device=spectrum.device, dtype=spectrum.real.dtype,
        ).view(1, -1, 1)
        phase = -2.0 * math.pi * frequency * tau_for_phase / self.lla_block.fft_len
        rotation = torch.polar(torch.ones_like(phase), phase)
        shifted = spectrum * rotation
        return torch.fft.irfft(shifted, n=self.lla_block.fft_len, dim=1)[:, :self.lla_block.seq_len]

    def align_history(
        self, x: torch.Tensor, tau: torch.Tensor, return_aux: bool = False,
    ) -> Any:
        """Reuse the original fusion/residual modules with one supplied tau."""
        aligned = self.shift_history(x, tau)
        daily_core = self.lla_block.gen_core(aligned)
        aligned_proposal = self.lla_block.fuse(torch.cat((aligned, daily_core), dim=-1))
        proposal = self.shift_history(aligned_proposal, -tau)
        gate = self.lla_block.residual_gate().to(device=x.device, dtype=x.dtype).view(1, 1, -1)
        output = x + gate * (proposal - x)
        if tuple(output.shape) != tuple(x.shape):
            raise RuntimeError("LLA output violated shape contract")
        if not return_aux:
            return output
        diagnostics: Dict[str, torch.Tensor] = {
            "tau": tau,
            "residual_gate": gate,
            "aligned": aligned,
            "daily_core": daily_core,
            "proposal": proposal,
        }
        return output, diagnostics

    def forward(
        self, features: torch.Tensor, cycle_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the unchanged [B,1,1] stock-score interface."""
        self._check_features(features)
        if cycle_index is None:
            cycle_index = torch.zeros(features.shape[0], dtype=torch.long, device=features.device)
        stock = features[:, :, :STOCK_FEATURES]
        market = features[:, :, STOCK_FEATURES:MODEL_INPUT_FEATURES]
        gated_stock = stock * self.compute_gate(market)
        tau = self.compute_tau(features).to(device=gated_stock.device, dtype=gated_stock.dtype)
        refined_stock = self.align_history(gated_stock, tau)
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
