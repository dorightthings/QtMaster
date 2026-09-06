"""Frozen short-window Lead-Lag Aligned Star Core for the stock experiment.

Adapted from ``LeadLagStarCore`` in
``Module-TQnet/模块/claude设计模块/new_ts_modules.py``.  The adaptation removes
batch-dependent cross-correlation/argmax, bounds the learned lag to three
trading days, zero-pads the length-eight window to 16 before every FFT, keeps
the core time-resolved, and uses the original 158-vector ``gate`` parameter as
the bounded residual strength.  Consequently the complete removable module
keeps the source class's exact 61,760-parameter budget for C=158, d_core=32.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
from torch import nn


class LeadLagAlignedStarCore(nn.Module):
    """Learn continuous per-factor lags and a time-resolved shared core."""

    def __init__(
        self,
        channels: int = 158,
        seq_len: int = 8,
        d_core: int = 32,
        max_lag: float = 3.0,
        fft_len: int = 16,
        residual_init: float = 0.1,
    ) -> None:
        super().__init__()
        if channels <= 0 or seq_len <= 0 or d_core <= 0:
            raise ValueError("channels, seq_len and d_core must be positive")
        if fft_len < seq_len + math.ceil(max_lag):
            raise ValueError("fft_len must leave explicit zero-padding for max_lag")
        if not 0.0 < residual_init < 1.0:
            raise ValueError("residual_init must be strictly between zero and one")

        self.channels = int(channels)
        self.seq_len = int(seq_len)
        self.d_core = int(d_core)
        self.max_lag = float(max_lag)
        self.fft_len = int(fft_len)

        # raw_tau replaces the source class's unconstrained tau.  ``gate`` is
        # retained as the source class's second C-vector but now has one clean
        # role: bounded per-factor residual strength.
        self.raw_tau = nn.Parameter(torch.zeros(self.channels))
        residual_logit = math.log(residual_init / (1.0 - residual_init))
        self.gate = nn.Parameter(torch.full((self.channels,), residual_logit))

        self.gen_core = nn.Sequential(
            nn.Linear(self.channels, self.d_core),
            nn.GELU(),
            nn.Linear(self.d_core, self.d_core),
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.channels + self.d_core, self.channels),
            nn.GELU(),
            nn.Linear(self.channels, self.channels),
        )
        self.register_buffer(
            "frequency_index",
            torch.arange(self.fft_len // 2 + 1, dtype=torch.float32),
        )

    def bounded_tau(self) -> torch.Tensor:
        return self.max_lag * torch.tanh(self.raw_tau)

    def residual_gate(self) -> torch.Tensor:
        return torch.sigmoid(self.gate)

    def _phase_shift(self, spectrum: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        if spectrum.ndim != 3:
            raise RuntimeError("spectrum must be [B,F,C]")
        frequency = self.frequency_index.to(
            device=spectrum.device, dtype=spectrum.real.dtype
        ).view(1, -1, 1)
        phase = -2.0 * math.pi * frequency * tau.view(1, 1, -1) / self.fft_len
        rotation = torch.polar(torch.ones_like(phase), phase)
        return spectrum * rotation

    def _shift_history(self, x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        # n=16 inserts eight zeros after the observed history.  Cropping the
        # first eight locations prevents the direct length-eight circular
        # roll used by the source implementation.
        spectrum = torch.fft.rfft(x, n=self.fft_len, dim=1)
        shifted = self._phase_shift(spectrum, tau)
        return torch.fft.irfft(shifted, n=self.fft_len, dim=1)[:, : self.seq_len]

    def forward(self, x: torch.Tensor, return_aux: bool = False) -> Any:
        expected: Tuple[int, int] = (self.seq_len, self.channels)
        if x.ndim != 3 or tuple(x.shape[1:]) != expected:
            raise RuntimeError("LLA input must be [B,%d,%d], got %r" % (
                self.seq_len, self.channels, tuple(x.shape)
            ))
        tau = self.bounded_tau().to(device=x.device, dtype=x.dtype)
        aligned = self._shift_history(x, tau)

        # Unlike the source version, no mean over the time dimension: each of
        # the eight observed days keeps its own d_core=32 shared representation.
        daily_core = self.gen_core(aligned)
        aligned_proposal = self.fuse(torch.cat((aligned, daily_core), dim=-1))
        proposal = self._shift_history(aligned_proposal, -tau)

        gate = self.residual_gate().to(device=x.device, dtype=x.dtype).view(1, 1, -1)
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
