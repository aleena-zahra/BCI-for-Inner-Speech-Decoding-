"""Shared neural-network layers used by the deep EEG architectures."""

from __future__ import annotations

import torch
from torch import nn


class RevIN(nn.Module):
    """Learnable affine instance normalization over each trial's time axis."""

    def __init__(self, num_channels: int, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if num_channels < 1:
            raise ValueError("num_channels must be positive")
        self.num_channels = num_channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, num_channels, 1)) if affine else None
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1)) if affine else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != self.num_channels:
            raise ValueError("RevIN expects (batch, channels, time) tensors")
        mean = x.mean(dim=-1, keepdim=True)
        variance = x.var(dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        if self.gamma is not None:
            normalized = normalized * self.gamma + self.beta
        return normalized
