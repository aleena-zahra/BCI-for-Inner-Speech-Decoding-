"""EEGNet baseline with a spatially constrained depthwise convolution."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from voice.models.layers import RevIN


class Conv2dWithConstraint(nn.Conv2d):
    """Apply the EEGNet max-norm constraint before each forward pass."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm is not None:
            with torch.no_grad():
                self.weight.copy_(torch.renorm(self.weight, p=2, dim=0, maxnorm=self.max_norm))
        return super().forward(x)


class EEGNet(nn.Module):
    """Compact temporal/depthwise-spatial EEG classifier.

    Inputs use ``(batch, channels, time)`` and logits use zero-based classes.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 128,
        n_timepoints: int = 256,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        kernel_len: int = 64,
        dropout_rate: float = 0.25,
        use_revin: bool = False,
    ):
        super().__init__()
        if min(n_classes, n_channels, n_timepoints) < 1:
            raise ValueError("Class, channel, and time dimensions must be positive")
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.revin = RevIN(n_channels) if use_revin else nn.Identity()
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)
        with torch.no_grad():
            feature_map = self._forward_features(torch.zeros(1, n_channels, n_timepoints))
        self.feature_dim = int(feature_map[0].numel())
        self.classifier = nn.Linear(self.feature_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = self.revin(x)
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_channels:
            raise ValueError("EEGNet expects (batch, channels, time) EEG tensors")
        x = self.drop1(self.pool1(F.elu(self.bn2(self.depthwise(self.bn1(self.conv1(x)))))))
        x = self.separable_depthwise(x)
        x = self.drop2(self.pool2(F.elu(self.bn3(self.separable_pointwise(x)))))
        return x

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Return the penultimate representation used for latent-space analysis."""
        return self._forward_features(x).flatten(start_dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
