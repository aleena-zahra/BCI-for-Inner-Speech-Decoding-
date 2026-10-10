import torch
import torch.nn as nn


class EEGNet(nn.Module):
    """EEGNet architecture for EEG-based BCI classification.

    Parameters
    ----------
    n_classes : int, default=2
        Number of target classes (e.g., binary inner speech directions).
    n_channels : int, default=128
        Number of EEG electrodes/channels.
    n_timepoints : int, default=256
        Number of time samples per EEG trial.
    F1 : int, default=8
        Number of temporal filters.
    D : int, default=2
        Depth multiplier for spatial convolution.
    F2 : int, default=16
        Number of pointwise filters.
    dropout_rate : float, default=0.25
        Dropout probability.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 128,
        n_timepoints: int = 256,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout_rate: float = 0.25,
    ):
        super().__init__()

        # Block 1: Temporal Conv + Depthwise Conv (Spatial Filter)
        self.conv1 = nn.Conv2d(1, F1, (1, 64), padding=(0, 32), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise = nn.Conv2d(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out = self._forward_features(dummy)
            self.flat_dim = out.view(1, -1).shape[1]

        # Classification Dense Head
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def _forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)  # Shape: (B, 1, C, T)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = torch.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = torch.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x