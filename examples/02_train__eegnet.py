import torch
import torch.nn as nn
from voice.models import EEGNet

# Initialize model
model = EEGNet(n_classes=2, n_channels=128, n_timepoints=256)

# Create dummy trial batch: [batch_size, n_channels, n_timepoints]
x_dummy = torch.randn(16, 128, 256)
outputs = model(x_dummy)

print("Output shape:", outputs.shape)  # torch.Size([16, 2])