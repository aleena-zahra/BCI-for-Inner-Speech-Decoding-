import torch
from voice.models import EEGConformer

# Initialize model
model = EEGConformer(
    n_classes=2,
    n_channels=128,
    n_timepoints=256,
    d_model=64,
    n_heads=4,
    n_layers=2,
)

# Dummy trial tensor: [batch_size=8, channels=128, timepoints=256]
x_dummy = torch.randn(8, 128, 256)
logits = model(x_dummy)

print("EEG-Conformer Output Shape:", logits.shape)  # torch.Size([8, 2])