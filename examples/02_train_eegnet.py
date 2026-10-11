import torch
from voice.models import EEGNet


def main() -> None:
	"""Construct EEGNet and run a forward pass on EEG-shaped input."""
	model = EEGNet(n_classes=2, n_channels=128, n_timepoints=256)
	signals = torch.randn(16, 128, 256)
	outputs = model(signals)
	print("Output shape:", outputs.shape)


if __name__ == "__main__":
	main()