
import torch
from voice.models import EEGConformer


def main() -> None:
    """Construct EEG-Conformer and run a forward pass."""
    model = EEGConformer(
        n_classes=2,
        n_channels=128,
        n_timepoints=256,
        d_model=64,
        n_heads=4,
        n_layers=2,
    )
    signals = torch.randn(8, 128, 256)
    logits = model(signals)
    print("EEG-Conformer output shape:", logits.shape)


if __name__ == "__main__":
    main()