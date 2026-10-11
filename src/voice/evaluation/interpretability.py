"""Model-specific EEGNet saliency and EEGConformer attention summaries."""

from __future__ import annotations

import numpy as np


def eegnet_channel_saliency(model, signals: np.ndarray, *, target_class=None, device: str | None = None) -> np.ndarray:
    """Return normalized input-gradient importance, averaged by EEG channel."""
    import torch

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device).eval()
    inputs = torch.as_tensor(np.asarray(signals, dtype=np.float32), device=selected_device)
    inputs.requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(inputs)
    targets = logits.argmax(dim=1) if target_class is None else torch.full(
        (len(inputs),), int(target_class), device=selected_device, dtype=torch.long
    )
    logits.gather(1, targets[:, None]).sum().backward()
    importance = inputs.grad.abs().mean(dim=(0, 2)).detach().cpu().numpy()
    total = importance.sum()
    return importance / total if total > 0 else importance


def conformer_attention_summary(
    model, signals: np.ndarray, *, device: str | None = None
) -> list[np.ndarray]:
    """Average attention heads and trials into one token-by-token matrix per layer."""
    import torch

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device).eval()
    inputs = torch.as_tensor(np.asarray(signals, dtype=np.float32), device=selected_device)
    with torch.no_grad():
        layer_maps = model.attention_maps(inputs)
    return [attention.mean(dim=(0, 1)).detach().cpu().numpy() for attention in layer_maps]
