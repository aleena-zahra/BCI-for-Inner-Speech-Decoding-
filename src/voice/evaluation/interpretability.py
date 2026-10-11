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
    model, signals: np.ndarray, *, device: str | None = None, head: int | None = None,
    batch_size: int = 8,
) -> list[np.ndarray]:
    """Average trials and either all heads or one head, in bounded batches."""
    import torch

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device).eval()
    values = np.asarray(signals, dtype=np.float32)
    if values.ndim != 3 or not len(values) or not np.isfinite(values).all() or batch_size < 1:
        raise ValueError("Attention requires finite non-empty trial/channel/time data and positive batch size")
    summaries = None
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            inputs = torch.as_tensor(values[start:start + batch_size], device=selected_device)
            layer_maps = model.attention_maps(inputs)
            batch_maps = []
            for attention in layer_maps:
                if head is not None and not 0 <= head < attention.shape[1]:
                    raise ValueError("Attention head index is outside the model's head count")
                averaged = attention.mean(dim=(0, 1)) if head is None else attention[:, head].mean(dim=0)
                batch_maps.append(averaged.cpu().numpy() * len(inputs))
            if summaries is None:
                summaries = batch_maps
            else:
                summaries = [total + batch for total, batch in zip(summaries, batch_maps)]
    return [total / len(values) for total in summaries]
