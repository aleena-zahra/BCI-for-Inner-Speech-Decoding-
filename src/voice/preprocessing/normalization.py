import numpy as np
from typing import Tuple


def zscore_normalize(X: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    """Normalize each trial/channel along time, safely handling constant channels."""
    values = np.asarray(X, dtype=np.float32)
    if values.ndim != 3 or not np.isfinite(values).all() or epsilon <= 0:
        raise ValueError("Normalization requires finite trial/channel/time data and positive epsilon")
    mean = values.mean(axis=-1, keepdims=True)
    std = values.std(axis=-1, keepdims=True)
    return (values - mean) / np.maximum(std, epsilon)


def baseline_correction(X: np.ndarray) -> np.ndarray:
    """Subtracts the mean time signal for each channel across each trial."""
    baseline = np.mean(X, axis=2, keepdims=True)
    return X - baseline


def remove_artifacts(X: np.ndarray, y: np.ndarray = None, threshold: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
    """Removes trials where any channel exceeds an absolute amplitude threshold."""
    keep_trials = []
    for i in range(X.shape[0]):
        if np.max(np.abs(X[i])) < threshold:
            keep_trials.append(i)

    X_clean = X[keep_trials]
    if y is not None:
        return X_clean, y[keep_trials]
    
    return X_clean, np.array(keep_trials)
