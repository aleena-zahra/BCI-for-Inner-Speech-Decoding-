import numpy as np
from typing import Tuple


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