import numpy as np
from voice.preprocessing.filtering import (
    select_time_window,
    resample_data,
    apply_bandpass,
    apply_notch_filter,
)
from voice.preprocessing.normalization import baseline_correction, remove_artifacts


def preprocess_eeg_dataset(
    X: np.ndarray,
    sfreq: float = 256.0,
    target_sfreq: float = 128.0,
    lowcut: float = 1.0,
    highcut: float = 40.0,
    notch_freq: float = 50.0,
    t_start: float = 1.5,
    t_end: float = 3.5,
    apply_baseline: bool = False,
    artifact_threshold: float = None,
    y: np.ndarray = None,
):
    """Executes the full EEG cleaning and filtering pipeline on raw trial data."""
    # 1. Select active time window (e.g., action period 1.5s - 3.5s)
    X_proc = select_time_window(X, t_start=t_start, t_end=t_end, fs=sfreq)

    # 2. Resample signal
    if sfreq != target_sfreq:
        X_proc = resample_data(X_proc, original_sfreq=sfreq, target_sfreq=target_sfreq)

    # 3. Apply Bandpass & Notch Filters
    X_proc = apply_bandpass(X_proc, lowcut=lowcut, highcut=highcut, fs=target_sfreq)
    X_proc = apply_notch_filter(X_proc, notch_freq=notch_freq, fs=target_sfreq)

    # 4. Baseline Correction (Optional)
    if apply_baseline:
        X_proc = baseline_correction(X_proc)

    # 5. Artifact Removal (Optional)
    if artifact_threshold is not None:
        if y is not None:
            X_proc, y = remove_artifacts(X_proc, y=y, threshold=artifact_threshold)
            return X_proc, y
        else:
            X_proc, kept_indices = remove_artifacts(X_proc, threshold=artifact_threshold)

    if y is not None:
        return X_proc, y
        
    return X_proc