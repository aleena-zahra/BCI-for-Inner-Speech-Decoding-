import numpy as np
from voice.preprocessing.filtering import (
    select_time_window,
    resample_data,
    apply_bandpass,
    apply_notch_filter,
)
from voice.preprocessing.normalization import baseline_correction, remove_artifacts


class EEGPreprocessor:
    """Configurable preprocessing pipeline for raw EEG trial data."""

    def __init__(
        self,
        sfreq: float = 256.0,
        target_sfreq: float = 128.0,
        lowcut: float = 1.0,
        highcut: float = 40.0,
        notch_freq: float = 50.0,
        t_start: float = 1.5,
        t_end: float = 3.5,
        apply_baseline: bool = False,
        artifact_threshold: float = None,
        common_average_reference: bool = False,
    ):
        self.sfreq = sfreq
        self.target_sfreq = target_sfreq
        self.lowcut = lowcut
        self.highcut = highcut
        self.notch_freq = notch_freq
        self.t_start = t_start
        self.t_end = t_end
        self.apply_baseline = apply_baseline
        self.artifact_threshold = artifact_threshold
        self.common_average_reference = common_average_reference

    def transform_with_indices(self, X: np.ndarray, y: np.ndarray = None):
        """Transform trials and return retained source indices for metadata alignment."""
        X = np.asarray(X)
        if X.ndim != 3 or X.shape[0] == 0:
            raise ValueError("EEG data must be a non-empty (trials, channels, time) array")
        if y is not None and len(y) != len(X):
            raise ValueError("Labels must have one entry per EEG trial")
        start = int(round(self.t_start * self.sfreq))
        stop = int(round(self.t_end * self.sfreq))
        if start < 0 or stop <= start or stop > X.shape[-1]:
            raise ValueError(
                f"Requested time window {self.t_start:g}-{self.t_end:g}s is outside "
                f"the {X.shape[-1] / self.sfreq:g}s trial"
            )
        # 1. Select time window
        X_proc = select_time_window(X, t_start=self.t_start, t_end=self.t_end, fs=self.sfreq)

        # 2. Resample signal
        if self.sfreq != self.target_sfreq:
            X_proc = resample_data(X_proc, original_sfreq=self.sfreq, target_sfreq=self.target_sfreq)

        # 3. Apply Bandpass & Notch Filters
        X_proc = apply_bandpass(X_proc, lowcut=self.lowcut, highcut=self.highcut, fs=self.target_sfreq)
        if self.notch_freq < self.target_sfreq / 2:
            X_proc = apply_notch_filter(X_proc, notch_freq=self.notch_freq, fs=self.target_sfreq)

        if self.common_average_reference:
            X_proc = X_proc - X_proc.mean(axis=1, keepdims=True)

        # 4. Baseline Correction
        if self.apply_baseline:
            X_proc = baseline_correction(X_proc)

        # 5. Artifact Removal
        kept_indices = np.arange(len(X_proc))
        if self.artifact_threshold is not None:
            X_proc, local_indices = remove_artifacts(
                X_proc, threshold=self.artifact_threshold
            )
            kept_indices = kept_indices[local_indices]
        if y is not None:
            return X_proc, np.asarray(y)[kept_indices], kept_indices
        return X_proc, kept_indices

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """Apply preprocessing, retaining the existing X or (X, y) API."""
        result = self.transform_with_indices(X, y)
        return result[:2] if y is not None else result[0]

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        """Convenience method to match standard ML interfaces."""
        return self.transform(X, y=y)


# Keep the function for users who prefer functional calls without instantiation
def preprocess_eeg_dataset(X: np.ndarray, **kwargs):
    """Functional wrapper for EEGPreprocessor."""
    preprocessor = EEGPreprocessor(**kwargs)
    return preprocessor.transform(X)
