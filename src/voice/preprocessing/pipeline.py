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

    def transform(self, X: np.ndarray, y: np.ndarray = None):
        """Applies the configured preprocessing steps to the input signal."""
        # 1. Select time window
        X_proc = select_time_window(X, t_start=self.t_start, t_end=self.t_end, fs=self.sfreq)

        # 2. Resample signal
        if self.sfreq != self.target_sfreq:
            X_proc = resample_data(X_proc, original_sfreq=self.sfreq, target_sfreq=self.target_sfreq)

        # 3. Apply Bandpass & Notch Filters
        X_proc = apply_bandpass(X_proc, lowcut=self.lowcut, highcut=self.highcut, fs=self.target_sfreq)
        X_proc = apply_notch_filter(X_proc, notch_freq=self.notch_freq, fs=self.target_sfreq)

        # 4. Baseline Correction
        if self.apply_baseline:
            X_proc = baseline_correction(X_proc)

        # 5. Artifact Removal
        if self.artifact_threshold is not None:
            if y is not None:
                X_proc, y = remove_artifacts(X_proc, y=y, threshold=self.artifact_threshold)
                return X_proc, y
            else:
                X_proc, kept_indices = remove_artifacts(X_proc, threshold=self.artifact_threshold)

        if y is not None:
            return X_proc, y

        return X_proc

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None):
        """Convenience method to match standard ML interfaces."""
        return self.transform(X, y=y)


# Keep the function for users who prefer functional calls without instantiation
def preprocess_eeg_dataset(X: np.ndarray, **kwargs):
    """Functional wrapper for EEGPreprocessor."""
    preprocessor = EEGPreprocessor(**kwargs)
    return preprocessor.transform(X)