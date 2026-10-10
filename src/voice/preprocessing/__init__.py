"""EEG data preprocessing module."""

from voice.preprocessing.filtering import (
    select_time_window,
    resample_data,
    apply_bandpass,
    apply_notch_filter,
)
from voice.preprocessing.normalization import (
    baseline_correction,
    remove_artifacts,
)
from voice.preprocessing.pipeline import preprocess_eeg_dataset

__all__ = [
    "select_time_window",
    "resample_data",
    "apply_bandpass",
    "apply_notch_filter",
    "baseline_correction",
    "remove_artifacts",
    "preprocess_eeg_dataset",
]