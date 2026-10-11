"""Reusable temporal patch and spectral feature utilities."""

from __future__ import annotations

import numpy as np
from scipy import signal


class EEGPatchTokenizer:
    """Build overlapping temporal patches from ``(trials, channels, time)`` EEG."""

    def __init__(self, patch_len: int = 25, stride: int = 6, flatten_channels: bool = True):
        if patch_len < 1 or stride < 1 or stride > patch_len:
            raise ValueError("patch_len and stride must be positive, and stride cannot exceed patch_len")
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.flatten_channels = flatten_channels

    def tokenize(self, X: np.ndarray) -> tuple[np.ndarray, int]:
        X = np.asarray(X)
        if X.ndim != 3:
            raise ValueError("Expected EEG with shape (trials, channels, time)")
        if X.shape[-1] < self.patch_len:
            raise ValueError("Trial time dimension is shorter than patch_len")
        windows = np.lib.stride_tricks.sliding_window_view(X, self.patch_len, axis=-1)
        patches = windows[:, :, :: self.stride, :].transpose(0, 2, 1, 3)
        if self.flatten_channels:
            patches = patches.reshape(len(X), patches.shape[1], -1)
        return patches, patches.shape[1]

    def get_token_metadata(self, total_timepoints: int, sfreq: float = 250.0) -> dict:
        if sfreq <= 0 or total_timepoints < self.patch_len:
            raise ValueError("Sampling rate must be positive and timepoints must fit one patch")
        count = (total_timepoints - self.patch_len) // self.stride + 1
        return {
            "n_patches": count,
            "patch_len_samples": self.patch_len,
            "stride_samples": self.stride,
            "patch_duration_ms": 1000 * self.patch_len / sfreq,
            "stride_duration_ms": 1000 * self.stride / sfreq,
            "patch_centers_s": [
                (index * self.stride + (self.patch_len - 1) / 2) / sfreq
                for index in range(count)
            ],
        }


class SpectralFeatureExtractor:
    """Compute Welch power summaries for standard EEG frequency bands."""

    BANDS = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 100.0),
    }

    def __init__(self, sfreq: float = 250.0, nperseg: int = 256):
        if sfreq <= 0 or nperseg < 2:
            raise ValueError("sfreq must be positive and nperseg at least 2")
        self.sfreq = float(sfreq)
        self.nperseg = int(nperseg)

    def compute_band_powers(self, X: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
        X = np.asarray(X)
        if X.ndim != 3 or not np.isfinite(X).all():
            raise ValueError("Expected finite EEG with shape (trials, channels, time)")
        frequencies, psd = signal.welch(
            X, fs=self.sfreq, nperseg=min(X.shape[-1], self.nperseg), axis=-1
        )
        powers = {}
        nyquist = self.sfreq / 2
        for name, (low, high) in self.BANDS.items():
            mask = (frequencies >= low) & (frequencies <= min(high, nyquist))
            if not mask.any():
                powers[name] = np.zeros(X.shape[:2], dtype=np.float64)
            else:
                powers[name] = np.mean(psd[..., mask], axis=-1)
        return powers, frequencies, psd

    def compute_trial_spectrogram(
        self, trial_signal: np.ndarray, nperseg: int = 32, noverlap: int = 28
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        values = np.asarray(trial_signal)
        if values.ndim != 1 or len(values) < 2:
            raise ValueError("A spectrogram requires a one-dimensional signal with at least two samples")
        segment = min(nperseg, len(values))
        overlap = min(noverlap, segment - 1)
        frequencies, times, transform = signal.stft(
            values, fs=self.sfreq, nperseg=segment, noverlap=overlap
        )
        return frequencies, times, np.abs(transform)
