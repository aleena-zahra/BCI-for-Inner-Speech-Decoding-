import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch

class EEGPreprocessor:
    """Combines DSP filtering (Bandpass, Notch, Resampling) and MNE Epoching."""

    def __init__(self, sfreq: float = 256.0, target_sfreq: float = 128.0):
        self.sfreq = sfreq
        self.target_sfreq = target_sfreq

    def apply_dsp_filters(self, X: np.ndarray, lowcut: float = 1.0, highcut: float = 40.0, notch_freq: float = 50.0) -> np.ndarray:
        """Applies Butterworth Bandpass, Notch Filter, and Resampling (From Baseline Notebook)."""
        # 1. Resample
        n_trials, n_channels, _ = X.shape
        resampled = np.array([
            mne.filter.resample(trial, down=self.sfreq / self.target_sfreq, npad="auto")
            for trial in X
        ])

        # 2. Bandpass
        nyq = 0.5 * self.target_sfreq
        b, a = butter(4, [lowcut / nyq, highcut / nyq], btype='band')
        filtered = np.zeros_like(resampled)
        for i in range(resampled.shape[0]):
            for ch in range(resampled.shape[1]):
                filtered[i, ch] = filtfilt(b, a, resampled[i, ch])

        # 3. Notch
        b_n, a_n = iirnotch(notch_freq, 30, self.target_sfreq)
        for i in range(filtered.shape[0]):
            for ch in range(filtered.shape[1]):
                filtered[i, ch] = filtfilt(b_n, a_n, filtered[i, ch])

        return filtered

    @staticmethod
    def baseline_correction(X: np.ndarray) -> np.ndarray:
        """Subtract mean across time for each channel."""
        return X - np.mean(X, axis=2, keepdims=True)