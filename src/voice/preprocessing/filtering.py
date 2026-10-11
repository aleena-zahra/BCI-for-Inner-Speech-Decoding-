import numpy as np
from fractions import Fraction
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


def select_time_window(X: np.ndarray, t_start: float, t_end: float, fs: float) -> np.ndarray:
    """Extracts a specific time slice from EEG trials [trials x channels x time]."""
    start_idx = int(t_start * fs)
    end_idx = int(t_end * fs)
    return X[:, :, start_idx:end_idx]


def resample_data(X: np.ndarray, original_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resamples EEG trials to a target sampling frequency."""
    ratio = Fraction(target_sfreq / original_sfreq).limit_denominator(1000)
    return resample_poly(
        X, up=ratio.numerator, down=ratio.denominator, axis=-1
    )


def apply_bandpass(X: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Applies a Butterworth bandpass filter across all trials and channels."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    return filtfilt(b, a, X, axis=-1)


def apply_notch_filter(X: np.ndarray, notch_freq: float, fs: float, quality: float = 30.0) -> np.ndarray:
    """Applies an IIR notch filter to remove powerline noise (e.g., 50Hz/60Hz)."""
    b, a = iirnotch(notch_freq, quality, fs)

    return filtfilt(b, a, X, axis=-1)
