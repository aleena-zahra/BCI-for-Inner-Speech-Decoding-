import numpy as np
import mne
from scipy.signal import butter, filtfilt, iirnotch


def select_time_window(X: np.ndarray, t_start: float, t_end: float, fs: float) -> np.ndarray:
    """Extracts a specific time slice from EEG trials [trials x channels x time]."""
    start_idx = int(t_start * fs)
    end_idx = int(t_end * fs)
    return X[:, :, start_idx:end_idx]


def resample_data(X: np.ndarray, original_sfreq: float, target_sfreq: float) -> np.ndarray:
    """Resamples EEG trials to a target sampling frequency."""
    resampled_trials = []
    down_factor = original_sfreq / target_sfreq
    
    for trial in X:
        trial_resampled = mne.filter.resample(
            trial,
            down=down_factor,
            npad="auto"
        )
        resampled_trials.append(trial_resampled)

    return np.array(resampled_trials)


def apply_bandpass(X: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 4) -> np.ndarray:
    """Applies a Butterworth bandpass filter across all trials and channels."""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')

    filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered[i, ch] = filtfilt(b, a, X[i, ch])

    return filtered


def apply_notch_filter(X: np.ndarray, notch_freq: float, fs: float, quality: float = 30.0) -> np.ndarray:
    """Applies an IIR notch filter to remove powerline noise (e.g., 50Hz/60Hz)."""
    b, a = iirnotch(notch_freq, quality, fs)

    filtered = np.zeros_like(X)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered[i, ch] = filtfilt(b, a, X[i, ch])

    return filtered