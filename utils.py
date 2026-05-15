


def resample_data(X, original_sfreq, target_sfreq):
    """
    X shape: [trials, channels, time]
    """
    n_trials, n_channels, _ = X.shape

    resampled_trials = []

    for trial in X:
        trial_resampled = mne.filter.resample(
            trial,
            down=original_sfreq / target_sfreq,
            npad="auto"
        )
        resampled_trials.append(trial_resampled)

    return np.array(resampled_trials)



def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    b, a = butter(order, [low, high], btype='band')
    return b, a


def apply_bandpass(X, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)

    filtered = np.zeros_like(X)

    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered[i, ch] = filtfilt(b, a, X[i, ch])

    return filtered


def bandpass(X, low, high, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [low/nyq, high/nyq], btype='band')

    X_filt = np.zeros_like(X)
    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            X_filt[i, ch] = filtfilt(b, a, X[i, ch])
    return X_filt

def apply_notch_filter(X, notch_freq, fs, quality=30):
    b, a = iirnotch(notch_freq, quality, fs)

    filtered = np.zeros_like(X)

    for i in range(X.shape[0]):
        for ch in range(X.shape[1]):
            filtered[i, ch] = filtfilt(b, a, X[i, ch])

    return filtered




def baseline_correction(X):
    """
    Subtract mean over time for each channel.
    """
    baseline = np.mean(X, axis=2, keepdims=True)
    return X - baseline


def remove_artifacts(X, threshold=100):
    """
    Remove trials with extreme amplitudes.
    """
    keep_trials = []

    for i in range(X.shape[0]):
        if np.max(np.abs(X[i])) < threshold:
            keep_trials.append(i)

    return X[keep_trials], keep_trials

