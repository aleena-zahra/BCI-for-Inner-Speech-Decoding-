"""Trial preprocessing shared by the dashboard and fold-fitted classifiers."""

import mne
import numpy as np

from voice.preprocessing.artifacts import fit_ica
from voice.preprocessing.filtering import select_time_window, resample_data, apply_bandpass, apply_notch_filter
from voice.preprocessing.normalization import baseline_correction, remove_artifacts, zscore_normalize


class EEGPreprocessor:
    """Process microvolt trial arrays; optional ICA is fitted on training trials only.

    Legacy IIR settings remain the default. FIR mode imports the POC's filtering
    order: filter complete input trials before resampling and interval cropping.
    For genuinely continuous filtering, use LiteraturePreprocessingPipeline.
    """

    def __init__(self, sfreq=256.0, target_sfreq=128.0, lowcut=1.0, highcut=40.0,
                 notch_freq=50.0, t_start=1.5, t_end=3.5, apply_baseline=False,
                 artifact_threshold=None, common_average_reference=False,
                 filter_method="iir", notch_freqs=None, zscore=False,
                 ica_components=None, ica_exclude=()):
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
        self.filter_method = filter_method
        self.notch_freqs = notch_freqs
        self.zscore = zscore
        self.ica_components = ica_components
        self.ica_exclude = tuple(ica_exclude)
        self.ica_ = None

    def _prepare(self, X):
        X = np.asarray(X)
        if X.ndim != 3 or not len(X) or not np.isfinite(X).all():
            raise ValueError("EEG data must be a finite non-empty (trials, channels, time) array")
        if self.sfreq <= 0 or self.target_sfreq <= 0:
            raise ValueError("Sampling frequencies must be positive")
        if not 0 < self.lowcut < self.highcut < min(self.sfreq, self.target_sfreq) / 2:
            raise ValueError("Bandpass cutoffs must be ordered and below both sampling rates' Nyquist limits")
        if self.t_start < 0 or self.t_end <= self.t_start or self.t_end > X.shape[-1] / self.sfreq:
            raise ValueError("Requested time window is outside the input trial")
        notches = self.notch_freqs if self.notch_freqs is not None else (self.notch_freq,)
        if self.filter_method == "fir":
            # Keep temporary float64 buffers bounded; MNE FIR requires float64.
            filtered = np.empty(X.shape, dtype=np.float32)
            for start in range(0, len(X), 8):
                batch = np.asarray(X[start:start + 8], dtype=np.float64)
                batch = mne.filter.filter_data(batch, self.sfreq, self.lowcut, self.highcut,
                                               method="fir", phase="zero", verbose=False)
                valid = [float(f) for f in notches if f is not None and 0 < f < self.sfreq / 2]
                if valid:
                    batch = mne.filter.notch_filter(batch, self.sfreq, valid, method="fir",
                                                   phase="zero", verbose=False)
                filtered[start:start + 8] = batch
            X_proc = resample_data(filtered, self.sfreq, self.target_sfreq) if self.sfreq != self.target_sfreq else filtered
            X_proc = select_time_window(X_proc, self.t_start, self.t_end, self.target_sfreq)
        elif self.filter_method == "iir":
            X_proc = select_time_window(X, self.t_start, self.t_end, self.sfreq)
            if self.sfreq != self.target_sfreq:
                X_proc = resample_data(X_proc, self.sfreq, self.target_sfreq)
            X_proc = apply_bandpass(X_proc, self.lowcut, self.highcut, self.target_sfreq)
            for frequency in notches:
                if frequency is not None and 0 < frequency < self.target_sfreq / 2:
                    X_proc = apply_notch_filter(X_proc, frequency, self.target_sfreq)
        else:
            raise ValueError("filter_method must be 'iir' or 'fir'")
        if self.common_average_reference:
            X_proc = X_proc - X_proc.mean(axis=1, keepdims=True)
        if self.apply_baseline:
            X_proc = baseline_correction(X_proc)
        return np.asarray(X_proc, dtype=np.float32)

    def _epochs(self, X):
        info = mne.create_info([f"EEG{i}" for i in range(X.shape[1])], self.target_sfreq, "eeg")
        return mne.EpochsArray(np.asarray(X, dtype=np.float64) * 1e-6, info, verbose=False)

    def _fit_prepared(self, X):
        self.ica_ = None
        if self.ica_components is not None:
            if self.common_average_reference and self.ica_components > X.shape[1] - 1:
                raise ValueError("ICA component count must respect the rank lost through CAR")
            # ICA fits a >=1 Hz copy; reconstruction retains the requested passband.
            fitting = self._epochs(X)
            fitting.filter(l_freq=max(1.0, self.lowcut), h_freq=None, method="iir", verbose=False)
            self.ica_ = fit_ica(fitting, self.ica_components, self.ica_exclude)
        elif self.ica_exclude:
            raise ValueError("Configure ICA components before specifying excluded components")

    def _finish(self, prepared, y=None):
        if y is not None and len(y) != len(prepared):
            raise ValueError("Labels must have one entry per trial")
        if self.ica_components is not None:
            if self.ica_ is None:
                raise RuntimeError("Fit the preprocessor on training data before applying ICA")
            if self.ica_.exclude:
                epochs = self._epochs(prepared)
                self.ica_.apply(epochs, verbose=False)
                prepared = (epochs.get_data() * 1e6).astype(np.float32)
        kept = np.arange(len(prepared))
        # Amplitude thresholds are measured in microvolts BEFORE z-scoring.
        if self.artifact_threshold is not None:
            prepared, kept = remove_artifacts(prepared, threshold=self.artifact_threshold)
        if not len(prepared):
            raise ValueError("Preprocessing rejected every trial")
        if self.zscore:
            prepared = zscore_normalize(prepared)
        return (prepared, np.asarray(y)[kept], kept) if y is not None else (prepared, kept)

    def fit(self, X, y=None):
        if y is not None and len(X) != len(y):
            raise ValueError("Signals and labels must be aligned")
        self._fit_prepared(self._prepare(X))
        return self

    def fit_transform_with_indices(self, X, y=None):
        prepared = self._prepare(X)
        self._fit_prepared(prepared)
        return self._finish(prepared, y)

    def transform_with_indices(self, X, y=None):
        return self._finish(self._prepare(X), y)

    def ica_sources(self, X):
        """Return trial/component/time traces for review using the fitted decomposition."""
        if self.ica_ is None:
            raise RuntimeError("Fit ICA before reviewing its sources")
        return self.ica_.get_sources(self._epochs(self._prepare(X))).get_data()

    def transform(self, X, y=None):
        result = self.transform_with_indices(X, y)
        return result[:2] if y is not None else result[0]

    def fit_transform(self, X, y=None):
        result = self.fit_transform_with_indices(X, y)
        return result[:2] if y is not None else result[0]


def preprocess_eeg_dataset(X, **kwargs):
    """Fit and transform a trial dataset for standalone preparation/preview."""
    return EEGPreprocessor(**kwargs).fit_transform(X)
