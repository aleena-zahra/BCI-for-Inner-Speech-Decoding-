import numpy as np
import mne
from mne.decoding import CSP

class MultiBandCSPExtractor:
    """Extracts spatial patterns across multiple frequency bands (Baseline Pipeline)."""

    def __init__(self, n_components: int = 4, sfreq: float = 128.0,
                 bands: tuple[tuple[float, float], ...] = ((8.0, 12.0), (13.0, 30.0))):
        if not bands or any(low <= 0 or high <= low or high >= sfreq / 2 for low, high in bands):
            raise ValueError("CSP bands must be positive and below the Nyquist frequency")
        self.sfreq = float(sfreq)
        self.bands = bands
        self.csps = [CSP(n_components=n_components, log=True) for _ in bands]
        # Preserve these attributes for existing callers.
        self.csp_alpha, self.csp_beta = self.csps[:2] if len(self.csps) >= 2 else (self.csps[0], self.csps[0])

    def _filter_bands(self, X: np.ndarray) -> list[np.ndarray]:
        if X.ndim != 3 or not np.isfinite(X).all():
            raise ValueError("CSP expects finite (trials, channels, time) EEG data")
        return [
            mne.filter.filter_data(
                X.astype(np.float64, copy=False), sfreq=self.sfreq,
                l_freq=low, h_freq=high, verbose=False,
            )
            for low, high in self.bands
        ]

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit CSP on train data and return stacked spatial features."""
        if len(X_train) != len(y_train):
            raise ValueError("Labels must have one entry per EEG trial")
        return np.hstack([
            csp.fit_transform(X_band, y_train)
            for csp, X_band in zip(self.csps, self._filter_bands(X_train))
        ])

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Transform test data through fitted CSP filters."""
        return np.hstack([
            csp.transform(X_band)
            for csp, X_band in zip(self.csps, self._filter_bands(X_test))
        ])
