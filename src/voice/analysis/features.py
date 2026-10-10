import numpy as np
from mne.decoding import CSP

class MultiBandCSPExtractor:
    """Extracts spatial patterns across multiple frequency bands (Baseline Pipeline)."""

    def __init__(self, n_components: int = 4):
        self.csp_alpha = CSP(n_components=n_components, log=True)
        self.csp_beta = CSP(n_components=n_components, log=True)

    def fit_transform(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Fit CSP on train data and return stacked spatial features."""
        # Split frequency bands using simple filtering
        X_alpha = mne.filter.filter_data(X_train.astype(np.float64), sfreq=128, l_freq=8, h_freq=12, verbose=False)
        X_beta = mne.filter.filter_data(X_train.astype(np.float64), sfreq=128, l_freq=13, h_freq=30, verbose=False)

        X_a = self.csp_alpha.fit_transform(X_alpha, y_train)
        X_b = self.csp_beta.fit_transform(X_beta, y_train)
        return np.hstack([X_a, X_b])

    def transform(self, X_test: np.ndarray) -> np.ndarray:
        """Transform test data through fitted CSP filters."""
        X_alpha = mne.filter.filter_data(X_test.astype(np.float64), sfreq=128, l_freq=8, h_freq=12, verbose=False)
        X_beta = mne.filter.filter_data(X_test.astype(np.float64), sfreq=128, l_freq=13, h_freq=30, verbose=False)

        X_a = self.csp_alpha.transform(X_alpha)
        X_b = self.csp_beta.transform(X_beta)
        return np.hstack([X_a, X_b])