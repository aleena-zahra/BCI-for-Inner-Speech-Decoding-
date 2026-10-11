"""Explicit ICA fitting and component exclusion for MNE Raw or Epochs data."""

from mne.preprocessing import ICA
from numbers import Integral


def fit_ica(inst, n_components: int, exclude=(), *, random_state: int = 42):
    """Fit ICA and set only manually reviewed exclusions; never infer brain-only data."""
    if not isinstance(n_components, Integral) or not 2 <= n_components <= len(inst.ch_names):
        raise ValueError("ICA components must be an integer between 2 and the EEG channel count")
    excluded = tuple(exclude)
    if any(not isinstance(index, Integral) or index < 0 or index >= n_components for index in excluded):
        raise ValueError("ICA exclusions must be zero-based indices below the component count")
    if len(set(excluded)) >= n_components:
        raise ValueError("Cannot exclude every ICA component")
    ica = ICA(n_components=n_components, method="infomax", fit_params={"extended": True},
              random_state=random_state, max_iter=500)
    ica.fit(inst, decim=3, verbose=False)
    ica.exclude = sorted(set(excluded))
    return ica
