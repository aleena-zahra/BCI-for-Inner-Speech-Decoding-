"""POC continuous-MNE workflow with synchronized events and explicit ICA exclusions."""

import mne
import numpy as np

from voice.preprocessing.artifacts import fit_ica
from voice.preprocessing.normalization import zscore_normalize


class LiteraturePreprocessingPipeline:
    """Filter continuous EEG before epoching; returned trials are in microvolts.

    ICA is optional and removes only explicitly reviewed component indices.
    Call this on separate training recordings if decomposition will feed an
    evaluation; fitting on held-out recordings changes the evaluation protocol.
    """

    def __init__(self, target_sfreq=250.0, l_freq=0.5, h_freq=100.0,
                 notch_freqs=(50.0,), ica_components=None, apply_car=True,
                 ica_exclude=(), zscore=True, artifact_threshold=None):
        self.target_sfreq = float(target_sfreq)
        self.l_freq = float(l_freq)
        self.h_freq = float(h_freq)
        self.notch_freqs = tuple(notch_freqs)
        self.ica_components = ica_components
        self.apply_car = apply_car
        self.ica_exclude = tuple(ica_exclude)
        self.zscore = zscore
        self.artifact_threshold = artifact_threshold
        self.ica_ = None

    def run_pipeline(self, raw, events=None, event_id=None, tmin=1.5, tmax=3.5, baseline=None):
        """Return ``(trials, retained_events, epochs)`` or cleaned Raw without events.

        The caller's Raw/events are never modified. MNE jointly resamples events
        and the recording, preserving their timing including nonzero first_samp.
        """
        if (events is None) != (event_id is None):
            raise ValueError("Provide both events and event_id, or neither")
        if not 0 < self.l_freq < self.h_freq < min(raw.info["sfreq"], self.target_sfreq) / 2:
            raise ValueError("Bandpass must be ordered and below original/target Nyquist frequencies")
        clean = raw.copy().load_data().pick("eeg")
        if not clean.ch_names:
            raise ValueError("No EEG channels found")
        clean.filter(self.l_freq, self.h_freq, method="fir", phase="zero", verbose=False)
        notches = [f for f in self.notch_freqs if f is not None and 0 < f < clean.info["sfreq"] / 2]
        if notches:
            clean.notch_filter(notches, method="fir", phase="zero", verbose=False)
        aligned_events = None if events is None else np.asarray(events, dtype=int).copy()
        if clean.info["sfreq"] != self.target_sfreq:
            if aligned_events is None:
                clean.resample(self.target_sfreq, verbose=False)
            else:
                clean, aligned_events = clean.resample(self.target_sfreq, events=aligned_events, verbose=False)
        if self.apply_car:
            clean.set_eeg_reference("average", projection=False, verbose=False)
        self.ica_ = None
        if self.ica_components is not None:
            if self.apply_car and self.ica_components > len(clean.ch_names) - 1:
                raise ValueError("ICA component count must respect CAR rank")
            fitting = clean.copy().filter(max(1.0, self.l_freq), None, method="iir", verbose=False)
            self.ica_ = fit_ica(fitting, self.ica_components, self.ica_exclude)
            if self.ica_.exclude:
                self.ica_.apply(clean, verbose=False)
        elif self.ica_exclude:
            raise ValueError("Specify ICA components before excluded indices")
        if aligned_events is None:
            return clean
        if tmax <= tmin:
            raise ValueError("Epoch end must be later than epoch start")
        epochs = mne.Epochs(clean, aligned_events, event_id=event_id, tmin=tmin, tmax=tmax,
                            baseline=baseline, preload=True, reject_by_annotation=True, verbose=False)
        values = (epochs.get_data() * 1e6).astype(np.float32)
        if self.artifact_threshold is not None:
            keep = np.max(np.abs(values), axis=(1, 2)) < self.artifact_threshold
            epochs = epochs[np.flatnonzero(keep)]
            values = values[keep]
        if not len(values):
            raise ValueError("No complete trials survived preprocessing")
        if self.zscore:
            values = zscore_normalize(values)
        return values, epochs.events.copy(), epochs
