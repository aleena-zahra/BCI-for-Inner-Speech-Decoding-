import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler

class LiteraturePreprocessingPipeline:
    """
    Literature-Based EEG Preprocessing Pipeline for Inner Speech Decoding.
    Preserves the Gamma band (30-100Hz) and removes low-frequency drifts (<4Hz) 
    and artifacts to prevent deep learning shortcut learning.
    """
    
    def __init__(self, target_sfreq=250, l_freq=4.0, h_freq=100.0, notch_freqs=[50.0, 60.0], ica_components=15):
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = [f for f in notch_freqs if f is not None]
        self.ica_components = ica_components
        
    def resample(self, raw):
        """
        Resamples the raw MNE object to the target sampling frequency.
        Must be >= 200Hz to preserve the 100Hz Gamma band.
        """
        if int(raw.info['sfreq']) != int(self.target_sfreq):
            print(f"Resampling data from {raw.info['sfreq']} Hz to {self.target_sfreq} Hz...")
            raw.resample(self.target_sfreq)
        return raw
        
    def filter_data(self, raw):
        """
        Applies a bandpass filter (4-100Hz) and a notch filter.
        """
        # Pick EEG channels if non-EEG (status, stim) are present
        try:
            raw.pick_types(eeg=True, stim=False, misc=False)
        except Exception:
            pass

        print(f"Applying bandpass filter ({self.l_freq} - {self.h_freq} Hz)...")
        # Ensure nyquist constraint
        max_h_freq = (raw.info['sfreq'] / 2.0) - 1.0
        actual_h_freq = min(self.h_freq, max_h_freq)
        raw.filter(l_freq=self.l_freq, h_freq=actual_h_freq, fir_design='firwin', verbose=False)
        
        if self.notch_freqs:
            valid_notches = [nf for nf in self.notch_freqs if nf < (raw.info['sfreq'] / 2.0)]
            if valid_notches:
                print(f"Applying notch filter at {valid_notches} Hz...")
                raw.notch_filter(freqs=valid_notches, fir_design='firwin', verbose=False)
        return raw
        
    def apply_ica(self, raw):
        """
        Applies Independent Component Analysis (ICA) to remove eye blinks (EOG) 
        and muscle artifacts (EMG).
        """
        n_channels = len(raw.ch_names)
        if isinstance(self.ica_components, (int, float)) and self.ica_components >= 1:
            n_comp = min(int(self.ica_components), n_channels - 1)
        elif isinstance(self.ica_components, float) and 0 < self.ica_components < 1:
            n_comp = self.ica_components
        else:
            n_comp = min(15, n_channels - 1)

        print(f"Fitting ICA (components={n_comp})...")
        try:
            ica = ICA(n_components=n_comp, random_state=42, max_iter='auto')
            ica.fit(raw, verbose=False)
            print("Applying ICA reconstruction...")
            ica.apply(raw, verbose=False)
            return raw, ica
        except Exception as e:
            print(f"ICA fit encountered an issue: {e}. Continuing with filtered data.")
            return raw, None

    def epoch_data(self, raw, events, event_id, tmin=0.5, tmax=3.5, baseline=None):
        """
        Epochs the data based on given events and time windows.
        Handles baseline interval safety.
        """
        print(f"Epoching data from {tmin}s to {tmax}s relative to event onset...")
        
        # Determine baseline safety:
        # If tmin >= 0, standard (None, 0) is invalid in MNE.
        if baseline == (None, 0) and tmin >= 0:
            baseline = None

        epochs = mne.Epochs(
            raw, 
            events=events, 
            event_id=event_id, 
            tmin=tmin, 
            tmax=tmax, 
            baseline=baseline, 
            preload=True,
            verbose=False
        )
        return epochs

    def zscore_normalize(self, epochs_data):
        """
        Applies channel-wise Z-score normalization to combat non-stationarity.
        epochs_data: numpy array of shape [trials, channels, timepoints]
        Returns normalized array.
        """
        print("Applying channel-wise Z-score normalization...")
        trials, channels, timepoints = epochs_data.shape
        normalized_data = np.empty_like(epochs_data)
        
        for i in range(trials):
            scaler = StandardScaler()
            # Transpose to [timepoints, channels], scale, then transpose back to [channels, timepoints]
            normalized_data[i] = scaler.fit_transform(epochs_data[i].T).T
            
        return normalized_data

    def run_pipeline(self, raw, events=None, event_id=None, tmin=0.5, tmax=3.5, baseline=None):
        """
        Executes the full pipeline on a raw MNE object.
        Properly rescales event sample timestamps during resampling.
        """
        print("--- Starting Literature-Based Preprocessing Pipeline ---")
        orig_sfreq = raw.info['sfreq']

        # 1. Filter continuous raw
        raw = self.filter_data(raw)

        # 2. Resample
        raw = self.resample(raw)
        new_sfreq = raw.info['sfreq']

        # 3. Scale events sample positions if resampling occurred
        if events is not None:
            scaled_events = events.copy()
            if int(orig_sfreq) != int(new_sfreq):
                scaled_events[:, 0] = (scaled_events[:, 0] * (float(new_sfreq) / float(orig_sfreq))).astype(int)
        else:
            scaled_events = None

        # 4. ICA Artifact Removal
        raw, ica = self.apply_ica(raw)
        
        # 5. Epoching & Normalization
        if scaled_events is not None and event_id is not None:
            epochs = self.epoch_data(raw, scaled_events, event_id, tmin, tmax, baseline=baseline)
            epochs_data = epochs.get_data(copy=True)
            final_data = self.zscore_normalize(epochs_data)
            print("--- Pipeline Complete ---")
            return final_data, epochs.events, epochs
        else:
            print("No events provided. Returning continuous raw data.")
            print("--- Pipeline Complete ---")
            return raw
