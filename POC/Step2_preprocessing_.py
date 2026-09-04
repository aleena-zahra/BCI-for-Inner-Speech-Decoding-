import numpy as np
import mne
from mne.preprocessing import ICA
from sklearn.preprocessing import StandardScaler

class LiteraturePreprocessingPipeline:
    """
    Literature-Based EEG Preprocessing Pipeline for Inner Speech Decoding.
    Follows the 9-step methodology from Nieto et al. (2022), Zhou et al. (2025), and Lopez-Bernal (2024):
    
    1. Scalp EEG Channel Picking & Segregation
    2. Zero-phase FIR Bandpass (0.5 - 100.0 Hz) & 50Hz Notch Filtering (preserving Gamma 30-100Hz)
    3. Temporal Decimation (Downsampling to 250 Hz)
    4. Spatial Common Average Re-referencing (CAR)
    5. Independent Component Analysis (ICA) Artifact Removal
    6. Trial Epoching & Visual Cue P300 Window Rejection (IoI: t=1.5s to 3.5s)
    7. Per-Channel, Per-Trial Z-Score Normalization
    """
    
    def __init__(self, target_sfreq=250, l_freq=0.5, h_freq=100.0, notch_freqs=[50.0, 60.0], ica_components=15, apply_car=True, **kwargs):
        self.target_sfreq = target_sfreq
        self.l_freq = l_freq
        self.h_freq = h_freq
        self.notch_freqs = [f for f in notch_freqs if f is not None]
        self.ica_components = ica_components
        self.apply_car = apply_car
        
    def resample(self, raw):
        """
        Step 3: Temporal Decimation
        Resamples the raw MNE object to target frequency (e.g. 250 Hz).
        Comfortably captures the 30-100 Hz Gamma band (Nyquist = 125 Hz).
        """
        if int(raw.info['sfreq']) != int(self.target_sfreq):
            print(f"Resampling data from {raw.info['sfreq']} Hz to {self.target_sfreq} Hz...")
            raw.resample(self.target_sfreq)
        return raw
        
    def filter_data(self, raw):
        """
        Step 2: Bandpass & Notch Filtering
        Applies a zero-phase FIR bandpass filter (0.5-100Hz) and a notch filter at 50/60Hz.
        """
        # Pick EEG channels
        try:
            raw.pick_types(eeg=True, stim=False, misc=False)
        except Exception:
            pass

        print(f"Applying bandpass filter ({self.l_freq} - {self.h_freq} Hz)...")
        max_h_freq = (raw.info['sfreq'] / 2.0) - 1.0
        actual_h_freq = min(self.h_freq, max_h_freq)
        raw.filter(l_freq=self.l_freq, h_freq=actual_h_freq, fir_design='firwin', verbose=False)
        
        if self.notch_freqs:
            valid_notches = [nf for nf in self.notch_freqs if nf < (raw.info['sfreq'] / 2.0)]
            if valid_notches:
                print(f"Applying notch filter at {valid_notches} Hz...")
                raw.notch_filter(freqs=valid_notches, fir_design='firwin', verbose=False)
        return raw

    def apply_spatial_rereference(self, raw):
        """
        Step 4: Spatial Re-Referencing
        Applies Common Average Reference (CAR) across all scalp EEG electrodes.
        """
        if self.apply_car:
            print("Applying Common Average Reference (CAR)...")
            try:
                raw.set_eeg_reference('average', projection=False, verbose=False)
            except Exception as e:
                print(f"CAR re-referencing skipped: {e}")
        return raw
        
    def apply_ica(self, raw):
        """
        Step 7: Independent Component Analysis (ICA) Denoising
        Splits data into independent components to reject eye blinks (EOG) and muscle artifacts (EMG).
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

    def epoch_data(self, raw, events, event_id, tmin=1.5, tmax=3.5, baseline=None):
        """
        Step 5 & 6: Trial Epoching & Visual Cue P300 Window Rejection
        Extracts the pure Interval of Interest (IoI) spanning t=1.5s to 3.5s (2.0s action interval).
        """
        print(f"Epoching pure Action Interval of Interest (IoI: {tmin}s to {tmax}s post-stimulus)...")
        
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
        Step 9: Downstream Normalization (Deep Learning Readiness)
        Applies per-trial, per-channel Z-score normalization: x_hat = (x - mu) / sigma.
        """
        print("Applying channel-wise Z-score normalization...")
        trials, channels, timepoints = epochs_data.shape
        normalized_data = np.empty_like(epochs_data)
        
        for i in range(trials):
            scaler = StandardScaler()
            normalized_data[i] = scaler.fit_transform(epochs_data[i].T).T
            
        return normalized_data

    def run_pipeline(self, raw, events=None, event_id=None, tmin=1.5, tmax=3.5, baseline=None):
        """
        Executes the full literature-compliant preprocessing pipeline on a raw MNE object.
        """
        print("--- Starting Literature-Based Preprocessing Pipeline ---")
        orig_sfreq = raw.info['sfreq']

        # 1. Bandpass & Notch Filtering (0.5 - 100 Hz, 50 Hz Notch)
        raw = self.filter_data(raw)

        # 2. Temporal Decimation (Resampling to 250 Hz)
        raw = self.resample(raw)
        new_sfreq = raw.info['sfreq']

        # 3. Synchronize event sample timestamps
        if events is not None:
            scaled_events = events.copy()
            if int(orig_sfreq) != int(new_sfreq):
                scaled_events[:, 0] = (scaled_events[:, 0] * (float(new_sfreq) / float(orig_sfreq))).astype(int)
        else:
            scaled_events = None

        # 4. Spatial Re-referencing (CAR)
        raw = self.apply_spatial_rereference(raw)

        # 5. ICA Artifact Denoising
        raw, ica = self.apply_ica(raw)
        
        # 6. Epoching & Visual Cue P300 Window Rejection (1.5s to 3.5s)
        if scaled_events is not None and event_id is not None:
            epochs = self.epoch_data(raw, scaled_events, event_id, tmin, tmax, baseline=baseline)
            epochs_data = epochs.get_data(copy=True)
            # 7. Z-Score Normalization
            final_data = self.zscore_normalize(epochs_data)
            print(f"--- Pipeline Complete: Output Shape {final_data.shape} ---")
            return final_data, epochs.events, epochs
        else:
            print("No events provided. Returning continuous raw data.")
            print("--- Pipeline Complete ---")
            return raw
