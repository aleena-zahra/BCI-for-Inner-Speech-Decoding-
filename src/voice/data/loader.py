import os
import re
import pandas as pd
import numpy as np
import mne

def scan_bids_eeg_dataset(root_dir: str) -> dict:
    """Scans root_dir for BIDS EEG files and events.tsv files."""
    runs = {}
    all_files = []
    for root, _, files in os.walk(root_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    eeg_files = [f for f in all_files if f.endswith((".bdf", ".set", ".fif", ".vhdr"))]
    tsv_files = [f for f in all_files if f.endswith("events.tsv")]

    for eeg_path in eeg_files:
        base_name = os.path.basename(eeg_path)
        subj_match = re.search(r"sub-[a-zA-Z0-9]+", base_name) or re.search(r"sub-[a-zA-Z0-9]+", eeg_path)
        subj_id = subj_match.group(0) if subj_match else "sub-01"

        matched_tsv = next((tsv for tsv in tsv_files if subj_id in tsv), None)
        runs[subj_id] = {
            "subj": subj_id,
            "eeg_file": eeg_path,
            "events_file": matched_tsv,
            "filename": base_name
        }
    return runs


def parse_and_load_eeg(run_info: dict):
    """Loads raw EEG recording and synchronizes events."""
    eeg_file = run_info["eeg_file"]
    events_file = run_info.get("events_file")

    if eeg_file.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".set"):
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".fif"):
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG format: {eeg_file}")

    try:
        raw.pick_types(eeg=True, stim=False, misc=False)
    except Exception:
        pass

    if events_file and os.path.exists(events_file):
        events_df = pd.read_csv(events_file, sep="\t")
        label_col = next((c for c in ["trial_type", "value", "condition"] if c in events_df.columns), None)
        labels_str = events_df[label_col].values if label_col else np.ones(len(events_df), dtype=str)
        unique_labels = sorted(set(labels_str))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels_int = np.array([label_map[l] for l in labels_str])

        onsets = events_df["onset"].values
        sfreq = raw.info["sfreq"]
        onset_samples = np.round(onsets * sfreq if np.max(onsets) <= raw.times[-1] * 2 else (onsets / 1000.0) * sfreq).astype(int)

        valid_idx = np.where(onset_samples < raw.n_times)[0]
        events = np.column_stack([onset_samples[valid_idx], np.zeros(len(valid_idx), dtype=int), labels_int[valid_idx]])
        cn = unique_labels
    else:
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        cn = list(event_dict.keys()) if len(events) > 0 else ["class_0", "class_1", "class_2", "class_3"]

    meta = {
        "filename": os.path.basename(eeg_file),
        "duration": raw.times[-1],
        "sfreq": raw.info["sfreq"],
        "subj_id": run_info.get("subj", "sub-01")
    }
    return raw, events, np.zeros(len(events), dtype=int), cn, meta