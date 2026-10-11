"""Convert BIDS recordings and events into aligned trial tensors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np

from voice.data.loader import parse_and_load_eeg, scan_bids_eeg_dataset


@dataclass(frozen=True)
class EEGTrials:
    signals: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray
    runs: np.ndarray
    label_names: tuple[str, ...]
    channel_names: tuple[str, ...]
    sfreq: float


def load_bids_trials(
    root_dir: str | Path,
    *,
    tmin: float = 0.0,
    tmax: float = 3.5,
) -> EEGTrials:
    """Load all runs as epochs; labels and subject/run IDs remain row-aligned."""
    run_infos = list(scan_bids_eeg_dataset(root_dir).values())
    if not run_infos:
        raise ValueError(f"No supported BIDS EEG recordings found in {root_dir}")

    loaded = [parse_and_load_eeg(info) + (info,) for info in run_infos]
    label_names = tuple(sorted({str(label) for _, _, labels, _, _, _ in loaded for label in labels}))
    if len(label_names) < 2:
        raise ValueError("At least two distinct event labels are required for classification")
    label_codes = {label: index + 1 for index, label in enumerate(label_names)}

    reference_channels = loaded[0][0].ch_names
    reference_sfreq = float(loaded[0][0].info["sfreq"])
    signal_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    subject_parts: list[np.ndarray] = []
    run_parts: list[np.ndarray] = []

    for raw, events, labels, _, meta, info in loaded:
        if raw.ch_names != reference_channels:
            raise ValueError(f"EEG channel mismatch in run {meta['run_id']}")
        if not np.isclose(raw.info["sfreq"], reference_sfreq):
            raise ValueError(f"Sampling-rate mismatch in run {meta['run_id']}")
        if meta["subject"] is None:
            raise ValueError(f"Recording has no BIDS subject identifier: {meta['run_id']}")

        mapped = np.asarray([label_codes[str(label)] for label in labels], dtype=int)
        event_array = events.copy()
        event_array[:, 2] = mapped
        event_id = {name: code for name, code in label_codes.items() if code in mapped}
        epochs = mne.Epochs(
            raw,
            event_array,
            event_id=event_id,
            tmin=tmin,
            tmax=tmax,
            baseline=None,
            preload=True,
            reject_by_annotation=True,
            verbose=False,
        )
        if not len(epochs):
            continue
        # MNE stores EEG in volts; use microvolts consistently with the UI and
        # amplitude-rejection threshold used by EEGPreprocessor.
        signal_parts.append((epochs.get_data() * 1e6).astype(np.float32, copy=False))
        label_parts.append(epochs.events[:, 2])
        subject_parts.append(np.repeat(meta["subject"], len(epochs)))
        run_parts.append(np.repeat(meta["run_id"], len(epochs)))

    if not signal_parts:
        raise ValueError("No complete event-locked trials were found in the selected recordings")
    present_codes = set(np.concatenate(label_parts).tolist())
    if len(present_codes) < 2:
        raise ValueError("At least two event classes must have complete trials")
    return EEGTrials(
        signals=np.concatenate(signal_parts).astype(np.float32, copy=False),
        labels=np.concatenate(label_parts).astype(int, copy=False),
        subjects=np.concatenate(subject_parts),
        runs=np.concatenate(run_parts),
        label_names=label_names,
        channel_names=tuple(reference_channels),
        sfreq=reference_sfreq,
    )
