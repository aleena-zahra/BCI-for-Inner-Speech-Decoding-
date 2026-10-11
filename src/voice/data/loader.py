"""BIDS EEG discovery and validated event loading."""

from __future__ import annotations

import os
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

import mne
import numpy as np
import pandas as pd

EEG_SUFFIXES = (".bdf", ".set", ".fif", ".vhdr")


def extract_bids_archive(archive: str | os.PathLike, destination: str | os.PathLike) -> Path:
    """Safely extract a BIDS ZIP, rejecting entries that escape destination."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        members = zipped.infolist()
        eeg_members = [member for member in members if PurePosixPath(member.filename).suffix.lower() in EEG_SUFFIXES]
        if not eeg_members:
            raise ValueError(
                "This ZIP contains no supported BIDS EEG recording (.bdf, .set, .fif, or .vhdr). "
                "CSV and MATLAB (.mat) archives are not supported by this Phase 1 loader."
            )
        for member in members:
            name = member.filename.replace("\\", "/")
            candidate = (destination / Path(*PurePosixPath(name).parts)).resolve()
            if candidate != destination and destination not in candidate.parents:
                raise ValueError(f"Unsafe path in ZIP archive: {member.filename}")
        zipped.extractall(destination)
    # Common archives contain one enclosing folder; use it as the dataset root.
    children = list(destination.iterdir())
    if len(children) == 1 and children[0].is_dir():
        return children[0]
    return destination


def _entities(path: Path) -> dict[str, str]:
    entities: dict[str, str] = {}
    for part in path.name.split("_"):
        if "-" in part:
            key, value = part.split("-", 1)
            entities[key] = value.split(".", 1)[0]
    return entities


def _matching_events(eeg_path: Path, tsv_paths: list[Path]) -> Path | None:
    eeg_entities = _entities(eeg_path)
    # Match all shared BIDS entities and require the same subject. This prevents
    # accidentally pairing an events file from another run or session.
    candidates = []
    for tsv in tsv_paths:
        event_entities = _entities(tsv)
        if event_entities.get("sub") != eeg_entities.get("sub"):
            continue
        shared = set(eeg_entities) & set(event_entities)
        if all(eeg_entities[key] == event_entities[key] for key in shared):
            candidates.append((len(shared), tsv))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    if len(candidates) > 1 and candidates[0][0] == candidates[1][0]:
        return None
    return candidates[0][1]


def scan_bids_eeg_dataset(root_dir: str | os.PathLike) -> dict[str, dict[str, Any]]:
    """Find EEG recordings and their matching events tables in a BIDS tree."""
    root = Path(root_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"BIDS dataset directory does not exist: {root}")
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    tsv_paths = [path for path in paths if path.name.endswith("events.tsv")]
    runs: dict[str, dict[str, Any]] = {}
    for eeg_path in paths:
        if eeg_path.suffix.lower() not in EEG_SUFFIXES:
            continue
        entities = _entities(eeg_path)
        subject = f"sub-{entities['sub']}" if "sub" in entities else None
        run_id = str(eeg_path.relative_to(root))
        events_file = _matching_events(eeg_path, tsv_paths)
        runs[run_id] = {
            "run_id": run_id,
            "subject": subject,
            "subj": subject,
            "task": entities.get("task"),
            "session": entities.get("ses"),
            "run": entities.get("run"),
            "eeg_file": str(eeg_path),
            "events_file": str(events_file) if events_file else None,
            "filename": eeg_path.name,
        }
    return runs


def _read_raw(eeg_file: str):
    suffix = Path(eeg_file).suffix.lower()
    readers = {
        ".bdf": mne.io.read_raw_bdf,
        ".vhdr": mne.io.read_raw_brainvision,
        ".set": mne.io.read_raw_eeglab,
        ".fif": mne.io.read_raw_fif,
    }
    if suffix not in readers:
        raise ValueError(f"Unsupported EEG format: {eeg_file}")
    raw = readers[suffix](eeg_file, preload=True, verbose=False)
    raw.pick("eeg")
    if not raw.ch_names:
        raise ValueError(f"No EEG channels found in {eeg_file}")
    return raw


def _events_from_tsv(raw, events_file: str) -> tuple[np.ndarray, dict[str, int], list[str]]:
    table = pd.read_csv(events_file, sep="\t")
    if "onset" not in table:
        raise ValueError(f"Events table has no 'onset' column: {events_file}")
    label_column = next((name for name in ("trial_type", "condition", "value") if name in table), None)
    if label_column is None:
        raise ValueError(f"Events table has no trial_type, condition, or value labels: {events_file}")
    labels = table[label_column]
    valid = labels.notna() & labels.astype(str).str.strip().ne("")
    table, labels = table.loc[valid], labels.loc[valid].astype(str).str.strip()
    if table.empty:
        raise ValueError(f"Events table contains no labeled trials: {events_file}")
    names = sorted(labels.unique())
    event_id = {name: index + 1 for index, name in enumerate(names)}
    onset_values = pd.to_numeric(table["onset"], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(onset_values).all():
        raise ValueError(f"Events table has invalid onset values: {events_file}")
    # BIDS specifies seconds, but some published datasets (including ds004196)
    # store millisecond values in their TSV despite the standard. Choose the
    # scale that places the greatest number of labeled events in this recording;
    # seconds wins ties, preserving standard BIDS behavior.
    duration_seconds = raw.n_times / raw.info["sfreq"]
    scales = (1.0, 1e-3, 1e-6)
    valid_counts = [
        int(np.count_nonzero((onset_values * scale >= 0) & (onset_values * scale < duration_seconds)))
        for scale in scales
    ]
    scale_index = int(np.argmax(valid_counts))
    if valid_counts[scale_index] == 0 or valid_counts[scale_index] < 0.9 * len(onset_values):
        raise ValueError(
            f"Could not map event onsets in {events_file} to the recording duration "
            f"({duration_seconds:.1f}s); check the onset units and event table."
        )
    onset_seconds = onset_values * scales[scale_index]
    sample = np.rint(onset_seconds * raw.info["sfreq"]).astype(int) + raw.first_samp
    recording_stop = raw.first_samp + raw.n_times
    inside = (sample >= raw.first_samp) & (sample < recording_stop)
    sample, labels = sample[inside], labels.iloc[np.flatnonzero(inside)]
    events = np.column_stack((sample, np.zeros(len(sample), dtype=int), [event_id[x] for x in labels]))
    if not len(events):
        raise ValueError(f"No labeled events fall within the recording: {events_file}")
    return events, event_id, names


def parse_and_load_eeg(run_info: dict[str, Any]):
    """Load one run and return raw EEG, events, labels, class names, and metadata."""
    eeg_file = run_info["eeg_file"]
    raw = _read_raw(eeg_file)
    events_file = run_info.get("events_file")
    if events_file and Path(events_file).is_file():
        events, event_id, class_names = _events_from_tsv(raw, events_file)
    else:
        events, event_id = mne.events_from_annotations(raw, verbose=False)
        if not len(events) or not event_id:
            raise ValueError(f"No events.tsv or usable event annotations for {eeg_file}")
        # Annotation labels are the source of truth; no fallback/synthetic classes.
        class_names = [name for name, _ in sorted(event_id.items(), key=lambda item: item[1])]
    event_names = {code: name for name, code in event_id.items()}
    labels = np.asarray([event_names[int(code)] for code in events[:, 2]], dtype=object)
    meta = {
        "run_id": run_info.get("run_id", Path(eeg_file).name),
        "filename": Path(eeg_file).name,
        "duration": float(raw.times[-1]),
        "sfreq": float(raw.info["sfreq"]),
        "subject": run_info.get("subject") or run_info.get("subj"),
        "channel_names": tuple(raw.ch_names),
    }
    return raw, events, labels, class_names, meta
