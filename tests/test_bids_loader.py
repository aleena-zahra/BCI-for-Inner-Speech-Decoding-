import zipfile

import mne
import numpy as np
import pandas as pd
import pytest

from voice.data.loader import _events_from_tsv, extract_bids_archive


def _raw(duration=10, sfreq=10):
    info = mne.create_info(["Cz"], sfreq, ch_types=["eeg"])
    return mne.io.RawArray(np.zeros((1, duration * sfreq)), info, verbose=False)


def test_events_tsv_detects_millisecond_onsets(tmp_path):
    events_path = tmp_path / "events.tsv"
    pd.DataFrame({
        "onset": [0, 4000, 8000],
        "duration": [4000, 4000, 2000],
        "trial_type": ["a", "b", "a"],
    }).to_csv(events_path, sep="\t", index=False)

    events, _, labels = _events_from_tsv(_raw(), str(events_path))

    assert events[:, 0].tolist() == [0, 40, 80]
    assert labels == ["a", "b"]


def test_events_tsv_keeps_standard_second_onsets(tmp_path):
    events_path = tmp_path / "events.tsv"
    pd.DataFrame({
        "onset": [0, 4, 8],
        "duration": [4, 4, 2],
        "trial_type": ["a", "b", "a"],
    }).to_csv(events_path, sep="\t", index=False)

    events, _, _ = _events_from_tsv(_raw(), str(events_path))

    assert events[:, 0].tolist() == [0, 40, 80]


def test_non_bids_zip_is_rejected_before_extraction(tmp_path):
    archive = tmp_path / "mat_dataset.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("dataset/sample.mat", b"placeholder")

    with pytest.raises(ValueError, match="no supported BIDS EEG recording"):
        extract_bids_archive(archive, tmp_path / "out")

