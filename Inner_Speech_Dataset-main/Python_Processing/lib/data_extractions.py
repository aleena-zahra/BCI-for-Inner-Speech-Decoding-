# -*- coding: utf-8 -*-

"""
@author: Nieto Nicolás
@email: nnieto@sinc.unl.edu.ar

Utilitys from extract, read and load data from Inner Speech Dataset
"""

import mne
import gc
import os
import numpy as np
import pandas as pd
import pickle
import warnings
from mne.io import Raw
from pathlib import Path
from lib.utils import sub_name, unify_names


def extract_subject_from_bdf(data_dir: Path, n_s: int, n_b: int) -> tuple[Raw, str]:
    """
    Extracts raw EEG data from a BDF file for a specific subject and block.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s (int): The subject number.
    - n_b (int): The block number.

    Returns:
    - tuple: A tuple containing raw EEG data and the corrected subject name.
    """
    # Name correction if N_Subj is less than 10
    num_s = sub_name(n_s)

    # Load data
    file_name = (
        data_dir / f"{num_s}/ses-0{n_b}/eeg/{num_s}_ses-0{n_b}_task-innerspeech_eeg.bdf"  # noqa
    )
    raw_data = mne.io.read_raw_bdf(
        input_fname=file_name, preload=True, verbose="WARNING"
    )

    return raw_data, num_s


def extract_data_from_subject(root_dir: Path, n_s: int, datatype: str) -> tuple:
    """
    Load all blocks for one subject and stack the results in X.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s (int): The subject number.
    - datatype (str): The type of data to extract ("eeg", "exg", or "baseline")

    Returns:
    - tuple: A tuple containing the stacked data (X) and events (Y).
    """
    data = dict()
    y = dict()
    n_b_arr = [1, 2, 3]
    datatype = datatype.lower()

    for n_b in n_b_arr:
        # Name correction if N_Subj is less than 10
        num_s = sub_name(n_s)

        y[n_b] = load_events(root_dir, n_s, n_b)

        if datatype == "eeg":
            file_name = f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_eeg-epo.fif"  # noqa
        elif datatype == "exg":
            file_name = f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_exg-epo.fif"  # noqa
        elif datatype == "baseline":
            file_name = f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}_baseline-epo.fif"  # noqa
        else:
            raise ValueError("Invalid Datatype")

        try:
            X = mne.read_epochs(file_name, verbose="WARNING")
            X._data = X._data.astype(np.float32, copy=False)
        except Exception as exc:
            warnings.warn(
                f"Skipping unreadable file for subject {num_s}, session {n_b}: {file_name} ({exc})",
                RuntimeWarning,
            )
            continue

        data[n_b] = X._data

    if not data:
        raise ValueError(f"No readable {datatype} files found for subject {n_s}")

    X_stacked = np.vstack([data[n_b] for n_b in n_b_arr if n_b in data])
    Y_stacked = np.vstack([y[n_b] for n_b in n_b_arr if n_b in data])

    return X_stacked, Y_stacked


def extract_block_data_from_subject(
    root_dir: Path, n_s: int, datatype: str, n_b: int
) -> tuple:
    """
    Load selected block from one subject.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s (int): The subject number.
    - datatype (str): The type of data to extract ("eeg", "exg", or "baseline")
    - n_b (int): The block number.

    Returns:
    - tuple: A tuple containing the loaded data (X) and events (Y).
    """
    # Get subject name
    num_s = sub_name(n_s)

    # Standarize datatype
    datatype = datatype.lower()

    # Get events
    y = load_events(root_dir, n_s, n_b)

    sub_dir = root_dir / "derivatives" / num_s / f"ses-0{n_b}"

    if datatype == "eeg":
        # Load EEG data
        file_name = sub_dir / f"{num_s}_ses-0{n_b}_eeg-epo.fif"
        X = mne.read_epochs(file_name, verbose="WARNING")

    elif datatype == "exg":
        # Load EXG data
        file_name = sub_dir / f"{num_s}_ses-0{n_b}_exg-epo.fif"
        X = mne.read_epochs(file_name, verbose="WARNING")

    elif datatype == "baseline":
        # Load Baseline data
        file_name = sub_dir / f"{num_s}_ses-0{n_b}_baseline-epo.fif"
        X = mne.read_epochs(file_name, verbose="WARNING")

    else:
        raise ValueError("Invalid Datatype")

    return X, y


def extract_report(root_dir: Path, n_b: int, n_s: int):
    """
    Extract a report for a specific block and subject.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_b (int): The block number.
    - n_s (int): The subject number.

    Returns:
    - dict: The loaded report.
    """
    # Get subject name
    num_s = sub_name(n_s)

    # Save report
    sub_dir = root_dir / "derivatives" / num_s / f"ses-0{n_b}"
    file_name = sub_dir / f"{num_s}_ses-0{n_b}_report.pkl"

    with open(file_name, "rb") as input_file:
        report = pickle.load(input_file)

    return report


def extract_tfr(
    trf_dir: Path, cond: str, class_label: str, tfr_method: str, trf_type: str
) -> mne.time_frequency:
    """
    Extract Time-Frequency Representation (TFR) data.

    Parameters:
    - trf_dir (str): The directory containing the TFR data.
    - cond (str): The condition.
    - class_label (str): The class label.
    - tfr_method (str): The TFR method used.
    - trf_type (str): The type of TRF.

    Returns:
    - mne.time_frequency.tfr.TFR: The extracted TFR data.
    """
    # Unify names as stored
    cond, class_label = unify_names(cond, class_label)

    fname = trf_dir / f"{tfr_method}_{cond}_{class_label}_{trf_type}-tfr.h5"

    trf = mne.time_frequency.read_tfrs(fname)[0]

    return trf


def extract_data_multisubject(
    root_dir: Path, n_s_list: list, datatype: str = "eeg"
) -> tuple:
    """
    Load all blocks for a list of subjects and stack the results.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s_list (list): List of subject numbers.
    - datatype (str): The type of data to extract ("eeg", "exg", or "baseline")

    Returns:
    - tuple: Tuple containing the stacked data (X) and events (Y) if applicable
    """
    n_b_arr = [1, 2, 3]
    datatype = datatype.lower()

    valid_x = []
    valid_y = []
    chann = None
    steps = None
    columns = None

    for s, n_s in enumerate(n_s_list):
        print("Iteration ", s)
        print("Subject ", n_s)
        for n_b in n_b_arr:
            num_s = sub_name(n_s)

            base_file_name = (
                f"{root_dir}/derivatives/{num_s}/ses-0{n_b}/{num_s}_ses-0{n_b}"  # noqa
            )
            events_file_name = f"{base_file_name}_events.dat"

            try:
                data_tmp_Y = np.load(events_file_name, allow_pickle=True)
            except Exception as exc:
                warnings.warn(
                    f"Skipping unreadable events file for subject {num_s}, session {n_b}: {events_file_name} ({exc})",
                    RuntimeWarning,
                )
                continue

            print("Inner iteration ", n_b)

            try:
                if datatype == "eeg":
                    eeg_file_name = f"{base_file_name}_eeg-epo.fif"
                    data_tmp_X = mne.read_epochs(eeg_file_name, verbose="WARNING")._data.astype(np.float32, copy=False)  # noqa
                elif datatype == "exg":
                    exg_file_name = f"{base_file_name}_exg-epo.fif"
                    data_tmp_X = mne.read_epochs(exg_file_name, verbose="WARNING")._data.astype(np.float32, copy=False)  # noqa
                elif datatype == "baseline":
                    baseline_file_name = f"{base_file_name}_baseline-epo.fif"
                    data_tmp_X = mne.read_epochs(
                        baseline_file_name, verbose="WARNING"
                    )._data.astype(np.float32, copy=False)  # noqa
                else:
                    raise ValueError("Invalid Datatype")
            except Exception as exc:
                warnings.warn(
                    f"Skipping unreadable file for subject {num_s}, session {n_b}: {base_file_name} ({exc})",
                    RuntimeWarning,
                )
                continue

            if chann is None:
                chann = data_tmp_X.shape[1]
                steps = data_tmp_X.shape[2]
                if datatype in {"eeg", "exg"}:
                    columns = data_tmp_Y.shape[1]

            valid_x.append(data_tmp_X)
            if datatype in {"eeg", "exg"}:
                valid_y.append(data_tmp_Y)

        gc.collect()

    if not valid_x:
        raise ValueError("No readable files found for the requested subjects")

    x = np.vstack(valid_x)
    print("X shape", x.shape)

    if datatype in {"eeg", "exg"}:
        y = np.vstack(valid_y)
        print("Y shape", y.shape)
        return x, y

    return x


def get_events_from_raw(rawdata, N_S, N_B):
    # Subject 10  on Block 1 have a spureos trigger
    if N_S == 10 and N_B == 1:
        events = mne.find_events(
            rawdata, initial_event=True, consecutive=True, min_duration=0.002
        )
        # The different load of the events delet
        # the spureos trigger but also the Baseline finish mark
    else:
        events = mne.find_events(rawdata, initial_event=True, consecutive=True)

    events = pd.DataFrame(events, columns=["Time", "Trigger", "Code"])

    return events


def load_events(root_dir: Path, n_s: int, n_b: int):
    """
    Load events data for a specific subject and block.

    Parameters:
    - root_dir (str): The root directory containing the data.
    - n_s (int): The subject number.
    - n_b (int): The block number.

    Returns:
    - np.ndarray: The loaded events.
    """
    num_s = sub_name(n_s)

    # Create file name
    file_name = os.path.join(
        root_dir, "derivatives", num_s, f"ses-0{n_b}", f"{num_s}_ses-0{n_b}_events.dat"
    )

    # Load events
    events = pd.read_pickle(file_name)

    return events


def get_age_gender(N_S: int) -> tuple[int, str]:
    """
    Retrieve the age and gender of a subject based on their subject number.

    Demographic information
    Subject_age = [56, 50, 34, 24, 31, 29, 26, 28, 35, 31];
    Subject_gender = ["F", "M", "M", "F", "F", "M", "M", "F", "M", "M"]

    Parameters:
    - N_S (int): The subject number.

    Returns:
    - tuple: A tuple containing the age (int) and gender (str) of the subject.
    """
    # Fixed demographic information
    Subject_age = [56, 50, 34, 24, 31, 29, 26, 28, 35, 31]
    Subject_gender = ["F", "M", "M", "F", "F", "M", "M", "F", "M", "M"]
    # Retrieve age and gender for the corresponding subject number
    age = Subject_age[N_S - 1]
    gender = Subject_gender[N_S - 1]
    return age, gender
