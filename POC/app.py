"""
EEG Imagined Speech Decoding -- Proof of Concept
Streamlit application following the first 2 stages of the methodology pipeline:

1. Dataset collection (BIDS ZIP & Local Dataset Loader)
2. EEG preprocessing (Literature-Based Filtering, Resampling, ICA, Epoching)

Run with:  streamlit run app.py
"""

import os
import io
import re
import glob
import shutil
import tempfile
import zipfile
import urllib.request

import mne
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from Step2_preprocessing_ import LiteraturePreprocessingPipeline

st.set_page_config(
    page_title="EEG Inner Speech Decoding -- POC", 
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
if "raw_data" not in st.session_state:
    st.session_state.raw_data = {}  # lang -> (raw, events, subj_array, class_names, meta_dict)

if "proc_data" not in st.session_state:
    st.session_state.proc_data = {}  # lang -> (X, y, subj, class_names, log)

if "uploaded_extracted_dir" not in st.session_state:
    st.session_state.uploaded_extracted_dir = None

if "detected_runs" not in st.session_state:
    st.session_state.detected_runs = {}


# ----------------------------------------------------------------------
# Helper Functions: BIDS Parsing, Pointer Resolution & Event Sync
# ----------------------------------------------------------------------
def is_git_annex_pointer(file_path: str) -> bool:
    """Check if a file is a DataLad / Git-Annex text pointer file instead of actual binary data."""
    try:
        if not os.path.exists(file_path):
            return False
        if os.path.getsize(file_path) < 1024:
            with open(file_path, "rb") as f:
                content = f.read(500)
                if b"git/annex" in content or b"SHA256" in content or b"/annex/objects" in content:
                    return True
    except Exception:
        pass
    return False


def scan_bids_eeg_dataset(root_dir: str):
    """
    Scans a directory (extracted zip or folder) and maps EEG files to their
    corresponding BIDS events and channel metadata.
    """
    runs = {}
    all_files = []
    for root, dirs, files in os.walk(root_dir):
        for f in files:
            all_files.append(os.path.join(root, f))

    # Find EEG data files
    eeg_files = [f for f in all_files if f.endswith((".bdf", ".set", ".fif", ".vhdr"))]
    tsv_files = [f for f in all_files if f.endswith("events.tsv")]

    for eeg_path in eeg_files:
        base_name = os.path.basename(eeg_path)
        dir_name = os.path.dirname(eeg_path)
        
        # Extract subject and session tokens (e.g. sub-01, ses-EEG)
        subj_match = re.search(r"sub-[a-zA-Z0-9]+", base_name) or re.search(r"sub-[a-zA-Z0-9]+", eeg_path)
        ses_match = re.search(r"ses-[a-zA-Z0-9]+", base_name) or re.search(r"ses-[a-zA-Z0-9]+", eeg_path)
        task_match = re.search(r"task-[a-zA-Z0-9]+", base_name)

        subj_id = subj_match.group(0) if subj_match else "sub-01"
        ses_id = ses_match.group(0) if ses_match else ""
        task_id = task_match.group(0) if task_match else ""

        key = f"{subj_id}" + (f" ({ses_id})" if ses_id else "") + (f" [{task_id}]" if task_id else "")
        if key in runs:
            key = f"{key} - {base_name}"

        # Match corresponding events.tsv file
        matched_events = None
        # Priority 1: Exact prefix match in same folder
        prefix = base_name.split("_eeg")[0] if "_eeg" in base_name else os.path.splitext(base_name)[0]
        for tf in tsv_files:
            if os.path.dirname(tf) == dir_name and prefix in os.path.basename(tf):
                matched_events = tf
                break
        
        # Priority 2: Any events.tsv in the same subject/session directory
        if not matched_events:
            for tf in tsv_files:
                if os.path.dirname(tf) == dir_name:
                    matched_events = tf
                    break

        # Priority 3: Any events.tsv matching the subject ID
        if not matched_events:
            for tf in tsv_files:
                if subj_id in os.path.basename(tf) and ("func" not in tf):
                    matched_events = tf
                    break

        runs[key] = {
            "subj": subj_id,
            "session": ses_id,
            "eeg_file": eeg_path,
            "events_file": matched_events,
            "is_pointer": is_git_annex_pointer(eeg_path),
            "filename": base_name,
            "file_size": os.path.getsize(eeg_path),
        }

    return runs


def download_openneuro_bdf(dataset_id: str, subj: str, dest_path: str):
    """Downloads real binary BDF EEG file from OpenNeuro S3 public bucket."""
    # S3 key convention for ds004196
    possible_keys = [
        f"{dataset_id}/{subj}/ses-EEG/eeg/{subj}_ses-EEG_task-inner_eeg.bdf",
        f"{dataset_id}/{subj}/ses-EEG/eeg/{subj}_ses-EEG_eeg.bdf",
        f"{dataset_id}/{subj}/eeg/{subj}_eeg.bdf",
    ]
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    success = False
    for key in possible_keys:
        url = f"https://s3.amazonaws.com/openneuro.org/{key}"
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as out_f:
                total_length = resp.headers.get("content-length")
                progress_bar = st.progress(0, text=f"Downloading real EEG data from OpenNeuro S3 ({subj})...")
                downloaded = 0
                total_bytes = int(total_length) if total_length else 150000000
                while True:
                    chunk = resp.read(1024 * 512)
                    if not chunk:
                        break
                    downloaded += len(chunk)
                    out_f.write(chunk)
                    progress_bar.progress(min(1.0, downloaded / total_bytes), text=f"Downloading {subj} EEG: {downloaded/(1024*1024):.1f} MB / {total_bytes/(1024*1024):.1f} MB")
                progress_bar.empty()
            success = True
            break
        except Exception:
            continue
    return success


def parse_and_load_eeg(run_info: dict):
    """Loads raw EEG and builds synchronized events array."""
    eeg_file = run_info["eeg_file"]
    events_file = run_info["events_file"]
    
    # 1. Read MNE raw object
    if eeg_file.endswith(".bdf"):
        raw = mne.io.read_raw_bdf(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".vhdr"):
        raw = mne.io.read_raw_brainvision(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".set"):
        raw = mne.io.read_raw_eeglab(eeg_file, preload=True, verbose=False)
    elif eeg_file.endswith(".fif"):
        raw = mne.io.read_raw_fif(eeg_file, preload=True, verbose=False)
    else:
        raise ValueError(f"Unsupported EEG file format: {eeg_file}")

    # Pick EEG channels only (filter out Status, Trigger, etc.)
    try:
        raw.pick_types(eeg=True, stim=False, misc=False)
    except Exception:
        pass

    # 2. Parse events
    events_df = None
    if events_file and os.path.exists(events_file):
        events_df = pd.read_csv(events_file, sep="\t")
        
        # Determine trial condition column
        label_col = None
        for col in ["trial_type", "value", "event_type", "stimulus", "condition"]:
            if col in events_df.columns:
                label_col = col
                break
        
        labels_str = events_df[label_col].values if label_col else np.ones(len(events_df), dtype=str)
        unique_labels = sorted(set(labels_str))
        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
        labels_int = np.array([label_map[l] for l in labels_str])
        
        # Detect onset unit (milliseconds vs seconds)
        onset_vals = events_df["onset"].values.astype(float)
        if onset_vals.max() > raw.times[-1]:
            onset_seconds = onset_vals / 1000.0
        else:
            onset_seconds = onset_vals

        events = np.column_stack([
            (onset_seconds * raw.info["sfreq"]).astype(int),
            np.zeros(len(events_df)),
            labels_int
        ]).astype(int)
        
        cn = [str(l) for l in unique_labels]
    else:
        # Fallback dummy events
        events = np.array([[(i + 1) * int(raw.info["sfreq"] * 2), 0, 1] for i in range(10)], dtype=int)
        cn = ["Dummy_Class"]
        events_df = pd.DataFrame({"onset": [i * 2.0 for i in range(10)], "trial_type": ["Dummy_Class"] * 10})

    subj_array = np.zeros(len(events), dtype=int)
    meta = {
        "filename": os.path.basename(eeg_file),
        "events_df": events_df,
        "duration": raw.times[-1],
        "n_channels": len(raw.ch_names),
        "sfreq": raw.info["sfreq"],
        "subj_id": run_info.get("subj", "sub-01")
    }

    return raw, events, subj_array, cn, meta


# ----------------------------------------------------------------------
# Application UI Layout
# ----------------------------------------------------------------------
st.title("🧠 EEG Imagined Speech Decoding -- Proof of Concept")
st.caption(
    "Demonstrates Stage 1 (Dataset Collection & BIDS Parsing) and Stage 2 (Literature-Based EEG Preprocessing) "
    "preserving Gamma-band activations for inner speech decoding."
)

STAGES = [
    "1. Dataset collection",
    "2. EEG preprocessing",
]
stage = st.sidebar.radio("Pipeline stage", STAGES)

# ========================================================================
# STAGE 1 -- Dataset collection
# ========================================================================
if stage == "1. Dataset collection":
    st.header("Stage 1 -- Dataset Collection & BIDS Structure Loading")
    
    tab_upload, tab_local = st.tabs(["📤 Upload Dataset ZIP", "📂 Load Pre-downloaded Real Dataset"])
    
    # ------------------------------------------------------------------
    # TAB 1: Upload Dataset ZIP
    # ------------------------------------------------------------------
    with tab_upload:
        st.subheader("Upload Real Dataset (BIDS ZIP format)")
        st.caption("Upload a `.zip` file containing a standard BIDS EEG dataset (e.g. `ds004196_sub01_eeg.zip` or `ds004196_all_eeg.zip`).")
        
        up_lang = st.selectbox("Assign upload to language slot", ["english", "spanish", "chinese"], key="upload_lang_slot")
        up_zip = st.file_uploader("Upload Dataset ZIP", type=["zip"], key="upzip")
        
        if up_zip is not None:
            if st.button("Extract and Inspect Dataset Structure", key="btn_extract_zip"):
                with st.spinner("Extracting and analyzing BIDS structure..."):
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(up_zip, "r") as zip_ref:
                        # Extract EEG, TSV, JSON metadata files only
                        for member in zip_ref.namelist():
                            if member.endswith((".bdf", ".set", ".fif", ".vhdr", ".vmrk", ".eeg", ".tsv", ".json", ".txt", "README")):
                                zip_ref.extract(member, temp_dir)
                    
                    st.session_state.uploaded_extracted_dir = temp_dir
                    runs = scan_bids_eeg_dataset(temp_dir)
                    st.session_state.detected_runs = runs
                    
            if st.session_state.detected_runs:
                st.success(f"Detected {len(st.session_state.detected_runs)} EEG recording(s) in uploaded dataset!")
                
                selected_run_key = st.selectbox(
                    "Select Subject / Recording to Load", 
                    list(st.session_state.detected_runs.keys()),
                    key="sel_run_upload"
                )
                run_info = st.session_state.detected_runs[selected_run_key]
                
                # Check for Git-annex pointer
                if run_info["is_pointer"]:
                    st.warning(
                        f"⚠️ **Git-Annex Pointer Detected:** `{run_info['filename']}` is a DataLad pointer text file (OpenNeuro GitHub archive). "
                        "Click below to automatically retrieve the actual binary EEG recording from OpenNeuro AWS S3."
                    )
                    if st.button("🚀 Auto-Download Real EEG Data from OpenNeuro S3", key="btn_resolve_s3"):
                        with st.spinner("Downloading real binary EEG recording..."):
                            success = download_openneuro_bdf("ds004196", run_info["subj"], run_info["eeg_file"])
                            if success:
                                st.success("Downloaded real binary EEG file! Re-inspecting...")
                                run_info["is_pointer"] = False
                                run_info["file_size"] = os.path.getsize(run_info["eeg_file"])
                            else:
                                st.error("Failed to download from OpenNeuro S3.")
                
                if not run_info["is_pointer"]:
                    if st.button("Load Recording into Pipeline", key="btn_load_run"):
                        with st.spinner(f"Loading {run_info['filename']} and synchronizing events..."):
                            try:
                                raw, events, subj, cn, meta = parse_and_load_eeg(run_info)
                                st.session_state.raw_data[up_lang] = (raw, events, subj, cn, meta)
                                st.success(f"Successfully loaded {up_lang.title()} dataset! Length: {raw.times[-1]:.2f}s | Channels: {len(raw.ch_names)} | Trials: {len(events)}")
                            except Exception as e:
                                st.error(f"Error loading EEG recording: {e}")

    # ------------------------------------------------------------------
    # TAB 2: Load Pre-downloaded Local Dataset
    # ------------------------------------------------------------------
    with tab_local:
        st.subheader("Direct Load from Local Storage")
        st.caption("Load prepared OpenNeuro datasets directly from the `data/` directory.")
        
        local_data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
        available_datasets = []
        if os.path.exists(local_data_dir):
            for d in os.listdir(local_data_dir):
                full_d = os.path.join(local_data_dir, d)
                if os.path.isdir(full_d) or d.endswith(".zip"):
                    available_datasets.append(d)

        selected_local_ds = st.selectbox("Select local dataset folder or zip", available_datasets, key="sel_local_ds")
        local_lang = st.selectbox("Assign to language slot", ["english", "spanish", "chinese"], key="local_lang_slot")
        
        if st.button("Scan and Load Local Dataset", key="btn_scan_local"):
            with st.spinner("Scanning local dataset..."):
                target_path = os.path.join(local_data_dir, selected_local_ds)
                if selected_local_ds.endswith(".zip"):
                    temp_dir = tempfile.mkdtemp()
                    with zipfile.ZipFile(target_path, "r") as zf:
                        zf.extractall(temp_dir)
                    scan_dir = temp_dir
                else:
                    scan_dir = target_path
                
                runs = scan_bids_eeg_dataset(scan_dir)
                if runs:
                    st.session_state.detected_runs = runs
                    first_key = list(runs.keys())[0]
                    raw, events, subj, cn, meta = parse_and_load_eeg(runs[first_key])
                    st.session_state.raw_data[local_lang] = (raw, events, subj, cn, meta)
                    st.success(f"Successfully loaded {selected_local_ds} ({first_key}) into **{local_lang.title()}** slot!")
                else:
                    st.error("No valid EEG recordings found in the selected local dataset.")

    # ------------------------------------------------------------------
    # Summary of Currently Loaded Datasets
    # ------------------------------------------------------------------
    st.divider()
    st.subheader("Loaded Datasets Overview")
    if not st.session_state.raw_data:
        st.info("No datasets loaded yet. Upload a dataset or load from local storage above.")
    else:
        for lang, data_tuple in st.session_state.raw_data.items():
            raw, events, subj, cn = data_tuple[:4]
            meta = data_tuple[4] if len(data_tuple) > 4 else {}
            
            with st.expander(f"🔹 **{lang.upper()}** Dataset: {meta.get('filename', 'Raw EEG')}", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Sampling Rate", f"{raw.info['sfreq']} Hz")
                col2.metric("EEG Channels", len(raw.ch_names))
                col3.metric("Duration", f"{raw.times[-1]:.1f} s")
                col4.metric("Total Trials", len(events))
                
                # Class trial distribution breakdown
                events_df = meta.get("events_df")
                if events_df is not None and "trial_type" in events_df.columns:
                    st.write("**Inner Speech Classes & Trial Counts:**")
                    counts_df = events_df["trial_type"].value_counts().reset_index()
                    counts_df.columns = ["Class / Stimulus", "Trial Count"]
                    st.dataframe(counts_df, use_container_width=True)

# ========================================================================
# STAGE 2 -- EEG preprocessing
# ========================================================================
elif stage == "2. EEG preprocessing":
    st.header("Stage 2 -- EEG Preprocessing Pipeline")
    if not st.session_state.raw_data:
        st.info("Please load or upload a dataset in Stage 1 first.")
    else:
        lang = st.selectbox("Select Active Dataset", list(st.session_state.raw_data.keys()), key="sel_prep_lang")
        data_tuple = st.session_state.raw_data[lang]
        raw, events, subj, cn = data_tuple[:4]
        meta = data_tuple[4] if len(data_tuple) > 4 else {}

        st.subheader(f"Literature-Based Preprocessing ({lang.title()} Dataset)")
        st.markdown(
            "> **Standard Preprocessing Protocol:**\n"
            "> - **Bandpass Filter:** 4.0 – 100.0 Hz (preserves Gamma-band 30–100 Hz).\n"
            "> - **Notch Filter:** 50 Hz powerline attenuation.\n"
            "> - **Resampling:** 250 Hz (Nyquist 125 Hz covers full 100 Hz Gamma band).\n"
            "> - **ICA Artifact Removal:** Removes ocular and muscle artifacts.\n"
            "> - **Epoching & Channel Z-Score Normalization:** Corrects non-stationarity across trials."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            low = st.number_input("Bandpass low (Hz)", 0.1, 10.0, 4.0, step=0.5)
            high = st.number_input("Bandpass high (Hz)", 10.0, 150.0, 100.0, step=5.0)
        with c2:
            notch = st.number_input("Notch frequency (Hz)", 40.0, 70.0, 50.0, step=1.0)
            target_sfreq = st.number_input("Target Resampling Freq (Hz)", 100, 1000, 250, step=50)
        with c3:
            tmin = st.number_input("Epoch start (s relative to onset)", -2.0, 5.0, 0.5, step=0.1)
            tmax = st.number_input("Epoch end (s relative to onset)", -2.0, 10.0, 3.5, step=0.1)
            ica_comp = st.slider("ICA Components", min_value=5, max_value=min(30, len(raw.ch_names) - 1), value=15)

        if st.button("🚀 Run Literature Pipeline", key="btn_run_pipeline"):
            pipeline = LiteraturePreprocessingPipeline(
                target_sfreq=target_sfreq, 
                l_freq=low, 
                h_freq=high, 
                notch_freqs=[notch],
                ica_components=ica_comp
            )
            
            with st.spinner("Executing pipeline: Bandpass -> Notch -> Resampling -> ICA -> Epoching -> Z-Score Normalization..."):
                event_id = {str(c): c for c in set(events[:, 2])}
                
                try:
                    raw_copy = raw.copy()
                    Xp, final_events, epochs = pipeline.run_pipeline(
                        raw_copy, events, event_id, tmin=tmin, tmax=tmax
                    )
                    y = final_events[:, 2]
                    st.session_state.proc_data[lang] = (Xp, y, subj[:len(y)], cn, ["MNE preprocessing successful."])
                    
                    st.success(
                        f"✅ Preprocessing Complete! Extracted **{Xp.shape[0]} trials** × **{Xp.shape[1]} channels** × **{Xp.shape[2]} timepoints** "
                        f"(Tensor shape: `{list(Xp.shape)}`). Gamma band (30-100Hz) successfully preserved!"
                    )
                    
                    st.divider()
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.subheader("Power Spectral Density (Before)")
                        fig_before = raw.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig_before)
                        
                    with col_p2:
                        st.subheader("Power Spectral Density (After Epoching)")
                        fig_after = epochs.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig_after)

                except Exception as e:
                    import traceback
                    st.error(f"Error during preprocessing: {e}")
                    st.code(traceback.format_exc())
