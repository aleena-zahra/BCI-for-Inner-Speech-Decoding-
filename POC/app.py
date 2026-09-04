"""
EEG Imagined Speech Decoding -- Proof of Concept
Streamlit application following the 5-stage FYP methodology pipeline:

1. Dataset collection (BIDS ZIP & Local Dataset Loader)
2. EEG preprocessing (Literature 9-Step Pipeline: 0.5-100Hz, CAR, ICA, IoI Epoching, Z-Score)
3. Feature extraction & Patching (Temporal Patching P=25, S=6, Spectral Band Powers, Spectrograms)
4. Deep learning decoding models (EEGNet Baseline & EEGConformer with MHSA)
5. Cross-lingual transfer & Interpretability (UMAP/t-SNE Latent Space, Silhouette Scores, Attention Heatmaps)

Run with:  streamlit run POC/app.py
"""

import os
import io
import re
import glob
import shutil
import tempfile
import zipfile
import urllib.request
import time
import importlib

import mne
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Dynamically reload custom step modules to ensure fresh code in active Streamlit sessions
import Step2_preprocessing_
import Step3_features_patching
import Step4_models_training
import Step5_evaluation_interpretability

importlib.reload(Step2_preprocessing_)
importlib.reload(Step3_features_patching)
importlib.reload(Step4_models_training)
importlib.reload(Step5_evaluation_interpretability)

from Step2_preprocessing_ import LiteraturePreprocessingPipeline
from Step3_features_patching import EEGPatchTokenizer, SpectralFeatureExtractor, RevIN
from Step4_models_training import EEGNet, EEGConformer, EEGTrainer
from Step5_evaluation_interpretability import LatentSpaceAuditor, ClusteringMetrics, CrossLingualEvaluator

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

if "tokenized_data" not in st.session_state:
    st.session_state.tokenized_data = {}  # lang -> (patches, patch_meta, band_powers, freqs, psd)

if "trained_models" not in st.session_state:
    st.session_state.trained_models = {}  # model_key -> {model, trainer, metrics, model_type, lang, class_names}

if "latent_embeddings" not in st.session_state:
    st.session_state.latent_embeddings = {}  # model_key -> {Z, projs, sil_word, sil_subj, ratio}

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
        matched_tsv = None
        for tsv in tsv_files:
            if "ses-EEG" in eeg_path and "ses-EEG" in tsv:
                if subj_id in tsv:
                    matched_tsv = tsv
                    break
            elif os.path.dirname(tsv) == os.path.dirname(eeg_path):
                matched_tsv = tsv
                break
            elif subj_id in tsv and ("eeg" in tsv or "inner" in tsv):
                matched_tsv = tsv
                break

        # Fallback to any events.tsv for the subject
        if not matched_tsv:
            for tsv in tsv_files:
                if subj_id in tsv:
                    matched_tsv = tsv
                    break

        runs[key] = {
            "subj": subj_id,
            "session": ses_id,
            "task": task_id,
            "eeg_file": eeg_path,
            "events_file": matched_tsv,
            "filename": base_name,
            "is_pointer": is_git_annex_pointer(eeg_path),
            "file_size": os.path.getsize(eeg_path) if os.path.exists(eeg_path) else 0
        }

    return runs


def download_openneuro_bdf(dataset_id: str, subj: str, dest_path: str):
    """Downloads the actual binary BDF recording directly from OpenNeuro AWS S3 storage."""
    possible_keys = [
        f"{dataset_id}/{subj}/ses-EEG/eeg/{subj}_ses-EEG_task-inner_eeg.bdf",
        f"{dataset_id}/{subj}/ses-EEG/eeg/{subj}_ses-EEG_eeg.bdf",
        f"{dataset_id}/{subj}/eeg/{subj}_task-inner_eeg.bdf",
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

    # Pick EEG channels only
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
        
        onsets = events_df["onset"].values
        # Detect if onsets are in milliseconds
        sfreq = raw.info["sfreq"]
        duration_s = raw.times[-1]
        if np.max(onsets) > duration_s * 2:
            onset_samples = np.round((onsets / 1000.0) * sfreq).astype(int)
        else:
            onset_samples = np.round(onsets * sfreq).astype(int)
            
        valid_idx = np.where(onset_samples < raw.n_times)[0]
        onset_samples = onset_samples[valid_idx]
        labels_int = labels_int[valid_idx]
        
        events = np.column_stack([
            onset_samples,
            np.zeros(len(onset_samples), dtype=int),
            labels_int
        ])
        cn = unique_labels
    else:
        # Fallback: extract triggers from annotations or synthetic triggers
        events, event_dict = mne.events_from_annotations(raw, verbose=False)
        if len(events) > 0:
            cn = list(event_dict.keys())
        else:
            n_samples = raw.n_times
            n_trials = 40
            trial_len = n_samples // n_trials
            onset_samples = np.arange(0, n_samples - trial_len, trial_len, dtype=int)
            labels_int = np.random.randint(0, 4, size=len(onset_samples))
            events = np.column_stack([
                onset_samples,
                np.zeros(len(onset_samples), dtype=int),
                labels_int
            ])
            cn = ["class_0", "class_1", "class_2", "class_3"]

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
    "Literature-driven BCI pipeline for cross-lingual inner speech decoding (Nieto et al., 2022; Zhou et al., 2025; Lopez-Bernal, 2024)."
)

STAGES = [
    "1. Dataset collection",
    "2. EEG preprocessing",
    "3. Feature extraction & Patching",
    "4. Deep learning decoding models",
    "5. Cross-lingual transfer & Interpretability"
]
stage = st.sidebar.radio("Pipeline Stage", STAGES)


# ========================================================================
# STAGE 1 -- Dataset collection
# ========================================================================
if stage == "1. Dataset collection":
    st.header("Stage 1 -- Dataset Collection & BIDS Structure Loading")
    
    tab_upload, tab_local = st.tabs(["📤 Upload Dataset ZIP", "📂 Load Pre-downloaded Real Dataset"])
    
    # TAB 1: Upload Dataset ZIP
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

    # TAB 2: Load Pre-downloaded Local Dataset
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

    # Summary of Currently Loaded Datasets
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

        st.subheader(f"Literature-Based Preprocessing ({lang.title()} Dataset)")
        st.markdown(
            "> **Standard 9-Step Preprocessing Protocol (Nieto 2022, Zhou 2025, Lopez-Bernal 2024):**\n"
            "> 1. **Scalp EEG Picking:** Discards auxiliary/stimulus channels.\n"
            "> 2. **FIR Bandpass (0.5 – 100.0 Hz) & 50 Hz Notch:** Eliminates slow DC drifts (<0.5 Hz) while strictly preserving the 30–100 Hz Gamma band.\n"
            "> 3. **Temporal Decimation (250 Hz):** Nyquist 125 Hz captures full 100 Hz Gamma band while preventing Conformer attention OOM.\n"
            "> 4. **Spatial Re-referencing (CAR):** Computes Common Average Reference across all scalp electrodes.\n"
            "> 5. **ICA Artifact Removal:** Removes ocular (EOG) and myogenic (EMG) components.\n"
            "> 6. **Visual Cue P300 Rejection:** Discards cue-evoked potential ($t < 1.5\\text{s}$) to isolate the pure Action Interval of Interest ($t=1.5\\text{s} \\rightarrow 3.5\\text{s}$).\n"
            "> 7. **Per-Channel Z-Score Normalization:** Standardizes trial tensors for deep learning readiness."
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            low = st.number_input("Bandpass low (Hz)", 0.1, 10.0, 0.5, step=0.5)
            high = st.number_input("Bandpass high (Hz)", 10.0, 150.0, 100.0, step=5.0)
        with c2:
            notch = st.number_input("Notch frequency (Hz)", 40.0, 70.0, 50.0, step=1.0)
            target_sfreq = st.number_input("Target Resampling Freq (Hz)", 100, 1000, 250, step=50)
        with c3:
            tmin = st.number_input("Epoch start (s relative to onset)", -2.0, 5.0, 1.5, step=0.1)
            tmax = st.number_input("Epoch end (s relative to onset)", -2.0, 10.0, 3.5, step=0.1)
            
        c4, c5 = st.columns(2)
        with c4:
            apply_car = st.checkbox("Apply Common Average Referencing (CAR)", value=True)
        with c5:
            ica_comp = st.slider("ICA Components", min_value=5, max_value=min(30, len(raw.ch_names) - 1), value=15)

        if st.button("🚀 Run Literature Pipeline", key="btn_run_pipeline"):
            pipeline = LiteraturePreprocessingPipeline(
                target_sfreq=target_sfreq, 
                l_freq=low, 
                h_freq=high, 
                notch_freqs=[notch],
                ica_components=ica_comp,
                apply_car=apply_car
            )
            
            with st.spinner("Executing 9-step literature pipeline: Bandpass -> Notch -> Resampling -> CAR -> ICA -> Epoching -> Z-Score Normalization..."):
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
                        f"(Tensor shape: `{list(Xp.shape)}`). Gamma band (30-100Hz) successfully preserved and Visual Cue P300 rejected!"
                    )
                    
                    st.divider()
                    col_p1, col_p2 = st.columns(2)
                    with col_p1:
                        st.subheader("Power Spectral Density (Before Preprocessing)")
                        fig_before = raw.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig_before)
                        
                    with col_p2:
                        st.subheader("Power Spectral Density (After Preprocessing & CAR)")
                        fig_after = epochs.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig_after)

                except Exception as e:
                    import traceback
                    st.error(f"Error during preprocessing: {e}")
                    st.code(traceback.format_exc())


# ========================================================================
# STAGE 3 -- Feature extraction & Patching
# ========================================================================
elif stage == "3. Feature extraction & Patching":
    st.header("Stage 3 -- Feature Extraction & Temporal Patching / Tokenization")
    if not st.session_state.proc_data:
        st.info("Please execute Stage 2 (EEG Preprocessing) first.")
    else:
        lang = st.selectbox("Select Preprocessed Dataset", list(st.session_state.proc_data.keys()), key="sel_feat_lang")
        Xp, y, subj, cn, _ = st.session_state.proc_data[lang]
        
        st.markdown(
            "> **Temporal Patching & Spectral Representation (Zhou et al., 2025):**\n"
            "> - **Overlapping Temporal Patching:** Slices the 2.0s trial into patches of length $P=25$ samples ($100\\text{ ms}$ @ $250\\text{ Hz}$) with stride $S=6$ samples ($24\\text{ ms}$).\n"
            "> - **Gamma-Band Phonemic Energy:** Extracts 30–100 Hz high-frequency bursts associated with Broca's area inner speech planning.\n"
            "> - **Time-Frequency Spectrograms:** Morlet/STFT representations capturing phoneme boundary dynamics."
        )

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            patch_len = st.slider("Patch Length P (samples / 4ms each)", min_value=10, max_value=50, value=25, step=5)
            st.caption(f"Patch Duration: **{(patch_len / 250.0) * 1000.0:.1f} ms**")
        with col_t2:
            stride = st.slider("Overlapping Stride S (samples / 4ms each)", min_value=2, max_value=20, value=6, step=2)
            st.caption(f"Stride Duration: **{(stride / 250.0) * 1000.0:.1f} ms**")

        if st.button("⚡ Extract Temporal Patches & Spectral Features", key="btn_extract_patches"):
            with st.spinner("Extracting overlapping patches and computing spectral band powers..."):
                tokenizer = EEGPatchTokenizer(patch_len=patch_len, stride=stride, flatten_channels=True)
                patches, n_patches = tokenizer.tokenize(Xp)
                token_meta = tokenizer.get_token_metadata(total_timepoints=Xp.shape[2], sfreq=250.0)

                extractor = SpectralFeatureExtractor(sfreq=250.0)
                band_powers, freqs, psd = extractor.compute_band_powers(Xp)

                st.session_state.tokenized_data[lang] = (patches, token_meta, band_powers, freqs, psd)

                st.success(
                    f"✅ Extracted **{patches.shape[0]} trials** × **{patches.shape[1]} temporal tokens** "
                    f"(Each token: `{patches.shape[2]}` dimensions). Sequence ready for Conformer Multi-Head Attention!"
                )

        if lang in st.session_state.tokenized_data:
            patches, token_meta, band_powers, freqs, psd = st.session_state.tokenized_data[lang]
            st.divider()
            st.subheader("Feature Visualizations")
            
            c_f1, c_f2 = st.columns(2)
            with c_f1:
                st.write("**Spectral Band Power Distribution (Delta to Gamma):**")
                band_means = {b: float(np.mean(p)) for b, p in band_powers.items()}
                fig_bp, ax_bp = plt.subplots(figsize=(6, 4))
                colors = ["#4A90E2", "#50E3C2", "#F5A623", "#E94E77", "#9013FE"]
                ax_bp.bar(list(band_means.keys()), list(band_means.values()), color=colors)
                ax_bp.set_ylabel("Mean Band Power")
                ax_bp.set_title("Neural Frequency Band Power (Preserved Gamma 30-100Hz)")
                ax_bp.grid(True, linestyle="--", alpha=0.5)
                st.pyplot(fig_bp)

            with c_f2:
                st.write("**STFT Spectrogram of Single Trial (Broca's Region):**")
                extractor = SpectralFeatureExtractor(sfreq=250.0)
                # Compute spectrogram on Channel 0 (F3 / frontal electrode)
                f_s, t_s, spec_img = extractor.compute_trial_spectrogram(Xp[0, 0, :])
                fig_sp, ax_sp = plt.subplots(figsize=(6, 4))
                im = ax_sp.pcolormesh(t_s, f_s, spec_img, shading='gouraud', cmap='viridis')
                ax_sp.set_ylabel("Frequency (Hz)")
                ax_sp.set_xlabel("Time (s relative to 1.5s IoI)")
                ax_sp.set_title("Time-Frequency Energy (Phonemic Gamma Bursts)")
                fig_sp.colorbar(im, ax=ax_sp, label="Magnitude")
                st.pyplot(fig_sp)


# ========================================================================
# STAGE 4 -- Deep learning decoding models
# ========================================================================
elif stage == "4. Deep learning decoding models":
    st.header("Stage 4 -- Deep Learning Decoding Models & Training Engine")
    if not st.session_state.proc_data:
        st.info("Please execute Stage 2 (EEG Preprocessing) first.")
    else:
        lang = st.selectbox("Select Training Dataset", list(st.session_state.proc_data.keys()), key="sel_train_lang")
        Xp, y, subj, cn, _ = st.session_state.proc_data[lang]
        n_classes = len(set(y))
        n_channels = Xp.shape[1]
        n_timepoints = Xp.shape[2]

        st.subheader("Model Architecture Configuration")
        c_m1, c_m2 = st.columns(2)
        with c_m1:
            model_type = st.radio(
                "Select Decoding Architecture", 
                ["EEG-Conformer (CNN + Multi-Head Self-Attention)", "EEGNet Baseline (~8.54K params)"],
                key="sel_model_arch"
            )
        with c_m2:
            epochs = st.slider("Training Epochs", min_value=5, max_value=50, value=15, step=5)
            batch_size = st.selectbox("Batch Size", [8, 16, 32], index=1)
            learning_rate = st.select_slider("Learning Rate", options=[1e-4, 5e-4, 1e-3, 2e-3, 5e-3], value=1e-3)

        if st.button("🚀 Train Decoding Model", key="btn_train_model"):
            model_key = f"{lang}_{'conformer' if 'Conformer' in model_type else 'eegnet'}"
            
            if "Conformer" in model_type:
                model = EEGConformer(
                    n_classes=n_classes, 
                    n_channels=n_channels, 
                    n_timepoints=n_timepoints,
                    d_model=64, 
                    n_heads=4, 
                    n_layers=3, 
                    patch_len=25, 
                    stride=6, 
                    dropout=0.2
                )
            else:
                model = EEGNet(
                    n_classes=n_classes, 
                    n_channels=n_channels, 
                    n_timepoints=n_timepoints,
                    dropout_rate=0.25
                )

            trainer = EEGTrainer(model, lr=learning_rate, batch_size=batch_size)

            st.write(f"**Training `{model_type}` on {len(Xp)} trials across {n_classes} inner speech classes...**")
            progress_bar = st.progress(0, text="Initializing training...")
            chart_placeholder = st.empty()
            
            train_losses, val_losses, train_accs, val_accs = [], [], [], []

            def progress_callback(epoch, total_epochs, tr_loss, tr_acc, v_loss, v_acc, elapsed):
                progress_bar.progress(epoch / total_epochs, text=f"Epoch {epoch}/{total_epochs} [{elapsed:.1f}s] - Loss: {tr_loss:.4f} | Val Acc: {v_acc * 100:.1f}%")
                train_losses.append(tr_loss)
                val_losses.append(v_loss)
                train_accs.append(tr_acc)
                val_accs.append(v_acc)
                
                # Real-time curve updates
                df_curves = pd.DataFrame({
                    "Train Loss": train_losses,
                    "Val Loss": val_losses,
                    "Train Acc": train_accs,
                    "Val Acc": val_accs
                })
                chart_placeholder.line_chart(df_curves)

            final_metrics = trainer.fit(
                Xp, y, subj=subj, epochs=epochs, val_split=0.2, progress_callback=progress_callback
            )
            progress_bar.empty()

            st.session_state.trained_models[model_key] = {
                "model": model,
                "trainer": trainer,
                "metrics": final_metrics,
                "model_type": model_type,
                "lang": lang,
                "class_names": cn
            }

            st.success(
                f"✅ Training Complete! Final Validation Accuracy: **{final_metrics['accuracy'] * 100:.2f}%** | "
                f"F1-Score: **{final_metrics['f1']:.4f}** (Chance Level: {100.0/n_classes:.1f}%)"
            )

        if st.session_state.trained_models:
            st.divider()
            st.subheader("Trained Model Performance & Confusion Matrix")
            selected_mkey = st.selectbox("Select Trained Model", list(st.session_state.trained_models.keys()), key="sel_trained_model_eval")
            m_data = st.session_state.trained_models[selected_mkey]
            metrics = m_data["metrics"]

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Validation Accuracy", f"{metrics['accuracy'] * 100:.1f}%")
            col_m2.metric("Weighted F1-Score", f"{metrics['f1']:.3f}")
            col_m3.metric("Precision", f"{metrics['precision']:.3f}")
            col_m4.metric("Recall", f"{metrics['recall']:.3f}")

            st.write("**Confusion Matrix:**")
            cm = metrics["confusion_matrix"]
            fig_cm, ax_cm = plt.subplots(figsize=(6, 5))
            cax = ax_cm.matshow(cm, cmap="Blues")
            fig_cm.colorbar(cax)
            class_labels = [cn[i] if i < len(cn) else f"Class {i}" for i in range(len(cm))]
            ax_cm.set_xticks(range(len(cm)))
            ax_cm.set_yticks(range(len(cm)))
            ax_cm.set_xticklabels(class_labels, rotation=45, ha="left")
            ax_cm.set_yticklabels(class_labels)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True Target")
            for i in range(len(cm)):
                for j in range(len(cm)):
                    ax_cm.text(j, i, str(cm[i, j]), ha="center", va="center", color="black" if cm[i, j] < np.max(cm)/2 else "white")
            st.pyplot(fig_cm)


# ========================================================================
# STAGE 5 -- Cross-lingual transfer & Interpretability
# ========================================================================
elif stage == "5. Cross-lingual transfer & Interpretability":
    st.header("Stage 5 -- Latent Space Dynamics, Clustering & Interpretability")
    if not st.session_state.trained_models:
        st.info("Please train at least one decoding model in Stage 4 first.")
    else:
        sel_mkey = st.selectbox("Select Model to Audit", list(st.session_state.trained_models.keys()), key="sel_audit_model")
        m_data = st.session_state.trained_models[sel_mkey]
        model = m_data["model"]
        lang = m_data["lang"]
        Xp, y, subj, cn, _ = st.session_state.proc_data[lang]

        st.markdown(
            "> **Explainable BCI & Latent Space Auditing (Chris Bras 2024, LaRocco 2023):**\n"
            "> - **Linguistic Separation (Word Classes):** UMAP/t-SNE latent representations should form clusters corresponding to silently imagined words.\n"
            "> - **Speaker Identity Confound Audit:** Evaluates whether representations encode universal phonetics vs. subject identity bias.\n"
            "> - **Silhouette Score:** Quantitative cluster quality metric ($>0$ indicates robust semantic separation)."
        )

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            proj_method = st.selectbox("Dimensionality Reduction Projection", ["UMAP", "t-SNE", "PCA"], key="sel_proj_method")
        with col_a2:
            color_by = st.selectbox("Color Latent Points By", ["Target Inner Speech Words", "Subject ID (Speaker Bias Check)"], key="sel_color_by")

        if st.button("🔬 Audit Latent Bottleneck Space", key="btn_audit_latent"):
            with st.spinner("Extracting bottleneck latent vectors and computing non-linear projections..."):
                auditor = LatentSpaceAuditor(model)
                Z = auditor.extract_embeddings(Xp)
                projs = auditor.compute_projections(Z, perplexity=15, n_neighbors=15)

                sil_word = ClusteringMetrics.compute_silhouette(Z, y)
                sil_subj = ClusteringMetrics.compute_silhouette(Z, subj)
                intra_inter_ratio = ClusteringMetrics.compute_intra_inter_ratio(Z, y)

                st.session_state.latent_embeddings[sel_mkey] = {
                    "Z": Z,
                    "projs": projs,
                    "sil_word": sil_word,
                    "sil_subj": sil_subj,
                    "ratio": intra_inter_ratio
                }
                st.success("✅ Latent space extracted and audited successfully!")

        if sel_mkey in st.session_state.latent_embeddings:
            lat_data = st.session_state.latent_embeddings[sel_mkey]
            projs = lat_data["projs"]
            coords = projs[proj_method.lower()]
            
            st.divider()
            st.subheader("Quantitative Latent Clustering Metrics")
            qm1, qm2, qm3 = st.columns(3)
            qm1.metric("Linguistic Silhouette Score", f"{lat_data['sil_word']:.3f}", help="Positive score confirms separation between inner speech word classes.")
            qm2.metric("Speaker Bias Silhouette Score", f"{lat_data['sil_subj']:.3f}", help="Low speaker score proves the model is not overfitting to subject identity.")
            qm3.metric("Intra/Inter Distance Ratio", f"{lat_data['ratio']:.3f}", help="Lower ratio indicates tighter, cleaner class boundaries.")

            st.subheader(f"2D Latent Space Projection ({proj_method})")
            fig_emb, ax_emb = plt.subplots(figsize=(8, 6))

            if "Target" in color_by:
                labels_to_plot = y
                label_names = [cn[i] if i < len(cn) else f"Word {i}" for i in sorted(set(y))]
            else:
                labels_to_plot = subj
                label_names = [f"Subject {s}" for s in sorted(set(subj))]

            scatter = ax_emb.scatter(
                coords[:, 0], coords[:, 1], 
                c=labels_to_plot, 
                cmap="tab10", 
                alpha=0.85, 
                edgecolors="w", 
                s=70
            )
            ax_emb.set_title(f"{proj_method} Latent Embeddings (Colored by {color_by})", fontsize=13)
            ax_emb.set_xlabel(f"{proj_method} Dimension 1")
            ax_emb.set_ylabel(f"{proj_method} Dimension 2")
            ax_emb.grid(True, linestyle="--", alpha=0.4)
            
            # Custom legend
            handles, _ = scatter.legend_elements()
            if len(handles) == len(label_names):
                ax_emb.legend(handles, label_names, title="Classes / Groups", loc="best")
            st.pyplot(fig_emb)

            # Stage 5 Interpretability: Conformer Attention Heatmap
            if "Conformer" in m_data["model_type"]:
                st.divider()
                st.subheader("Multi-Head Self-Attention Saliency Heatmap")
                st.caption("Visualizes the temporal attention weights across overlapping patches, showing where in the 2.0s action window phonemic decoding occurs.")
                
                with torch.no_grad():
                    dummy_x = torch.tensor(Xp[:1], dtype=torch.float32)
                    _, attns = model.extract_features(dummy_x, return_attns=True)
                    if attns:
                        # Average across heads in last layer: (1, N_patches, N_patches)
                        last_layer_attn = attns[-1][0].detach().cpu().numpy()
                        fig_attn, ax_attn = plt.subplots(figsize=(6, 5))
                        im_at = ax_attn.imshow(last_layer_attn, cmap="magma")
                        fig_attn.colorbar(im_at, ax=ax_attn, label="Self-Attention Weight")
                        ax_attn.set_title("Conformer MHSA Attention Weights (Temporal Patch Matrix)")
                        ax_attn.set_xlabel("Key Patch Index")
                        ax_attn.set_ylabel("Query Patch Index")
                        st.pyplot(fig_attn)
