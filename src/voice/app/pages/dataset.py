from pathlib import Path
from uuid import uuid4

import streamlit as st

from voice.data.loader import extract_bids_archive, scan_bids_eeg_dataset
from voice.data.trials import load_bids_trials

st.header("Dataset")
st.write("Load a BIDS EEG dataset from a local folder or ZIP archive.")

dataset_path = st.text_input("Local BIDS dataset path", placeholder="D:/data/inner-speech", key="dataset_path")
uploaded_file = st.file_uploader("Or upload a BIDS ZIP archive", type=["zip"], key="dataset_upload")

if st.button("Load dataset and trials", type="primary"):
    try:
        if uploaded_file is not None:
            archive_dir = Path(st.session_state.get("voice_upload_root", Path.home() / ".voice-bci" / "uploads"))
            archive_dir.mkdir(parents=True, exist_ok=True)
            unique_dir = archive_dir / uuid4().hex
            unique_dir.mkdir()
            archive_path = unique_dir / uploaded_file.name
            archive_path.write_bytes(uploaded_file.getvalue())
            root = extract_bids_archive(archive_path, unique_dir / "dataset")
        elif dataset_path.strip():
            root = Path(dataset_path).expanduser()
        else:
            st.warning("Choose a local dataset folder or upload a ZIP archive.")
            st.stop()

        manifest = scan_bids_eeg_dataset(root)
        if not manifest:
            raise ValueError("No supported BIDS EEG recordings were found.")
        with st.spinner("Reading recordings and creating event-locked trials…"):
            trials = load_bids_trials(root)
        st.session_state.raw_data.clear()
        st.session_state.raw_data["active"] = {
            "path": str(root.resolve()), "manifest": manifest, "trials": trials,
        }
        for key in ("proc_data", "tokenized_data", "trained_models", "latent_embeddings"):
            st.session_state[key].clear()
        st.success(f"Loaded {len(trials.signals)} trials from {len(manifest)} recordings.")
    except Exception as exc:
        st.error(f"Could not load this dataset: {exc}")

active = st.session_state.raw_data.get("active")
if active:
    trials = active["trials"]
    col1, col2, col3 = st.columns(3)
    col1.metric("Trials", len(trials.signals))
    col2.metric("Subjects", len(set(trials.subjects)))
    col3.metric("EEG channels", len(trials.channel_names))
    st.caption(f"Sampling rate: {trials.sfreq:g} Hz · Classes: {', '.join(trials.label_names)}")
    st.dataframe(
        [{"Recording": item["run_id"], "Subject": item["subject"], "Task": item["task"]}
         for item in active["manifest"].values()],
        use_container_width=True,
        hide_index=True,
    )
