import numpy as np
import streamlit as st

from voice.preprocessing.pipeline import EEGPreprocessor

st.header("Preprocessing")
active = st.session_state.raw_data.get("active")
if not active:
    st.info("Load a dataset on the Dataset page first.")
    st.stop()

trials = active["trials"]
left, right = st.columns(2)
with left:
    lowcut = st.number_input("Low cutoff (Hz)", min_value=0.1, max_value=30.0, value=1.0, step=0.5)
    highcut = st.number_input("High cutoff (Hz)", min_value=10.0, max_value=60.0, value=40.0, step=1.0)
    target_sfreq = st.selectbox("Target sampling rate (Hz)", [128.0, 256.0], index=0)
with right:
    common_average_reference = st.checkbox("Apply common average reference", value=True)
    apply_baseline = st.checkbox("Subtract each trial's mean", value=False)
    artifact_threshold = st.number_input(
        "Optional peak-amplitude rejection (µV; 0 disables)", min_value=0.0, value=0.0, step=10.0
    )

if st.button("Run preprocessing", type="primary"):
    if highcut >= target_sfreq / 2:
        st.error("High cutoff must be below the target sampling rate's Nyquist frequency.")
    else:
        try:
            settings = dict(
                sfreq=trials.sfreq, target_sfreq=target_sfreq,
                lowcut=lowcut, highcut=highcut, t_start=1.5, t_end=3.5,
                common_average_reference=common_average_reference,
                apply_baseline=apply_baseline,
                artifact_threshold=artifact_threshold or None,
            )
            processor = EEGPreprocessor(**settings)
            signals, labels, kept = processor.transform_with_indices(trials.signals, trials.labels)
            if len(signals) == 0:
                raise ValueError("Preprocessing rejected every trial; lower the artifact threshold.")
            st.session_state.proc_data["active"] = {
                "signals": signals,
                "labels": labels,
                "subjects": trials.subjects[kept],
                "runs": trials.runs[kept],
                "kept_indices": kept,
                "settings": settings,
            }
            st.session_state.tokenized_data.clear()
            st.session_state.trained_models.clear()
            st.session_state.latent_embeddings.clear()
            st.success(f"Preprocessed {len(signals)} aligned trials ({len(trials.signals) - len(signals)} rejected).")
        except Exception as exc:
            st.error(f"Preprocessing failed: {exc}")

processed = st.session_state.proc_data.get("active")
if processed:
    st.metric("Retained trials", len(processed["signals"]))
    st.caption(f"Shape: {processed['signals'].shape[0]} trials × {processed['signals'].shape[1]} channels × {processed['signals'].shape[2]} samples")
