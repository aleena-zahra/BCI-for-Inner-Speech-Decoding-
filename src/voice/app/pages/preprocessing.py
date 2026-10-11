import numpy as np
import pandas as pd
import streamlit as st
from scipy.signal import welch

from voice.preprocessing.pipeline import EEGPreprocessor

st.header("Preprocessing")
active = st.session_state.raw_data.get("active")
if not active:
    st.info("Load a dataset on the Dataset page first.")
    st.stop()
trials = active["trials"]
profile = st.selectbox("Preprocessing workflow", ["Voice IIR", "POC literature FIR"])
literature = profile == "POC literature FIR"
left, right = st.columns(2)
with left:
    lowcut = st.number_input("Low cutoff (Hz)", 0.1, 30.0, 0.5 if literature else 1.0, step=0.5)
    highcut = st.number_input("High cutoff (Hz)", 10.0, 120.0, 100.0 if literature else 40.0, step=5.0)
    target_sfreq = st.selectbox("Target sampling rate (Hz)", [128.0, 250.0, 256.0], index=1 if literature else 0)
    notch = st.number_input("Notch frequency (Hz; 0 disables)", 0.0, 100.0, 50.0)
    t_start = st.number_input("Interval start after event (s)", 0.0, 3.0, 1.5, step=0.1)
    t_end = st.number_input("Interval end after event (s)", 0.1, 3.5, 3.5, step=0.1)
with right:
    car = st.checkbox("Apply common average reference", value=True)
    baseline = st.checkbox("Subtract each trial's mean", value=False)
    zscore = st.checkbox("Per-trial, per-channel z-score normalization", value=literature)
    threshold = st.number_input("Optional peak-amplitude rejection (µV; 0 disables)", 0.0, value=0.0, step=10.0)
    enable_ica = st.checkbox("Fit ICA for component review", value=False)
    components = None
    exclusions = ()
    if enable_ica:
        max_components = len(trials.channel_names) - int(car)
        if max_components < 2:
            st.error("ICA needs at least two independent EEG channels.")
            st.stop()
        components = int(st.number_input("ICA components", 2, max_components, min(15, max_components)))
        excluded = st.text_input("Preview ICA exclusions (comma-separated, zero-based)", value="")
        st.caption("Only reviewed components are removed. The Training page provides a separate review for each training fold.")
st.caption(
    "FIR mode filters full loaded trials before resampling/cropping. The continuous Raw/events workflow is also available "
    "in the library. Choose the event-relative interval for your dataset; this does not prove visual-cue artifacts are absent."
)

if st.button("Run preprocessing", type="primary"):
    try:
        if enable_ica:
            exclusions = tuple(int(value.strip()) for value in excluded.split(",") if value.strip())
        settings = dict(
            sfreq=trials.sfreq, target_sfreq=target_sfreq, lowcut=lowcut, highcut=highcut,
            notch_freq=notch, t_start=t_start, t_end=t_end,
            common_average_reference=car, apply_baseline=baseline,
            artifact_threshold=threshold or None, filter_method="fir" if literature else "iir",
            zscore=zscore, ica_components=components, ica_exclude=exclusions,
        )
        processor = EEGPreprocessor(**settings)
        with st.spinner("Filtering, resampling, and preparing aligned trials…"):
            signals, labels, kept = processor.fit_transform_with_indices(trials.signals, trials.labels)
        st.session_state.proc_data["active"] = {
            "signals": signals, "labels": labels, "subjects": trials.subjects[kept],
            "runs": trials.runs[kept], "kept_indices": kept, "settings": settings,
            "processor": processor, "ica_reviews": {},
        }
        for key in ("tokenized_data", "trained_models", "latent_embeddings"):
            st.session_state[key].clear()
        st.success(f"Preprocessed {len(signals)} aligned trials ({len(trials.signals) - len(signals)} rejected).")
    except Exception as exc:
        st.error(f"Preprocessing failed: {exc}")

processed = st.session_state.proc_data.get("active")
if processed:
    st.metric("Retained trials", len(processed["signals"]))
    st.caption(f"Shape: {processed['signals'].shape} · amplitude rejection is applied before z-score normalization.")
    st.subheader("Power spectral density")
    before, after = st.columns(2)
    for column, title, values, sampling in (
        (before, "Before preprocessing (µV²/Hz)", trials.signals[processed["kept_indices"][:8]], trials.sfreq),
        (after, "After preprocessing (normalized units²/Hz)" if processed["settings"].get("zscore") else "After preprocessing (µV²/Hz)",
         processed["signals"][:8], processed["settings"]["target_sfreq"]),
    ):
        freq, power = welch(values, fs=sampling, nperseg=min(256, values.shape[-1]), axis=-1)
        column.write(title)
        column.line_chart(pd.DataFrame({"Mean PSD": power.mean(axis=(0, 1))}, index=freq))
    processor = processed.get("processor")
    if processor and processor.ica_ is not None:
        st.subheader("Preview ICA component review")
        st.caption(f"Excluded components: {processor.ica_.exclude or 'none (decomposition only)'}. Component indices are specific to this fitted decomposition.")
        component = int(st.number_input("Inspect preview component", 0, processor.ica_.n_components_ - 1, 0))
        sources = processor.ica_sources(trials.signals[:1])
        st.line_chart(pd.DataFrame({"ICA activation": sources[0, component]}, index=np.arange(sources.shape[-1]) / processor.target_sfreq))
        st.bar_chart(pd.DataFrame({"Spatial loading": processor.ica_.get_components()[:, component]}, index=trials.channel_names))
