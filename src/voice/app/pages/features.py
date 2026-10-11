import streamlit as st

from voice.preprocessing.features import EEGPatchTokenizer, SpectralFeatureExtractor

st.header("Features and temporal patches")
processed = st.session_state.proc_data.get("active")
if not processed:
    st.info("Run preprocessing first.")
    st.stop()

signals = processed["signals"]
sfreq = processed["settings"]["target_sfreq"]
st.write("CSP is fitted inside each model evaluation fold. The deep models consume the preprocessed EEG tensors directly.")
n_components = st.slider("CSP components per frequency band", min_value=2, max_value=8, value=4)
patch_length = st.slider("Conformer patch length (samples)", min_value=8, max_value=min(100, signals.shape[-1]), value=min(25, signals.shape[-1]))
stride = st.slider("Conformer patch stride (samples)", min_value=1, max_value=patch_length, value=min(6, patch_length))
use_revin = st.checkbox("Use learnable RevIN in EEGNet / EEGConformer", value=False)

tokenizer = EEGPatchTokenizer(patch_len=patch_length, stride=stride)
patch_metadata = tokenizer.get_token_metadata(signals.shape[-1], sfreq=sfreq)
st.caption(
    f"Conformer input: {patch_metadata['n_patches']} overlapping tokens · "
    f"{patch_metadata['patch_duration_ms']:.1f} ms per patch · "
    f"{patch_metadata['stride_duration_ms']:.1f} ms stride."
)
sample_patches, _ = tokenizer.tokenize(signals[:1])
st.caption(f"Preview tensor shape: {sample_patches.shape} (one trial; patches are formed inside Conformer during training).")

if st.button("Save feature configuration", type="primary"):
    st.session_state.tokenized_data["active"] = {
        "n_components": n_components,
        "patch_length": patch_length,
        "stride": stride,
        "use_revin": use_revin,
        "patch_metadata": patch_metadata,
    }
    st.session_state.trained_models.clear()
    st.session_state.latent_embeddings.clear()
    st.success("Feature configuration saved.")

if st.button("Compute descriptive spectral band powers"):
    extractor = SpectralFeatureExtractor(sfreq=sfreq)
    powers, _, _ = extractor.compute_band_powers(signals)
    summary = {
        "Band": list(powers),
        "Mean power across trials and channels": [float(values.mean()) for values in powers.values()],
    }
    st.session_state.tokenized_data["spectral_summary"] = summary
    st.dataframe(summary, hide_index=True, use_container_width=True)
    st.caption("These summaries are exploratory and are not used as neural-model inputs.")

saved = st.session_state.tokenized_data.get("active")
if saved:
    st.caption(
        f"Saved: {saved['n_components']} CSP components per band, "
        f"patch {saved['patch_length']}/{saved['stride']}, RevIN={'on' if saved['use_revin'] else 'off'}."
    )
