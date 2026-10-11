import streamlit as st

st.header("Features")
if not st.session_state.proc_data.get("active"):
    st.info("Run preprocessing first.")
    st.stop()

st.write("The CSP features are fitted inside each training fold to prevent test-data leakage.")
n_components = st.slider("CSP components per frequency band", min_value=2, max_value=6, value=4)
if st.button("Save feature configuration", type="primary"):
    st.session_state.tokenized_data["active"] = {"n_components": n_components}
    st.session_state.trained_models.clear()
    st.session_state.latent_embeddings.clear()
    st.success("Feature configuration saved. CSP fitting will happen during model evaluation.")

if st.session_state.tokenized_data.get("active"):
    st.caption(f"Configured {st.session_state.tokenized_data['active']['n_components']} CSP components per band.")
