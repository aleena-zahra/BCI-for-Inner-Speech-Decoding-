import streamlit as st

st.header("Features")

if not st.session_state.proc_data:
    st.info("Run preprocessing first.")
    st.stop()

patch_length = st.slider("Patch length", min_value=10, max_value=50, value=25, key="patch_length")
stride = st.slider("Patch stride", min_value=2, max_value=20, value=6, key="patch_stride")

if st.button("Extract features", type="primary"):
    st.session_state.tokenized_data["active"] = {
        "patch_length": patch_length,
        "stride": stride,
    }
    st.success("Feature settings recorded. Connect the tokenizer here next.")