import streamlit as st

st.header("Dataset")
st.write("Load an EEG dataset to begin the pipeline.")

dataset_path = st.text_input(
    "Local dataset path",
    placeholder="D:/data/inner-speech",
    key="dataset_path",
)
uploaded_file = st.file_uploader(
    "Or upload a dataset archive",
    type=["zip"],
    key="dataset_upload",
)

if st.button("Register dataset", type="primary"):
    if not dataset_path and uploaded_file is None:
        st.warning("Choose a local path or upload a ZIP file first.")
    else:
        st.session_state.raw_data["active"] = {
            "path": dataset_path,
            "uploaded_file": uploaded_file.name if uploaded_file else None,
        }
        st.success("Dataset registered for this session.")