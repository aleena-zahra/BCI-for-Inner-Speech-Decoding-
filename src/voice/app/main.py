from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Voice BCI",
    page_icon=":material/neurology:",
    layout="wide",
)

st.session_state.setdefault("raw_data", {})
st.session_state.setdefault("proc_data", {})
st.session_state.setdefault("tokenized_data", {})
st.session_state.setdefault("trained_models", {})
st.session_state.setdefault("latent_embeddings", {})

page_dir = Path(__file__).resolve().parent / "pages"

pages = {
    "Pipeline": [
        st.Page(
            page_dir / "dataset.py",
            title="Dataset",
            icon=":material/folder_open:",
        ),
        st.Page(
            page_dir / "preprocessing.py",
            title="Preprocessing",
            icon=":material/tune:",
        ),
        st.Page(
            page_dir / "features.py",
            title="Features",
            icon=":material/analytics:",
        ),
        st.Page(
            page_dir / "training.py",
            title="Training",
            icon=":material/model_training:",
        ),
        st.Page(
            page_dir / "interpretability.py",
            title="Interpretability",
            icon=":material/insights:",
        ),
    ]
}

page = st.navigation(pages)
page.run()
