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

pages = {
    "Pipeline": [
        st.Page(
            "pages/dataset.py",
            title="Dataset",
            icon=":material/folder_open:",
        ),
        st.Page(
            "pages/preprocessing.py",
            title="Preprocessing",
            icon=":material/tune:",
        ),
        st.Page(
            "pages/features.py",
            title="Features",
            icon=":material/analytics:",
        ),
        st.Page(
            "pages/training.py",
            title="Training",
            icon=":material/model_training:",
        ),
        st.Page(
            "pages/interpretability.py",
            title="Interpretability",
            icon=":material/insights:",
        ),
    ]
}

page = st.navigation(pages)
page.run()