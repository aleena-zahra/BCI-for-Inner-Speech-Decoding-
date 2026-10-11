import streamlit as st

st.header("Interpretability")

if not st.session_state.trained_models:
    st.info("Train a model first.")
    st.stop()

projection = st.selectbox("Projection", ["PCA", "t-SNE", "UMAP"], key="projection")

if st.button("Audit latent space", type="primary"):
    st.session_state.latent_embeddings["active"] = {"projection": projection}
    st.success("Interpretability settings recorded. Connect latent-space auditing here next.")