import numpy as np
import pandas as pd
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

st.header("Interpretability")
trained = st.session_state.trained_models.get("active")
if not trained:
    st.info("Train and evaluate a model first.")
    st.stop()

projection = st.selectbox("Feature-space projection", ["PCA", "t-SNE", "UMAP"], key="projection")
result = trained["result"]
all_features = np.asarray(result["features"])
all_labels = np.asarray(result["y_true"])
all_subjects = np.asarray(result["subjects"])
selected_subject = None
if trained["result"].get("fold_accuracy"):
    selected_subject = st.selectbox("Held-out subject", sorted(result["fold_accuracy"]))
    selected = all_subjects.astype(str) == selected_subject
else:
    selected = np.ones(len(all_labels), dtype=bool)
features, labels = all_features[selected], all_labels[selected]

if st.button("Project held-out CSP features", type="primary"):
    try:
        if len(features) < 3:
            raise ValueError("At least three held-out trials are needed for a 2D projection")
        if projection == "PCA":
            coordinates = PCA(n_components=2).fit_transform(features)
        elif projection == "t-SNE":
            perplexity = min(30.0, max(1.0, (len(features) - 1) / 3.0))
            coordinates = TSNE(
                n_components=2, perplexity=perplexity,
                init="pca", learning_rate="auto", random_state=42,
            ).fit_transform(features)
        else:
            try:
                import umap
            except ImportError as exc:
                raise ValueError("UMAP needs the optional dependency: pip install 'voice[deep-learning]'") from exc
            coordinates = umap.UMAP(n_components=2, random_state=42).fit_transform(features)

        dataset = st.session_state.raw_data["active"]["trials"]
        name_by_code = {i + 1: name for i, name in enumerate(dataset.label_names)}
        frame = pd.DataFrame({
            "x": coordinates[:, 0], "y": coordinates[:, 1],
            "class": [name_by_code.get(int(label), str(label)) for label in labels],
        })
        st.session_state.latent_embeddings["active"] = {
            "projection": projection, "frame": frame,
            "subject": selected_subject,
        }
    except Exception as exc:
        st.error(f"Projection failed: {exc}")

embedding = st.session_state.latent_embeddings.get("active")
if embedding and embedding.get("subject") == selected_subject:
    st.caption(f"{embedding['projection']} projection of out-of-fold CSP features")
    st.scatter_chart(embedding["frame"], x="x", y="y", color="class")
    if len(np.unique(labels)) >= 2 and len(np.unique(labels)) < len(labels):
        st.metric("CSP-feature silhouette score", f"{silhouette_score(features, labels):.3f}")
    else:
        st.info("Silhouette score requires multiple classes and at least one repeated class sample.")
elif embedding:
    st.info("Run the projection again for the selected held-out subject.")
