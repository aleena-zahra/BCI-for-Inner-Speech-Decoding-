import numpy as np
import pandas as pd
import streamlit as st

from voice.evaluation.interpretability import (
    conformer_attention_summary,
    eegnet_channel_saliency,
)
from voice.evaluation.latent_space import LatentSpaceAuditor

st.header("Latent analysis and model interpretation")
trained = st.session_state.trained_models.get("active")
if not trained:
    st.info("Train and evaluate a model first.")
    st.stop()

result = trained["result"]
kind = trained.get("kind", "CSP + SVM")
selected_fold = None
model = None
test_signals = None
if kind in {"EEGNet", "EEGConformer"}:
    fold_names = list(result["fold_latents"])
    selected_fold = st.selectbox("Model fold", fold_names)
    features = np.asarray(result["fold_latents"][selected_fold])
    labels = np.asarray(result["fold_true_labels"][selected_fold])
    model = result["models"][selected_fold]
    test_signals = result["fold_test_signals"][selected_fold]
else:
    features_all = np.asarray(result["features"])
    labels_all = np.asarray(result["y_true"])
    subjects = np.asarray(result["subjects"])
    if result.get("fold_accuracy"):
        selected_fold = st.selectbox("Held-out subject", sorted(result["fold_accuracy"]))
        selected = subjects.astype(str) == selected_fold
    else:
        selected = np.ones(len(labels_all), dtype=bool)
    features, labels = features_all[selected], labels_all[selected]

st.subheader("Latent representation")
projection = st.selectbox("Projection", ["PCA", "t-SNE", "UMAP"], key="projection")
if st.button("Project held-out embeddings", type="primary"):
    try:
        projected = LatentSpaceAuditor.compute_projections(features, methods=(projection,))
        key = projection.lower().replace("-", "")
        coordinates = projected[key]
        dataset = st.session_state.raw_data["active"]["trials"]
        names = {index + 1: name for index, name in enumerate(dataset.label_names)}
        frame = pd.DataFrame({
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "class": [names.get(int(label), str(label)) for label in labels],
        })
        st.session_state.latent_embeddings["active"] = {
            "projection": projection,
            "frame": frame,
            "fold": selected_fold,
            "kind": kind,
        }
    except Exception as exc:
        st.error(f"Projection failed: {exc}")

embedding = st.session_state.latent_embeddings.get("active")
if embedding and embedding.get("fold") == selected_fold and embedding.get("kind") == kind:
    st.caption(f"{embedding['projection']} projection for {selected_fold or 'the trial holdout'}")
    st.scatter_chart(embedding["frame"], x="x", y="y", color="class")
    try:
        metrics = LatentSpaceAuditor.compute_metrics(features, labels)
        if metrics["silhouette_score"] is not None:
            st.metric("Silhouette score", f"{metrics['silhouette_score']:.3f}")
        if metrics["intra_inter_distance_ratio"] is not None:
            st.metric("Intra/inter-class distance ratio", f"{metrics['intra_inter_distance_ratio']:.3f}")
    except Exception as exc:
        st.info(f"Cluster metrics unavailable: {exc}")
elif embedding:
    st.info("Run the projection again for the selected model fold.")

if kind == "EEGNet":
    st.subheader("EEGNet input-gradient channel saliency")
    st.caption("Absolute input gradients are summarized by channel; this is a sensitivity view, not a causal explanation.")
    if st.button("Compute channel saliency"):
        try:
            saliency = eegnet_channel_saliency(model, test_signals[:32])
            channel_names = st.session_state.raw_data["active"]["trials"].channel_names
            st.bar_chart(pd.DataFrame({"channel": channel_names, "saliency": saliency}).set_index("channel"))
        except Exception as exc:
            st.error(f"Saliency calculation failed: {exc}")
elif kind == "EEGConformer":
    st.subheader("EEGConformer self-attention")
    st.caption("Attention weights summarize token-to-token weighting; they should not be read as causal importance.")
    if st.button("Compute attention maps"):
        try:
            maps = conformer_attention_summary(model, test_signals[:16])
            st.session_state.latent_embeddings["attention"] = {
                "fold": selected_fold, "maps": maps,
            }
        except Exception as exc:
            st.error(f"Attention extraction failed: {exc}")
    attention = st.session_state.latent_embeddings.get("attention")
    if attention and attention.get("fold") == selected_fold:
        layer = st.selectbox("Attention layer", list(range(len(attention["maps"]))))
        import plotly.graph_objects as go

        matrix = attention["maps"][layer]
        figure = go.Figure(data=go.Heatmap(z=matrix, colorscale="Viridis"))
        figure.update_layout(xaxis_title="Key token", yaxis_title="Query token")
        st.plotly_chart(figure, use_container_width=True)
elif kind == "CSP + SVM":
    st.info("The SVM uses the CSP feature-space projection above; neural attribution maps do not apply to this model.")
