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
    subjects = result.get("fold_test_subjects", {}).get(selected_fold)
    if subjects is None:
        subjects = result["subjects"] if selected_fold == "trial holdout" else np.repeat(selected_fold, len(labels))
    subjects = np.asarray(subjects)
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
    subjects = subjects[selected]

st.subheader("Latent representation")
projection = st.selectbox("Projection", ["PCA", "t-SNE", "UMAP"], key="projection")
color_by = st.selectbox("Color embeddings by", ["Word class", "Subject identity"])
with st.expander("Projection settings"):
    perplexity = st.number_input("t-SNE perplexity", min_value=1, value=15)
    n_neighbors = st.number_input("UMAP neighbors", min_value=2, value=15)
st.caption("Audits use held-out trials from one fitted fold. A subject silhouette is unavailable when that fold contains only one subject; these metrics do not prove linguistic decoding.")
if st.button("Project held-out embeddings", type="primary"):
    try:
        audit = LatentSpaceAuditor.audit(features, labels, subjects, method=projection,
                                         perplexity=perplexity, n_neighbors=n_neighbors)
        projected = audit["projections"]
        key = projection.lower().replace("-", "")
        coordinates = projected[key]
        dataset = st.session_state.raw_data["active"]["trials"]
        names = {index + 1: name for index, name in enumerate(dataset.label_names)}
        frame = pd.DataFrame({
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "class": [names.get(int(label), str(label)) for label in labels],
            "subject": subjects.astype(str),
        })
        st.session_state.latent_embeddings["active"] = {
            "projection": projection,
            "frame": frame,
            "fold": selected_fold,
            "kind": kind,
            "class_metrics": audit["class_metrics"],
            "subject_metrics": audit["subject_metrics"],
        }
    except Exception as exc:
        st.error(f"Projection failed: {exc}")

embedding = st.session_state.latent_embeddings.get("active")
if embedding and embedding.get("fold") == selected_fold and embedding.get("kind") == kind:
    st.caption(f"{embedding['projection']} projection for {selected_fold or 'the trial holdout'}")
    color_column = "class" if color_by == "Word class" else "subject"
    if color_column in embedding["frame"]:
        st.scatter_chart(embedding["frame"], x="x", y="y", color=color_column)
    else:
        st.info("Run the projection again to add subject colors.")
    try:
        metrics = embedding.get("class_metrics") or LatentSpaceAuditor.compute_metrics(features, labels)
        subject_metrics = embedding.get("subject_metrics") or LatentSpaceAuditor.compute_metrics(features, subjects)
        metrics_cols = st.columns(3)
        values = [metrics["silhouette_score"], subject_metrics["silhouette_score"], metrics["intra_inter_distance_ratio"]]
        titles = ["Word-class silhouette", "Subject silhouette", "Intra/inter-class distance ratio"]
        for column, title, value in zip(metrics_cols, titles, values):
            column.metric(title, "N/A" if value is None else f"{value:.3f}")
    except Exception as exc:
        st.info(f"Cluster metrics unavailable: {exc}")
elif embedding:
    st.info("Run the projection again for the selected model fold.")

if kind == "EEGNet":
    st.subheader("EEGNet input-gradient channel saliency")
    st.caption("Absolute input gradients are summarized by channel; this is a sensitivity view, not a causal explanation.")
    dataset = st.session_state.raw_data["active"]["trials"]
    class_options = ["Predicted class", *[dataset.label_names[int(code) - 1] for code in result["classes"]]]
    target = st.selectbox("Saliency target", class_options)
    target_class = None if target == "Predicted class" else class_options.index(target) - 1
    if st.button("Compute channel saliency"):
        try:
            saliency = eegnet_channel_saliency(model, test_signals[:32], target_class=target_class)
            channel_names = st.session_state.raw_data["active"]["trials"].channel_names
            st.bar_chart(pd.DataFrame({"channel": channel_names, "saliency": saliency}).set_index("channel"))
        except Exception as exc:
            st.error(f"Saliency calculation failed: {exc}")
elif kind == "EEGConformer":
    st.subheader("EEGConformer self-attention")
    st.caption("Attention weights summarize token-to-token weighting; they should not be read as causal importance.")
    head_options = ["Average all heads", *range(model.blocks[0].attn.num_heads)]
    selected_head = st.selectbox("Attention head", head_options)
    head = None if selected_head == "Average all heads" else int(selected_head)
    sample_count = int(st.number_input("Held-out trials in attention summary", min_value=1, max_value=len(test_signals), value=min(16, len(test_signals))))
    if st.button("Compute attention maps"):
        try:
            maps = conformer_attention_summary(model, test_signals[:sample_count], head=head)
            st.session_state.latent_embeddings["attention"] = {
                "fold": selected_fold, "maps": maps, "head": head, "sample_count": sample_count,
            }
        except Exception as exc:
            st.error(f"Attention extraction failed: {exc}")
    attention = st.session_state.latent_embeddings.get("attention")
    if attention and attention.get("fold") == selected_fold and attention.get("head") == head and attention.get("sample_count") == sample_count:
        layer = st.selectbox("Attention layer", list(range(len(attention["maps"]))))
        import plotly.graph_objects as go

        matrix = attention["maps"][layer]
        settings = st.session_state.proc_data["active"]["settings"]
        centers = (np.arange(len(matrix)) * model.stride + (model.patch_len - 1) / 2) / settings["target_sfreq"] + settings["t_start"]
        figure = go.Figure(data=go.Heatmap(z=matrix, x=centers, y=centers, colorscale="Viridis"))
        figure.update_layout(xaxis_title="Key patch center after event (s)", yaxis_title="Query patch center after event (s)")
        st.plotly_chart(figure, use_container_width=True)
        st.line_chart(pd.DataFrame({"Mean received attention": matrix.mean(axis=0)}, index=centers))
elif kind == "CSP + SVM":
    st.info("The SVM uses the CSP feature-space projection above; neural attribution maps do not apply to this model.")
