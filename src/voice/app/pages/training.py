import streamlit as st

from voice.evaluation.features import MultiBandCSPExtractor
from voice.pipeline import InnerSpeechPipeline
from voice.preprocessing.pipeline import EEGPreprocessor

st.header("Training and evaluation")
processed = st.session_state.proc_data.get("active")
raw_dataset = st.session_state.raw_data.get("active")
feature_settings = st.session_state.tokenized_data.get("active")
if not processed or not feature_settings:
    st.info("Run preprocessing and save the feature configuration first.")
    st.stop()

st.selectbox("Model", ["CSP + SVM"], key="model_name")
evaluation = st.radio(
    "Evaluation strategy",
    ["Leave one subject out", "Stratified trial holdout"],
    help="Subject holdout tests generalization to unseen people; trial holdout compares trials from the same participants.",
)

if st.button("Train and evaluate", type="primary"):
    settings = dict(processed["settings"])
    preprocessor = EEGPreprocessor(**settings)
    extractor = MultiBandCSPExtractor(
        n_components=feature_settings["n_components"],
        sfreq=settings["target_sfreq"],
    )
    pipeline = InnerSpeechPipeline(preprocessor=preprocessor, feature_extractor=extractor)
    try:
        with st.spinner("Fitting CSP and SVM on training folds…"):
            source = raw_dataset["trials"]
            if evaluation == "Leave one subject out":
                result = pipeline.evaluate_leave_one_subject_out(
                    source.signals, source.labels, source.subjects
                )
            else:
                result = pipeline.evaluate_trial_holdout(
                    source.signals, source.labels, source.subjects
                )
        result["evaluation"] = evaluation
        st.session_state.trained_models["active"] = {"pipeline": pipeline, "result": result}
        st.session_state.latent_embeddings.clear()
        st.success("Fold evaluation completed. Each fold fitted preprocessing and CSP on training data only.")
    except Exception as exc:
        st.error(f"Training/evaluation failed: {exc}")

trained = st.session_state.trained_models.get("active")
if trained:
    result = trained["result"]
    st.metric("Accuracy", f"{result['accuracy']:.1%}")
    st.subheader("Accuracy by held-out subject")
    st.dataframe(
        [{"Subject": subject, "Accuracy": accuracy} for subject, accuracy in result["per_subject_accuracy"].items()],
        hide_index=True,
        use_container_width=True,
    )
    st.subheader("Confusion matrix")
    st.dataframe(result["confusion_matrix"], use_container_width=True)
