import streamlit as st
import pandas as pd

from voice.evaluation.features import MultiBandCSPExtractor
from voice.pipeline import InnerSpeechPipeline
from voice.preprocessing.pipeline import EEGPreprocessor
from voice.evaluation.splits import evaluation_folds

st.header("Training and evaluation")
processed = st.session_state.proc_data.get("active")
raw_dataset = st.session_state.raw_data.get("active")
feature_settings = st.session_state.tokenized_data.get("active")
if not processed or not feature_settings or not raw_dataset:
    st.info("Load a dataset, run preprocessing, and save the feature configuration first.")
    st.stop()

model_name = st.selectbox("Model", ["CSP + SVM", "EEGNet", "EEGConformer"], key="model_name")
evaluation = st.radio(
    "Evaluation strategy",
    ["Leave one subject out", "Stratified trial holdout"],
    help="Subject holdout tests generalization to unseen people; trial holdout compares trials from the same participants.",
)
if model_name != "CSP + SVM":
    epochs = st.number_input("Training epochs", min_value=1, max_value=100, value=10, key="deep_epochs")
    batch_size = st.select_slider("Batch size", options=[8, 16, 32, 64], value=16)
    st.caption("PyTorch is optional. Install it with `pip install -e '.[deep-learning]'` to enable these models.")

source = raw_dataset["trials"]
settings = dict(processed["settings"])
strategy = "loso" if evaluation == "Leave one subject out" else "trial"
deep = model_name != "CSP + SVM"
ica_exclusions = {}
if settings.get("ica_components") is not None:
    # Preview component indices cannot be transferred to independently fitted folds.
    settings["ica_exclude"] = ()
    review_group = f"{'deep' if deep else 'svm'}:{strategy}"
    reviews = processed.setdefault("ica_reviews", {}).setdefault(review_group, {})
    with st.expander("Review ICA on training folds before excluding components"):
        st.caption("Each decomposition uses only that fold's training trials. Preview exclusions from the Preprocessing page are not reused. Folds without saved exclusions perform decomposition only.")
        try:
            folds = evaluation_folds(source.labels, source.subjects, strategy=strategy, deep=deep)
            fold = st.selectbox("ICA review fold", [item[0] for item in folds], key=f"ica-fold-{review_group}")
            indices = next(item[1] for item in folds if item[0] == fold)
            if st.button("Fit ICA for this training fold", key=f"fit-ica-{review_group}"):
                reviewer = EEGPreprocessor(**settings)
                with st.spinner("Fitting a training-only ICA decomposition…"):
                    reviewer.fit(source.signals[indices])
                reviews[fold] = {"processor": reviewer, "exclude": ()}
                st.session_state.pop(f"exclude-ica-{review_group}-{fold}", None)
            review = reviews.get(fold)
            if review:
                reviewer = review["processor"]
                component = int(st.number_input("Inspect training-fold component", 0, reviewer.ica_.n_components_ - 1, 0, key=f"inspect-ica-{review_group}-{fold}"))
                sources = reviewer.ica_sources(source.signals[indices[:1]])
                st.line_chart(pd.DataFrame({"ICA activation": sources[0, component]}))
                st.bar_chart(pd.DataFrame({"Spatial loading": reviewer.ica_.get_components()[:, component]}, index=source.channel_names))
                excluded = st.text_input("Reviewed exclusions for this fold", value=",".join(map(str, review["exclude"])), key=f"exclude-ica-{review_group}-{fold}")
                if st.button("Save reviewed ICA exclusions", key=f"save-ica-{review_group}-{fold}"):
                    selected = tuple(int(value.strip()) for value in excluded.split(",") if value.strip())
                    if any(index < 0 or index >= reviewer.ica_.n_components_ for index in selected) or len(set(selected)) >= reviewer.ica_.n_components_:
                        raise ValueError("Exclusions must be valid zero-based indices and retain at least one component")
                    review["exclude"] = selected
                    st.success(f"Saved exclusions for {fold}: {selected or 'none'}")
                st.caption(f"Saved fold exclusions: {review['exclude'] or 'none'}")
        except Exception as exc:
            st.error(f"ICA review failed: {exc}")
    ica_exclusions = {name: review["exclude"] for name, review in reviews.items()}

if st.button("Train and evaluate", type="primary"):
    try:
        if model_name == "CSP + SVM":
            preprocessor = EEGPreprocessor(**settings)
            extractor = MultiBandCSPExtractor(
                n_components=feature_settings["n_components"],
                sfreq=settings["target_sfreq"],
            )
            pipeline = InnerSpeechPipeline(preprocessor=preprocessor, feature_extractor=extractor)
            source = raw_dataset["trials"]
            with st.spinner("Fitting CSP and SVM on training folds…"):
                if evaluation == "Leave one subject out":
                    result = pipeline.evaluate_leave_one_subject_out(
                        source.signals, source.labels, source.subjects, ica_exclusions=ica_exclusions
                    )
                else:
                    result = pipeline.evaluate_trial_holdout(
                        source.signals, source.labels, source.subjects, ica_exclusions=ica_exclusions
                    )
            st.session_state.trained_models["active"] = {
                "kind": "CSP + SVM", "pipeline": pipeline, "result": result,
            }
        else:
            from voice.models.training import evaluate_deep_model

            progress = st.progress(0, text="Preparing training folds…")
            status = st.empty()
            fold_names = sorted(set(source.subjects))
            subject_count = len(fold_names) if evaluation == "Leave one subject out" else 1

            def report(fold, epoch, total_epochs, metrics):
                fold_index = fold_names.index(fold) if fold in fold_names else 0
                progress.progress(min(1.0, (fold_index * total_epochs + epoch) / (subject_count * total_epochs)))
                status.caption(
                    f"{fold} · epoch {epoch}/{total_epochs} · "
                    f"validation accuracy {metrics['validation_accuracy']:.1%}"
                )

            result = evaluate_deep_model(
                source.signals, source.labels, source.subjects,
                model_name=model_name,
                strategy="loso" if evaluation == "Leave one subject out" else "trial",
                epochs=int(epochs),
                batch_size=int(batch_size),
                patch_len=feature_settings.get("patch_length", 25),
                stride=feature_settings.get("stride", 6),
                use_revin=feature_settings.get("use_revin", False),
                progress_callback=report,
                preprocessor=EEGPreprocessor(**settings),
                ica_exclusions=ica_exclusions,
            )
            result["evaluation"] = evaluation
            st.session_state.trained_models["active"] = {"kind": model_name, "result": result}

        st.session_state.latent_embeddings.clear()
        st.success("Evaluation completed. Model weights and scores are available for interpretation.")
    except Exception as exc:
        st.error(f"Training/evaluation failed: {exc}")

trained = st.session_state.trained_models.get("active")
if trained:
    result = trained["result"]
    st.metric("Accuracy", f"{result['accuracy']:.1%}")
    if "f1" in result:
        st.metric("Weighted F1", f"{result['f1']:.3f}")
    if result.get("per_subject_accuracy"):
        st.subheader("Accuracy by held-out subject")
        st.dataframe(
            [{"Subject": subject, "Accuracy": accuracy} for subject, accuracy in result["per_subject_accuracy"].items()],
            hide_index=True,
            use_container_width=True,
        )
    st.subheader("Confusion matrix")
    st.dataframe(result["confusion_matrix"], use_container_width=True)
    history = result.get("history")
    if history:
        st.subheader("Training history")
        st.line_chart({
            "Train loss": [row["train_loss"] for row in history],
            "Validation loss": [row["validation_loss"] for row in history],
            "Train accuracy": [row["train_accuracy"] for row in history],
            "Validation accuracy": [row["validation_accuracy"] for row in history],
        })
