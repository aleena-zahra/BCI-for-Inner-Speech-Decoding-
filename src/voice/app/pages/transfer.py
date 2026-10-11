from pathlib import Path
from uuid import uuid4

import numpy as np
import streamlit as st

from voice.data.loader import extract_bids_archive
from voice.data.trials import load_bids_trials
from voice.evaluation.transferlearning import evaluate_concept_mapped_transfer
from voice.preprocessing.pipeline import EEGPreprocessor

st.header("Cross-language transfer experiment")
st.warning(
    "This is a separate target-dataset experiment. It only reports concept-mapped "
    "scores after you explicitly map every target label to a source concept."
)
trained = st.session_state.trained_models.get("active")
source = st.session_state.raw_data.get("active")
processed = st.session_state.proc_data.get("active")
if not trained or not source or not processed:
    st.info("Load and preprocess a source dataset, then train a model first.")
    st.stop()

target_path = st.text_input("Target BIDS dataset folder", key="transfer_target_path")
uploaded = st.file_uploader("Or upload target BIDS ZIP", type=["zip"], key="transfer_target_zip")
if st.button("Load target dataset"):
    try:
        if uploaded:
            root_dir = Path(st.session_state.get("voice_upload_root", Path.home() / ".voice-bci" / "uploads"))
            root_dir.mkdir(parents=True, exist_ok=True)
            target_root = root_dir / f"transfer-{uuid4().hex}"
            target_root.mkdir()
            zip_path = target_root / uploaded.name
            zip_path.write_bytes(uploaded.getvalue())
            target_root = extract_bids_archive(zip_path, target_root / "dataset")
        elif target_path.strip():
            target_root = Path(target_path).expanduser()
        else:
            raise ValueError("Select a local BIDS folder or upload a BIDS ZIP archive")
        target_trials = load_bids_trials(target_root)
        st.session_state["transfer_data"] = {"path": str(target_root.resolve()), "trials": target_trials}
        st.session_state.pop("transfer_result", None)
        st.success(f"Loaded {len(target_trials.signals)} target trials.")
    except Exception as exc:
        st.error(f"Could not load target dataset: {exc}")

target = st.session_state.get("transfer_data")
if not target:
    st.stop()
target_trials = target["trials"]
source_trials = source["trials"]
language_left, language_right = st.columns(2)
source_language = language_left.text_input("Source language", key="source_language", placeholder="e.g., English")
target_language = language_right.text_input("Target language", key="target_language", placeholder="e.g., Spanish")
st.caption(
    f"Source classes: {', '.join(source_trials.label_names)} · "
    f"Target classes: {', '.join(target_trials.label_names)}"
)
if tuple(source_trials.channel_names) != tuple(target_trials.channel_names):
    st.error("Source and target channel names/order differ. Channel harmonization is required before transfer.")
    st.stop()

st.subheader("Explicit concept mapping")
st.caption("Enter shared semantic concept names for both datasets. Labels are never matched by numeric ID or spelling automatically.")
source_label_to_concept = {}
for label in source_trials.label_names:
    concept = st.text_input(f"Source class '{label}' means", key=f"source-concept-{label}").strip()
    if concept:
        source_label_to_concept[label] = concept
target_label_to_concept = {}
for target_label in target_trials.label_names:
    concept = st.text_input(f"Target class '{target_label}' means", key=f"target-concept-{target_label}").strip()
    if concept:
        target_label_to_concept[target_label] = concept

model_kind = trained["kind"]
if model_kind != "CSP + SVM":
    result = trained["result"]
    fold_name = st.selectbox("Trained source fold", list(result["models"].keys()))
else:
    fold_name = None

if st.button("Run mapped transfer evaluation", type="primary"):
    if set(target_label_to_concept) != set(target_trials.label_names) or set(source_label_to_concept) != set(source_trials.label_names):
        st.error("Enter an explicit concept name for every source and target class.")
    elif len(set(target_label_to_concept.values())) != len(target_label_to_concept) or len(set(source_label_to_concept.values())) != len(source_label_to_concept):
        st.error("Each class within a dataset must map to a distinct concept.")
    else:
        try:
            settings = dict(processed["settings"])
            settings["sfreq"] = target_trials.sfreq
            if settings["highcut"] >= settings["target_sfreq"] / 2:
                raise ValueError("The source preprocessing cutoff exceeds the target Nyquist frequency")
            if model_kind == "CSP + SVM":
                fitted = trained.get("source_pipeline")
                if fitted is None:
                    raise ValueError("Retrain the source model to create a fitted transfer model")
                processed_target = fitted.preprocessor.transform(target_trials.signals)
                features = fitted.feature_extractor.transform(processed_target)
                predictions = fitted.classifier.model.predict(features)
                source_codes = list(fitted.classifier.model.classes_)
                source_predictions = [source_trials.label_names[int(code) - 1] for code in predictions]
                source_map = {code: source_label_to_concept[source_trials.label_names[int(code) - 1]] for code in source_codes}
            else:
                model_result = trained["result"]
                model = model_result["models"][fold_name]
                target_preprocessor = EEGPreprocessor(**settings)
                target_signals = target_preprocessor.transform(target_trials.signals)
                if target_signals.shape[1:] != tuple(processed["signals"].shape[1:]):
                    raise ValueError("Preprocessed target shape differs from source model input shape")
                import torch

                model.eval()
                with torch.no_grad():
                    logits = model(torch.as_tensor(np.asarray(target_signals, dtype=np.float32)))
                    predictions = logits.argmax(dim=1).cpu().numpy()
                source_classes = model_result["classes"]
                source_predictions = [source_label_to_concept[source_trials.label_names[int(source_classes[int(i)]) - 1]] for i in predictions]
                source_map = {
                    index: source_label_to_concept[source_trials.label_names[int(label) - 1]]
                    for index, label in enumerate(source_classes)
                }
            target_labels = [target_trials.label_names[int(code) - 1] for code in target_trials.labels]
            transfer_metrics = evaluate_concept_mapped_transfer(
                target_labels,
                source_predictions,
                target_to_concept=target_label_to_concept,
                source_class_to_concept=source_map,
            )
            transfer_metrics["source_language"] = source_language or "Unspecified"
            transfer_metrics["target_language"] = target_language or "Unspecified"
            transfer_metrics["source_model"] = model_kind
            transfer_metrics["target_path"] = target["path"]
            st.session_state["transfer_result"] = transfer_metrics
        except Exception as exc:
            st.error(f"Mapped transfer evaluation failed: {exc}")

transfer_result = st.session_state.get("transfer_result")
if transfer_result:
    st.metric("Concept-mapped accuracy", f"{transfer_result['concept_mapped_accuracy']:.1%}")
    st.caption(f"{transfer_result['n_trials']} target trials · model: {transfer_result['source_model']}")
    st.write("Target label concepts", transfer_result["target_to_concept"])
    st.write("Source output concepts", transfer_result["source_class_to_concept"])
    st.dataframe(transfer_result["confusion_matrix"], use_container_width=True)
    st.write("Per-concept accuracy", transfer_result["per_concept_accuracy"])
