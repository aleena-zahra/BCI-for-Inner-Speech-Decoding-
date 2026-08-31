"""
EEG Imagined Speech Decoding -- Proof of Concept
Streamlit application following the 9-stage methodology pipeline:

1. Dataset collection        6. Cross-dataset transfer learning
2. EEG preprocessing         7. Embedding space analysis
3/4. Decoding models         8. Evaluation metrics
5. Embedding extraction      9. Visualizations & interpretability

Run with:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import torch
import streamlit as st
import matplotlib.pyplot as plt
import mne

from modules import synthetic_data, preprocessing, models, training, embedding_analysis, interpretability

st.set_page_config(page_title="EEG Imagined Speech Decoding -- POC", layout="wide")

# ----------------------------------------------------------------------
# Session state initialization
# ----------------------------------------------------------------------
for key, default in [
    ("raw_data", {}),       # language -> (X, y, subj, class_names) OR (raw, events, subj, class_names)
    ("proc_data", {}),      # language -> (X, y, subj, class_names, log)
    ("trained_models", {}), # (language_or_joint, model_name) -> (model, history, X_val, y_val)
    ("eval_results", {}),   # same keys -> evaluate_model() output
]:
    if key not in st.session_state:
        st.session_state[key] = default

st.title("EEG Imagined Speech Decoding -- Proof of Concept")
st.caption(
    "Demonstrates the full pipeline (preprocessing -> EEGNet/Conformer -> "
    "embeddings -> cross-lingual transfer -> evaluation -> interpretability) "
    "using synthetically generated EEG-like signals in place of the real "
    "Spanish and English recordings. Swap in real data via ZIP upload."
)
st.warning(
    "**This is a proof-of-concept, not a scientific result.** All signals "
    "below are synthetically generated to have a learnable class-dependent "
    "structure, so the pipeline has something real to find. Accuracy numbers "
    "here say nothing about performance on real EEG.",
    icon="⚠️",
)

STAGES = [
    "1. Dataset collection",
    "2. EEG preprocessing",
    "3 & 4. Decoding models",
    "5. Embedding extraction",
    "6. Cross-dataset transfer learning",
    "7. Embedding space analysis",
    "8. Evaluation metrics",
    "9. Visualizations & interpretability",
]
stage = st.sidebar.radio("Pipeline stage", STAGES)
device = "cpu"

# ========================================================================
# STAGE 1 -- Dataset collection
# ========================================================================
if stage == "1. Dataset collection":
    st.header("Stage 1 -- Dataset Collection")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Spanish dataset (Synthetic)")
        st.write(f"Subjects: {synthetic_data.DATASET_CONFIG['spanish']['n_subjects']}")
        st.write(f"Words: {', '.join(synthetic_data.DATASET_CONFIG['spanish']['words'])}")
        n_trials_es = st.slider("Trials per class (Spanish)", 10, 60, 30, key="n_es")
        if st.button("Generate Spanish dataset"):
            X, y, subj, cn = synthetic_data.generate_dataset("spanish", n_trials_per_class=n_trials_es, seed=1)
            st.session_state.raw_data["spanish"] = (X, y, subj, cn)
            st.success(f"Generated {X.shape[0]} trials, shape {X.shape}")

    with col2:
        st.subheader("English dataset (Synthetic)")
        st.write(f"Subjects: {synthetic_data.DATASET_CONFIG['english']['n_subjects']}")
        st.write(f"Words: {', '.join(synthetic_data.DATASET_CONFIG['english']['words'])}")
        n_trials_en = st.slider("Trials per class (English)", 10, 60, 30, key="n_en")
        if st.button("Generate English dataset"):
            X, y, subj, cn = synthetic_data.generate_dataset("english", n_trials_per_class=n_trials_en, seed=2)
            st.session_state.raw_data["english"] = (X, y, subj, cn)
            st.success(f"Generated {X.shape[0]} trials, shape {X.shape}")

    st.divider()
    st.subheader("Or upload your own real dataset (BIDS ZIP format)")
    st.caption("Upload a `.zip` file containing a standard BIDS structured EEG dataset. The POC will automatically extract it, find the EEG file and the events file.")
    up_lang = st.selectbox("Assign upload to language slot", ["spanish", "english"])
    up_zip = st.file_uploader("Upload Dataset ZIP", type=["zip"], key="upzip")
    
    if up_zip is not None and st.button("Extract and Load Dataset"):
        import zipfile
        import tempfile
        import os
        
        with st.spinner("Extracting and loading..."):
            temp_dir = tempfile.mkdtemp()
            with zipfile.ZipFile(up_zip, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the first .vhdr, .bdf, .set, or .fif
            raw_file = None
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(('.vhdr', '.bdf', '.set', '.fif')):
                        raw_file = os.path.join(root, file)
                        break
                if raw_file:
                    break
            
            if not raw_file:
                st.error("Could not find any EEG files (.vhdr, .bdf, .set, .fif) in the uploaded zip.")
            else:
                st.write(f"Found EEG file: {os.path.basename(raw_file)}")
                try:
                    if raw_file.endswith('.vhdr'):
                        raw = mne.io.read_raw_brainvision(raw_file, preload=True)
                    elif raw_file.endswith('.bdf'):
                        raw = mne.io.read_raw_bdf(raw_file, preload=True)
                    elif raw_file.endswith('.set'):
                        raw = mne.io.read_raw_eeglab(raw_file, preload=True)
                    elif raw_file.endswith('.fif'):
                        raw = mne.io.read_raw_fif(raw_file, preload=True)
                        
                    # Find events.tsv if possible, else dummy
                    events_file = None
                    for root, dirs, files in os.walk(temp_dir):
                        for file in files:
                            if file.endswith('events.tsv'):
                                events_file = os.path.join(root, file)
                                break
                        if events_file:
                            break
                            
                    if events_file:
                        st.write(f"Found events file: {os.path.basename(events_file)}")
                        events_df = pd.read_csv(events_file, sep='\t')
                        
                        # Fallback for trial_type if not available
                        if 'trial_type' in events_df.columns:
                            labels_str = events_df['trial_type'].values
                        elif 'value' in events_df.columns:
                            labels_str = events_df['value'].values
                        else:
                            labels_str = np.ones(len(events_df))

                        unique_labels = sorted(set(labels_str))
                        label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
                        labels_int = np.array([label_map[l] for l in labels_str])
                        
                        events = np.column_stack([
                            (events_df['onset'].values * raw.info['sfreq']).astype(int),
                            np.zeros(len(events_df)),
                            labels_int
                        ]).astype(int)
                        
                        cn = [str(l) for l in unique_labels]
                    else:
                        st.warning("No events.tsv found. Creating dummy events (10 trials).")
                        # create 10 evenly spaced dummy events
                        events = np.array([[(i+1)*int(raw.info['sfreq']*2), 0, 1] for i in range(10)], dtype=int)
                        cn = ["Dummy_Class"]

                    st.session_state.raw_data[up_lang] = (raw, events, np.zeros(len(events), dtype=int), cn)
                    st.success(f"Successfully loaded MNE raw object! Length: {raw.times[-1]:.2f}s, Channels: {len(raw.ch_names)}")
                    
                except Exception as e:
                    st.error(f"Error loading EEG file: {e}")

    st.divider()
    for lang, data_tuple in st.session_state.raw_data.items():
        if isinstance(data_tuple[0], mne.io.BaseRaw):
            raw, events, subj, cn = data_tuple
            st.write(f"**{lang}** (Real MNE Data): {len(events)} trials | {len(raw.ch_names)} channels | Fs: {raw.info['sfreq']} Hz | classes: {cn}")
        else:
            X, y, subj, cn = data_tuple
            st.write(f"**{lang}** (Synthetic Data): {X.shape[0]} trials | {X.shape[1]} channels | {X.shape[2]} timepoints | classes: {cn}")

# ========================================================================
# STAGE 2 -- EEG preprocessing
# ========================================================================
elif stage == "2. EEG preprocessing":
    st.header("Stage 2 -- EEG Preprocessing")
    if not st.session_state.raw_data:
        st.info("Generate or upload a dataset in Stage 1 first.")
    else:
        lang = st.selectbox("Dataset", list(st.session_state.raw_data.keys()))
        data_tuple = st.session_state.raw_data[lang]

        if isinstance(data_tuple[0], mne.io.BaseRaw):
            st.subheader("Literature-Based Preprocessing Pipeline (Real Data)")
            c1, c2, c3 = st.columns(3)
            with c1:
                low = st.number_input("Bandpass low (Hz)", 0.1, 10.0, 4.0)
                high = st.number_input("Bandpass high (Hz)", 10.0, 150.0, 100.0)
            with c2:
                notch = st.number_input("Notch frequency (Hz)", 40.0, 70.0, 50.0)
                target_sfreq = st.number_input("Target Resampling Freq (Hz)", 100, 1000, 250)
            with c3:
                tmin = st.number_input("Epoch start (s)", -2.0, 5.0, 1.5)
                tmax = st.number_input("Epoch end (s)", -2.0, 5.0, 3.5)

            if st.button("Run Literature Pipeline"):
                from modules.preprocessing_literature import LiteraturePreprocessingPipeline
                pipeline = LiteraturePreprocessingPipeline(
                    target_sfreq=target_sfreq, l_freq=low, h_freq=high, notch_freqs=[notch]
                )
                
                with st.spinner("Preprocessing with MNE (Filtering, ICA, Epoching)..."):
                    raw, events, subj, cn = data_tuple
                    event_id = {str(c): c for c in set(events[:, 2])}
                    
                    try:
                        raw_copy = raw.copy()
                        Xp, final_events, epochs = pipeline.run_pipeline(raw_copy, events, event_id, tmin=tmin, tmax=tmax)
                        y = final_events[:, 2]
                        st.session_state.proc_data[lang] = (Xp, y, subj[:len(y)], cn, ["MNE preprocessing successful."])
                        st.success("Preprocessing complete. (Gamma band preserved!)")
                        
                        st.subheader("PSD Before Preprocessing")
                        fig = raw.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig)
                        
                        st.subheader("PSD After Preprocessing (Epochs)")
                        fig2 = epochs.compute_psd(fmax=120).plot(show=False)
                        st.pyplot(fig2)
                        
                    except Exception as e:
                        st.error(f"Error during preprocessing: {e}")
        else:
            st.subheader("Legacy Synthethic Preprocessing")
            c1, c2, c3 = st.columns(3)
            with c1:
                low = st.number_input("Bandpass low (Hz)", 0.1, 10.0, 1.0)
                high = st.number_input("Bandpass high (Hz)", 10.0, 60.0, 40.0)
            with c2:
                notch = st.number_input("Notch frequency (Hz)", 40.0, 70.0, 50.0)
                z_thresh = st.number_input("Artifact clip |z| threshold", 2.0, 10.0, 6.0)
            with c3:
                baseline_samples = st.number_input("Baseline samples", 1, 50, 10)
                do_norm = st.checkbox("Z-score normalize", value=True)

            if st.button("Run preprocessing"):
                X, y, subj, cn = data_tuple
                Xp, log = preprocessing.run_pipeline(
                    X, fs=synthetic_data.FS, low=low, high=high, notch=notch,
                    z_thresh=z_thresh, baseline_samples=int(baseline_samples), do_normalize=do_norm,
                )
                st.session_state.proc_data[lang] = (Xp, y, subj, cn, log)
                st.success("Preprocessing complete.")
                for l in log:
                    st.write("-", l)

            if lang in st.session_state.proc_data:
                X, y, subj, cn = data_tuple
                Xp, _, _, _, log = st.session_state.proc_data[lang]
                st.divider()
                st.subheader("Raw vs. preprocessed (trial 0, channel 0)")
                fig, ax = plt.subplots(figsize=(8, 3))
                t = np.arange(X.shape[-1]) / synthetic_data.FS
                ax.plot(t, X[0, 0], label="raw", alpha=0.6)
                ax.plot(t, Xp[0, 0], label="preprocessed", alpha=0.9)
                ax.set_xlabel("Time (s)")
                ax.legend()
                st.pyplot(fig)

# ========================================================================
# STAGE 3 & 4 -- Decoding models
# ========================================================================
elif stage == "3 & 4. Decoding models":
    st.header("Stage 3 & 4 -- Baseline (EEGNet) and Representation (Conformer) Models")
    if not st.session_state.proc_data:
        st.info("Run preprocessing in Stage 2 first.")
    else:
        lang = st.selectbox("Dataset", list(st.session_state.proc_data.keys()))
        Xp, y, subj, cn, _ = st.session_state.proc_data[lang]

        model_name = st.radio("Model", ["eegnet", "conformer"], horizontal=True)
        epochs = st.slider("Training epochs", 5, 60, 25)
        test_size = st.slider("Validation split fraction", 0.1, 0.4, 0.25)

        if st.button("Train model"):
            X_train, X_val, y_train, y_val = training.split_data(Xp, y, test_size=test_size, seed=0)
            n_channels, n_timepoints = Xp.shape[1], Xp.shape[2]
            n_classes = len(cn)
            model = models.build_model(model_name, n_channels, n_timepoints, n_classes)

            progress_bar = st.progress(0)
            status = st.empty()

            def cb(epoch, total, tl, vl, va):
                progress_bar.progress(epoch / total)
                status.write(f"Epoch {epoch}/{total} -- train loss {tl:.3f} | val loss {vl:.3f} | val acc {va:.3f}")

            model, hist = training.train_model(
                model, X_train, y_train, X_val, y_val, epochs=epochs, progress_callback=cb
            )
            key = (lang, model_name)
            st.session_state.trained_models[key] = (model, hist, X_val, y_val, cn)
            st.success(f"Training complete. Final validation accuracy: {hist['val_acc'][-1]:.3f}")

            fig, ax = plt.subplots(figsize=(7, 3))
            ax.plot(hist["train_loss"], label="train loss")
            ax.plot(hist["val_loss"], label="val loss")
            ax.set_xlabel("Epoch")
            ax.legend()
            st.pyplot(fig)

        st.divider()
        st.write("**Trained models this session:**")
        for (l, m) in st.session_state.trained_models:
            _, hist, *_ = st.session_state.trained_models[(l, m)]
            st.write(f"- {l} / {m}: final val acc = {hist['val_acc'][-1]:.3f}")

# ========================================================================
# STAGE 5 -- Embedding extraction
# ========================================================================
elif stage == "5. Embedding extraction":
    st.header("Stage 5 -- Embedding Extraction")
    if not st.session_state.trained_models:
        st.info("Train a model in Stage 3/4 first.")
    else:
        keys = list(st.session_state.trained_models.keys())
        sel = st.selectbox("Trained model", keys, format_func=lambda k: f"{k[0]} / {k[1]}")
        model, hist, X_val, y_val, cn = st.session_state.trained_models[sel]

        if st.button("Extract embeddings"):
            emb = embedding_analysis.extract_embeddings(model, X_val)
            st.session_state[f"emb_{sel}"] = emb
            st.success(f"Extracted embeddings: shape {emb.shape}")
            st.dataframe(pd.DataFrame(emb[:10, :8]).round(3),
                         use_container_width=True)
            st.caption("Showing first 10 trials, first 8 embedding dimensions.")

# ========================================================================
# STAGE 6 -- Cross-dataset transfer learning
# ========================================================================
elif stage == "6. Cross-dataset transfer learning":
    st.header("Stage 6 -- Cross-Dataset / Cross-Lingual Transfer Learning")
    st.caption(
        "Experiment A trains one model per language from scratch (no shared weights). "
        "Experiment B pretrains jointly on the pooled Spanish + English data, then "
        "fine-tunes per language."
    )
    if "spanish" not in st.session_state.proc_data or "english" not in st.session_state.proc_data:
        st.info("Preprocess both the Spanish and English datasets in Stage 2 first.")
    else:
        model_name = st.radio("Model architecture", ["eegnet", "conformer"], horizontal=True, key="transfer_model")
        epochs_a = st.slider("Epochs (Experiment A, per language)", 5, 60, 25, key="ep_a")
        epochs_pretrain = st.slider("Epochs (Experiment B, joint pretrain)", 5, 60, 25, key="ep_pre")
        epochs_ft = st.slider("Epochs (Experiment B, fine-tune)", 3, 30, 10, key="ep_ft")

        if st.button("Run Experiment A: train from scratch per language"):
            results_a = {}
            for lang in ["spanish", "english"]:
                Xp, y, subj, cn, _ = st.session_state.proc_data[lang]
                X_train, X_val, y_train, y_val = training.split_data(Xp, y, test_size=0.25, seed=0)
                n_channels, n_timepoints = Xp.shape[1], Xp.shape[2]
                model = models.build_model(model_name, n_channels, n_timepoints, len(cn))
                model, hist = training.train_model(model, X_train, y_train, X_val, y_val, epochs=epochs_a)
                res = training.evaluate_model(model, X_val, y_val)
                results_a[lang] = res["accuracy"]
            st.session_state["exp_a_results"] = results_a
            st.success("Experiment A complete.")

        if st.button("Run Experiment B: joint pretraining + per-language fine-tune"):
            data_es = st.session_state.proc_data["spanish"]
            data_en = st.session_state.proc_data["english"]
            n_channels = data_es[0].shape[1]
            n_timepoints = data_es[0].shape[2]

            X_pool = np.concatenate([data_es[0], data_en[0]], axis=0)
            y_pool = np.concatenate([np.zeros(len(data_es[1])), np.ones(len(data_en[1]))]).astype(np.int64)
            Xp_train, Xp_val, yp_train, yp_val = training.split_data(X_pool, y_pool, test_size=0.2, seed=0)

            backbone = models.build_model(model_name, n_channels, n_timepoints, 2)
            backbone, _ = training.train_model(backbone, Xp_train, yp_train, Xp_val, yp_val, epochs=epochs_pretrain)
            st.write("Joint pretraining (language-id proxy task) complete.")

            results_b = {}
            for lang, data in [("spanish", data_es), ("english", data_en)]:
                Xp, y, subj, cn, _ = data
                X_train, X_val, y_train, y_val = training.split_data(Xp, y, test_size=0.25, seed=0)

                ft_model = models.build_model(model_name, n_channels, n_timepoints, len(cn))
                own_state = ft_model.state_dict()
                pretrained_state = backbone.state_dict()
                transfer_state = {
                    k: v for k, v in pretrained_state.items()
                    if k in own_state and "classifier" not in k and own_state[k].shape == v.shape
                }
                own_state.update(transfer_state)
                ft_model.load_state_dict(own_state)

                ft_model, hist = training.train_model(ft_model, X_train, y_train, X_val, y_val, epochs=epochs_ft)
                res = training.evaluate_model(ft_model, X_val, y_val)
                results_b[lang] = res["accuracy"]
            st.session_state["exp_b_results"] = results_b
            st.success("Experiment B complete.")

        st.divider()
        if "exp_a_results" in st.session_state or "exp_b_results" in st.session_state:
            rows = []
            for lang in ["spanish", "english"]:
                rows.append({
                    "Language": lang,
                    "Experiment A (per-language, from scratch)": st.session_state.get("exp_a_results", {}).get(lang, None),
                    "Experiment B (joint pretrain + fine-tune)": st.session_state.get("exp_b_results", {}).get(lang, None),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True)

# ========================================================================
# STAGE 7 -- Embedding space analysis
# ========================================================================
elif stage == "7. Embedding space analysis":
    st.header("Stage 7 -- Embedding Space Analysis")
    emb_keys = [k for k in st.session_state.keys() if k.startswith("emb_")]
    if not emb_keys:
        st.info("Extract embeddings in Stage 5 first.")
    else:
        sel_key = st.selectbox("Embedding set", emb_keys)
        model_key = eval(sel_key.replace("emb_", ""))
        emb = st.session_state[sel_key]
        _, _, X_val, y_val, cn = st.session_state.trained_models[model_key]

        method = st.radio("Reduction method", ["PCA", "t-SNE", "UMAP"], horizontal=True)
        if st.button("Compute projection"):
            if method == "PCA":
                proj = embedding_analysis.reduce_pca(emb)
            elif method == "t-SNE":
                proj = embedding_analysis.reduce_tsne(emb)
            else:
                proj = embedding_analysis.reduce_umap(emb)

            fig, ax = plt.subplots(figsize=(6, 5))
            for cls_idx, cls_name in enumerate(cn):
                mask = y_val == cls_idx
                ax.scatter(proj[mask, 0], proj[mask, 1], label=cls_name, alpha=0.7, s=25)
            ax.legend(fontsize=8)
            ax.set_title(f"{method} projection of learned embeddings")
            st.pyplot(fig)

            qm = embedding_analysis.embedding_quality_metrics(emb, y_val)
            st.subheader("Embedding quality metrics")
            st.json(qm)

# ========================================================================
# STAGE 8 -- Evaluation metrics
# ========================================================================
elif stage == "8. Evaluation metrics":
    st.header("Stage 8 -- Evaluation Metrics")
    if not st.session_state.trained_models:
        st.info("Train a model in Stage 3/4 first.")
    else:
        keys = list(st.session_state.trained_models.keys())
        sel = st.selectbox("Trained model", keys, format_func=lambda k: f"{k[0]} / {k[1]}")
        model, hist, X_val, y_val, cn = st.session_state.trained_models[sel]

        res = training.evaluate_model(model, X_val, y_val)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{res['accuracy']:.3f}")
        c2.metric("Precision (macro)", f"{res['precision']:.3f}")
        c3.metric("Recall (macro)", f"{res['recall']:.3f}")
        c4.metric("F1 (macro)", f"{res['f1']:.3f}")

        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(res["confusion_matrix"], cmap="Blues")
        ax.set_xticks(range(len(cn))); ax.set_xticklabels(cn, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(cn))); ax.set_yticklabels(cn, fontsize=7)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        for i in range(len(cn)):
            for j in range(len(cn)):
                ax.text(j, i, str(res["confusion_matrix"][i, j]), ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        st.pyplot(fig)

# ========================================================================
# STAGE 9 -- Visualizations & interpretability
# ========================================================================
elif stage == "9. Visualizations & interpretability":
    st.header("Stage 9 -- Visualizations & Interpretability")
    if not st.session_state.trained_models:
        st.info("Train a model in Stage 3/4 first.")
    else:
        keys = list(st.session_state.trained_models.keys())
        sel = st.selectbox("Trained model", keys, format_func=lambda k: f"{k[0]} / {k[1]}")
        model, hist, X_val, y_val, cn = st.session_state.trained_models[sel]
        lang = sel[0]
        
        # Determine channel names
        if lang in st.session_state.raw_data and isinstance(st.session_state.raw_data[lang][0], mne.io.BaseRaw):
            ch_names = st.session_state.raw_data[lang][0].ch_names
            fs = st.session_state.raw_data[lang][0].info['sfreq']
        else:
            ch_names = synthetic_data.channel_names(X_val.shape[1])
            fs = synthetic_data.FS

        tabs = st.tabs(["Scalp topography", "ERP waveform", "Attention / Grad-CAM"])

        with tabs[0]:
            trial_idx = st.slider("Trial index", 0, X_val.shape[0] - 1, 0, key="topo_trial")
            time_idx = st.slider("Timepoint", 0, X_val.shape[2] - 1, X_val.shape[2] // 2, key="topo_time")
            values = X_val[trial_idx, :, time_idx]
            fig = interpretability.plot_scalp_topography(values, ch_names, title=f"Trial {trial_idx}, t-index {time_idx}")
            st.pyplot(fig)

        with tabs[1]:
            channel_idx = st.slider("Channel", 0, X_val.shape[1] - 1, 0, key="erp_channel")
            fig = interpretability.plot_erp(X_val, y_val, cn, ch_names, fs, channel_idx=channel_idx)
            st.pyplot(fig)

        with tabs[2]:
            trial_idx2 = st.slider("Trial index", 0, X_val.shape[0] - 1, 0, key="interp_trial")
            x_single = torch.tensor(X_val[trial_idx2:trial_idx2 + 1], dtype=torch.float32)
            true_class = int(y_val[trial_idx2])

            if sel[1] == "eegnet":
                cam = interpretability.grad_cam_eegnet(model, x_single, target_class=true_class)
                fig = interpretability.plot_grad_cam(
                    X_val[trial_idx2], cam, ch_names, fs,
                    title=f"Grad-CAM -- true class: {cn[true_class]}",
                )
                st.pyplot(fig)
            else:
                attn = model.attention_weights(x_single)[0].detach().numpy()
                fig = interpretability.plot_attention_map(attn, title=f"Attention map -- true class: {cn[true_class]}")
                st.pyplot(fig)
