# EEG Imagined Speech Decoding -- Proof of Concept

A Streamlit application demonstrating the full 9-stage methodology pipeline
end to end: dataset collection, EEG preprocessing, dual decoding models
(EEGNet baseline + Conformer representation model), embedding extraction,
cross-lingual transfer learning, embedding-space analysis, evaluation
metrics, and interpretability visualizations.

## Why synthetic data

No real EEG recordings are bundled with this POC. `modules/synthetic_data.py`
generates synthetic, EEG-like multi-channel signals with a class-dependent
structure (a distinct oscillatory signature per imagined word, plus
per-subject variability and occasional artifact bursts), standing in for
the real Spanish (6 subjects, 5 words) and English (8 subjects, 4 words)
datasets. This lets every stage of the pipeline -- including training,
embeddings, and interpretability -- run and be inspected without the actual
recordings on hand.

**Every number this app produces (accuracy, embedding quality, etc.) is a
demonstration of pipeline mechanics only. None of it is evidence about real
EEG decoding performance.**

## Using real data instead

The app also accepts real data via file upload (Stage 1): a `.npy` array of
shape `(n_trials, n_channels, n_timepoints)` and a matching `.npy` labels
array. Alternatively, replace calls to `synthetic_data.generate_dataset()`
in `app.py` with your own loader that returns arrays in the same shape.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Then open the local URL Streamlit prints (usually `http://localhost:8501`).

## Pipeline stages (sidebar navigation)

1. **Dataset collection** -- generate synthetic Spanish/English datasets, or upload your own.
2. **EEG preprocessing** -- bandpass filter, notch filter, baseline correction, artifact clipping, normalization. Adjustable parameters, with a raw-vs-processed plot.
3. **& 4. Decoding models** -- train EEGNet or the Conformer-style model, with a live loss curve.
4. **Embedding extraction** -- pull penultimate-layer feature vectors from a trained model.
5. **Cross-dataset transfer learning** -- Experiment A (train from scratch per language) vs. Experiment B (joint pretraining + per-language fine-tuning), compared side by side.
6. **Embedding space analysis** -- PCA / t-SNE / UMAP projections, plus silhouette score and intra-/inter-class distance.
7. **Evaluation metrics** -- accuracy, precision, recall, F1, confusion matrix.
8. **Visualizations & interpretability** -- scalp topography, ERP waveforms, Grad-CAM (EEGNet), and self-attention maps (Conformer).

## Notes on Experiment B (cross-lingual transfer)

The Spanish and English datasets have different vocabularies (5 words vs. 4
words), so there is no single shared classification head across languages.
Experiment B instead pretrains the shared feature-extraction backbone
(everything except the final classification layer) on a language-ID proxy
task over the pooled data, then transplants those weights into a
fresh per-language model before fine-tuning its own classification head.
This illustrates the mechanism of joint pretraining + fine-tuning; a full
implementation on real data may use a different self-supervised or
multi-task pretraining objective.

## Project structure

```
eeg_poc/
├── app.py                       # Streamlit UI, one section per pipeline stage
├── requirements.txt
├── README.md
└── modules/
    ├── synthetic_data.py        # Stage 1: synthetic dataset generator
    ├── preprocessing.py         # Stage 2: filtering, artifact handling, normalization
    ├── models.py                # Stage 3/4: EEGNet and Conformer (PyTorch)
    ├── training.py               # training loop + evaluation metrics
    ├── embedding_analysis.py    # Stage 5/7: embedding extraction, PCA/t-SNE/UMAP, quality metrics
    └── interpretability.py      # Stage 9: scalp topography, ERP, Grad-CAM, attention maps
```
