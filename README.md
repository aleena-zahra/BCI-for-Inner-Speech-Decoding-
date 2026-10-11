# Voice: EEG Inner Speech Analysis

Voice is a Python library and Streamlit dashboard for loading event-labeled BIDS
EEG, preprocessing trials, extracting temporal/spectral features, training
classifiers, and inspecting held-out model representations. The dashboard is
the interface to reusable functionality in `src/voice`; it does not implement
a separate analysis pipeline. Cross-language transfer is outside the project
scope.

## Install and run

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui,dev]"
voice-ui
```

The Voice dashboard provides these library-backed pages:

1. **Dataset collection** — upload a BIDS ZIP or load a local BIDS folder; inspect trials, subjects, channels, labels, and recordings.
2. **EEG preprocessing** — configure filtering, notch frequency, resampling, epoch window, common-average reference, baseline handling, and optional amplitude rejection; compare PSD before and after.
3. **Feature extraction and patching** — configure overlapping patches and RevIN, and inspect spectral band summaries.
4. **Decoding models** — train CSP + SVM, EEGNet, or EEGConformer with trial holdout or leave-one-subject-out evaluation.
5. **Latent analysis and interpretability** — view PCA/t-SNE/UMAP projections and clustering metrics, plus EEGNet channel sensitivity or Conformer attention summaries.

EEGNet and EEGConformer require PyTorch. UMAP is also optional:

```powershell
python -m pip install -e ".[ui,dev,deep-learning]"
```

## Data requirements

The BIDS loader supports event-labeled EEG in `.bdf`, `.set`, `.fif`, or
`.vhdr` recordings. Event labels must be present in a matching `events.tsv` or
usable recording annotations. It will not invent labels for unlabeled data.
The current loader also requires consistent channel names/order and sampling
rate across runs in one dataset. CSV-only and MATLAB archives are not accepted
by this BIDS ingestion path.

## Development checks

```powershell
python -m pytest
```

The neural model tests require the optional PyTorch dependency.

## Merged POC preprocessing and interpretation

In **Preprocessing**, choose **POC literature FIR** for the 0.5–100 Hz,
250 Hz resampling, CAR, and per-channel z-score profile. Cutoffs and the
event-relative interval remain configurable. Optional amplitude rejection is
measured in microvolts before normalization. The before/after PSD plots show
the actual transformed trials.

ICA only removes explicitly reviewed components. Review the preview
decomposition on the Preprocessing page, then review and save exclusions
separately for each training fold in **Training**. Evaluation fits ICA on the
training subset and applies that fitted decomposition to validation/test data.
Unreviewed folds do not exclude components. The preview's component indices
are not reused across different decompositions.

**Interpretability** supports coloring held-out embeddings by word class or
subject, class/subject silhouettes, intra/inter-class distance, EEGNet
class-specific input-gradient sensitivity, and Conformer layer/head attention
with event-relative patch timing. The subject silhouette is N/A for a fold
with only one subject; these views do not prove causal or semantic decoding.

See [the merged pipeline API and limitations](docs/poc_pipeline_merge.md) for
continuous MNE Raw/events processing and library examples.
