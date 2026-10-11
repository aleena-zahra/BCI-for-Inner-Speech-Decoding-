# Phase 2: POC capability comparison and migration

Phase 2 moves reusable research capabilities into the `voice` library and uses
the Streamlit dashboard as their interface. The POC remains a comparison
reference; its code is not copied wholesale.

| POC capability | Phase 2 status in Voice | Notes |
| --- | --- | --- |
| EEGNet and Conformer training (Step 4) | Migrated and refactored | `voice.models` contains the classifiers and reusable fold-based trainer. The dashboard exposes stratified trial holdout and leave-one-subject-out evaluation. PyTorch remains optional. |
| Temporal patches, Welch spectral summaries, RevIN (Step 3) | Migrated as reusable modules | `EEGPatchTokenizer`, `SpectralFeatureExtractor`, and `RevIN` are in `voice.preprocessing` / `voice.models`; feature controls and preview are in the dashboard. |
| Latent projection and cluster metrics (Step 5) | Migrated and generalized | PCA, t-SNE, optional UMAP, silhouette, and intra/inter-class distance operate on fold-aligned model embeddings. |
| Model interpretation (Step 5) | Migrated with scoped claims | EEGNet input-gradient channel sensitivity and Conformer attention summaries are available. Attention summaries are not causal explanations. |
| POC CSP/SVM training | Already implemented in Voice before Phase 2 | Voice keeps a single library pipeline and exposes its evaluation through the dashboard. |
| BIDS ingestion and event-locked trials (`POC/app.py`) | Existing Voice loader retained | Voice requires real event labels and validates channel/rate alignment; it does not synthesize labels when events are absent. |
| POC literature preprocessing (Step 2) | Not copied as-is | The POC ICA path does not reliably identify/remove EOG/EMG components, and some advertised steps are not a validated reusable workflow. Voice retains its explicit trial preprocessing; ICA artifact rejection remains future work requiring validated references. |
| Subject embedding described in POC prose | Demonstrative, not migrated | No usable implementation was found in the POC feature module. |
| POC cross-language accuracy | Replaced by explicit concept-mapped experiment | Directly comparing class indices is invalid when datasets encode different word meanings. Voice requires user-entered shared concepts for every source and target class, then reports only `concept_mapped_accuracy`. |
| POC synthetic/demo outputs | Not treated as evidence | Demonstration metrics are not real dataset results and are not surfaced as model evaluation. |

## Cross-language experiment requirements

Load and preprocess a source BIDS dataset and train a model in the Pipeline
pages. Open **Cross-language transfer**, load the target BIDS dataset, and enter
the shared semantic meaning of each class on both sides. The page rejects
incomplete or many-to-one class mappings, missing source classes, and channel
order/name mismatches. It evaluates the separately loaded target trials using
the source model and source preprocessing configuration; it does not overwrite
the active source data or the source model's held-out evaluation.

The dashboard reports concept-mapped accuracy and a confusion matrix in the
shared concept space. This is still an experimental cross-dataset result: it is
not evidence of language transfer unless the concepts genuinely correspond
across languages and the evaluation design controls subject and dataset
differences. The page does not claim cross-lingual generalization from raw
cross-dataset class-index agreement.

## Current data constraints

Voice's loader accepts BIDS datasets with supported EEG recordings and actual
event labels. Archives containing only CSV/MAT data, documentation, or EEG with
no usable event annotations cannot enter this BIDS path. Channel harmonization
between distinct montages is not implemented; transfer currently requires the
same channel names and order. EEGNet/Conformer and UMAP need their optional
dependencies (`pip install -e '.[deep-learning]'`).
