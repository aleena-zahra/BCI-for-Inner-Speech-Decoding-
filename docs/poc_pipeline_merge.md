# Merged POC preprocessing and interpretability

Voice retains its multipage dashboard and imports no POC modules at runtime.
The POC implementations have been refactored into reusable library operations.
Cross-language transfer remains excluded.

## Trial-array workflow

```python
from voice import EEGPreprocessor, InnerSpeechPipeline

processor = EEGPreprocessor(
    sfreq=512, target_sfreq=250, lowcut=0.5, highcut=100,
    filter_method="fir", notch_freqs=(50,),
    common_average_reference=True, t_start=1.5, t_end=3.5,
    zscore=True,
)
model = InnerSpeechPipeline(preprocessor=processor)
scores = model.evaluate_trial_holdout(signals, labels, subjects)
```

Arrays are `(trial, channel, time)` in microvolts. FIR mode filters complete
input trials, resamples, then crops the selected interval. The legacy IIR
workflow remains available. Short already-epoched trials may be shorter than
the FIR kernel at a low cutoff; MNE reports this condition. Use continuous
filtering before epoching when recording boundaries and filter length matter.

ICA is optional. `ica_components` controls decomposition and `ica_exclude`
selects explicitly reviewed zero-based indices for that fitted ICA. Component
indices change across fits. Dashboard evaluation uses a separate review for
each training fold and passes its exclusions through `ica_exclusions`, a
mapping from fold name to component indices. A trial holdout fold is named
`trial holdout`; a LOSO fold is named after its held-out subject.

The dashboard displays training-fold component traces and spatial loadings,
then deterministically refits the same training subset with its saved
exclusions. Validation and test data never fit that ICA. EEGNet/Conformer
evaluation similarly accepts `preprocessor=processor` and operates on raw
trial arrays, fitting preprocessing inside each fold. Amplitude rejection
precedes normalization, with label/subject/run rows kept aligned.

## Continuous MNE workflow

```python
from voice import LiteraturePreprocessingPipeline

pipeline = LiteraturePreprocessingPipeline(
    target_sfreq=250, l_freq=0.5, h_freq=100,
    notch_freqs=(50,), apply_car=True, zscore=True,
)
trials, retained_events, epochs = pipeline.run_pipeline(
    raw, events, event_id, tmin=1.5, tmax=3.5
)
```

The input Raw and events are copied. Scalp EEG is picked, FIR filters run
before resampling, MNE resamples the events together with the recording, and
CAR/optional reviewed ICA precede epoch extraction. Returned arrays use
microvolts unless z-scored; the returned MNE Epochs remain in volts and can be
used for inspection. Supply training recordings separately when fitting ICA
for held-out evaluation. A selected time interval does not establish absence
of cue-evoked activity, and no automatic EOG/EMG identification is claimed.

## Interpretation

`LatentSpaceAuditor.audit(embeddings, labels, subjects, method="PCA")` returns
the selected projection plus independent class and subject clustering metrics.
`ClusteringMetrics` exposes the POC-style silhouette and intra/inter-distance
entry points backed by the same implementation. Undefined cluster metrics are
`None` (displayed as N/A), and distance sampling is bounded for large datasets.

The dashboard maintains fold-aligned embeddings and allows word/subject
coloring. It does not mix latent vectors from separately trained networks or
CSP fits. EEGNet sensitivity can target predicted or specified output classes.
`conformer_attention_summary(..., head=None, batch_size=8)` summarizes all
heads or one selected head, in batches. Layer heatmaps and mean received
attention use actual model patch positions and the preprocessing sampling rate.
Attention and gradients describe model behavior rather than causal importance.
