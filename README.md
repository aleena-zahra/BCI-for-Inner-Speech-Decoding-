# BCI for Inner Speech Decoding
Inner Speech Decoding on open-source datasets


This project explores EEG-based inner speech decoding using the Inner Speech Dataset. The goal is to classify imagined speech directions (e.g., Up vs Down) from brain activity using signal processing and machine learning techniques.

# Project Overview

EEG signals are highly noisy and low signal-to-noise ratio, especially for inner speech tasks. This project builds a full pipeline to:

  1. preprocess raw EEG signals
  2. extract meaningful spatial-frequency features
  3. classify mental states using machine learning
  4. evaluate performance using proper validation techniques

The final system uses Common Spatial Patterns (CSP) and Support Vector Machines (SVM), with extensions into dual-band CSP (alpha + beta bands).

```
Project Structure
project/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_preprocessing_and_features.ipynb
│   ├── 03_classification_and_evaluation.ipynb
│
├── utils.py
│   └── helper functions (filtering, CSP, plotting)
│
└── README.md
```

## Pipeline Steps
  ### 1. Data Loading
      Dataset: Inner Speech EEG Dataset
      Subjects: single-subject analysis (expandable)
      Signals: 128-channel EEG recordings
  ### 2. Preprocessing
      Resampling (256 Hz → 128 Hz)
      Bandpass filtering (1–40 Hz)
      Notch filtering (50 Hz)
      Time window selection (1.5s–3.5s post-stimulus)
  ### 3. Feature Extraction
      Common Spatial Patterns (CSP)
      Dual-band CSP:
      Alpha band (8–12 Hz)
      Beta band (13–30 Hz)
      Feature concatenation across bands
  ### 4. Classification
      Support Vector Machine (SVM)
      Linear kernel
      Stratified train-test split

## Results
  Baseline CSP performance: ~55–60%
  Dual-band CSP performance: ~60–67%
  Best observed accuracy: ~66%

Performance varies due to:

    high inter-class EEG overlap
    low signal-to-noise ratio in inner speech signals
    subject variability
## Evaluation Metrics
    Accuracy
    Confusion Matrix
    Cross-validation scores
    Decision score distribution (SVM margin analysis)
    ROC-AUC score
    
## Key Insight

Inner speech EEG signals are weak and highly overlapping. Improvements in performance come more from:

    better feature representations
    frequency-specific filtering
    spatial filtering (CSP)


## Requirements
  Python 3.8+
  numpy
  scipy
  scikit-learn
  mne
  matplotlib
## How to Run

Clone repository
Load dataset (Inner Speech Dataset)
Run notebooks in order:
01_data_exploration.ipynb
02_preprocessing_and_features.ipynb
03_classification_and_evaluation.ipynb
Inner speech classification is a challenging EEG task with inherently overlapping signals
## Reference
Nieto et al., Thinking out loud: an open-access EEG-based BCI dataset for inner speech recognition, Scientific Data (2022)
