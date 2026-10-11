from voice.data.loader import (
    extract_bids_archive,
    parse_and_load_eeg,
    scan_bids_eeg_dataset,
)
from voice.data.trials import EEGTrials, load_bids_trials
from voice.preprocessing.pipeline import EEGPreprocessor
from voice.evaluation.features import MultiBandCSPExtractor
from voice.models.svm import CSPSVMClassifier
from voice.pipeline import InnerSpeechPipeline

__version__ = "0.1.0"

__all__ = [
    "scan_bids_eeg_dataset",
    "parse_and_load_eeg",
    "extract_bids_archive",
    "EEGTrials",
    "load_bids_trials",
    "EEGPreprocessor",
    "MultiBandCSPExtractor",
    "CSPSVMClassifier",
    "InnerSpeechPipeline",
    "launch_app",
]
