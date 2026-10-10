from voice.data.loader import scan_bids_eeg_dataset, parse_and_load_eeg
from voice.preprocessing.pipeline import EEGPreprocessor
from voice.analysis.features import MultiBandCSPExtractor
from voice.models.baseline_svm import CSPSVMClassifier

__version__ = "0.1.0"

__all__ = [
    "scan_bids_eeg_dataset",
    "parse_and_load_eeg",
    "EEGPreprocessor",
    "MultiBandCSPExtractor",
    "CSPSVMClassifier"
]