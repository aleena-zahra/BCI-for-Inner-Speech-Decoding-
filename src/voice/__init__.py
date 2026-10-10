from eeg_voice.data.loader import scan_bids_eeg_dataset
from eeg_voice.preprocessing.pipeline import EEGPreprocessor
from eeg_voice.analysis.features import MultiBandCSPExtractor
from eeg_voice.models.svm import CSPSVMClassifier

__version__ = "0.1.0"

__all__ = [
    "scan_bids_eeg_dataset",
    "EEGPreprocessor",
    "MultiBandCSPExtractor",
    "CSPSVMClassifier",
]