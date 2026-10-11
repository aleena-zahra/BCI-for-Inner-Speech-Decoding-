"""Models subpackage for Voice BCI Inner Speech Decoding."""

from voice.models.svm import CSPSVMClassifier

__all__ = [
    "CSPSVMClassifier",
    "EEGNet",
    "EEGConformer",
    "RevIN",
]


def __getattr__(name: str):
    if name == "EEGNet":
        from voice.models.eegnet import EEGNet

        return EEGNet
    if name == "EEGConformer":
        from voice.models.eegconformer import EEGConformer

        return EEGConformer
    if name == "RevIN":
        from voice.models.layers import RevIN

        return RevIN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
