"""Evaluation and interpretability subpackage."""

from voice.evaluation.metrics import evaluate_classifier

__all__ = [
    "LatentSpaceAuditor",
    "evaluate_classifier",
]


def __getattr__(name: str):
    if name == "LatentSpaceAuditor":
        from voice.evaluation.latent_space import LatentSpaceAuditor

        return LatentSpaceAuditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")