"""Evaluation and interpretability subpackage."""

from voice.evaluation.metrics import evaluate_classifier

__all__ = [
    "LatentSpaceAuditor",
    "eegnet_channel_saliency",
    "conformer_attention_summary",
    "evaluate_classifier",
    "evaluate_concept_mapped_transfer",
]


def __getattr__(name: str):
    if name == "LatentSpaceAuditor":
        from voice.evaluation.latent_space import LatentSpaceAuditor

        return LatentSpaceAuditor
    if name in {"eegnet_channel_saliency", "conformer_attention_summary"}:
        from voice.evaluation import interpretability

        return getattr(interpretability, name)
    if name == "evaluate_concept_mapped_transfer":
        from voice.evaluation.transferlearning import evaluate_concept_mapped_transfer

        return evaluate_concept_mapped_transfer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
