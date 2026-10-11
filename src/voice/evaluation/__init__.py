"""Evaluation and interpretability subpackage."""

from voice.evaluation.metrics import evaluate_classifier

__all__ = [
    "LatentSpaceAuditor",
    "ClusteringMetrics",
    "eegnet_channel_saliency",
    "conformer_attention_summary",
    "evaluate_classifier",
]


def __getattr__(name: str):
    if name in {"LatentSpaceAuditor", "ClusteringMetrics"}:
        from voice.evaluation import latent_space

        return getattr(latent_space, name)
    if name in {"eegnet_channel_saliency", "conformer_attention_summary"}:
        from voice.evaluation import interpretability

        return getattr(interpretability, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
