"""Batched model embeddings, low-dimensional projections, and cluster metrics."""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score


class LatentSpaceAuditor:
    """Audit representations from a trained Voice model without eager torch/UMAP imports."""

    def __init__(self, model, device: str | None = None):
        self.model = model
        self.device = device

    def extract_embeddings(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        from voice.models.training import extract_model_embeddings

        return extract_model_embeddings(
            self.model, X, batch_size=batch_size, device=self.device
        )

    @staticmethod
    def compute_projections(
        Z: np.ndarray,
        *,
        methods: tuple[str, ...] = ("PCA", "t-SNE", "UMAP"),
        perplexity: float = 15,
        n_neighbors: int = 15,
        random_state: int = 42,
    ) -> dict[str, np.ndarray | list[float]]:
        Z = np.asarray(Z)
        if Z.ndim != 2 or len(Z) < 3 or not np.isfinite(Z).all():
            raise ValueError("Projection requires at least three finite embedding rows")
        projections = {}
        for method in methods:
            name = method.lower()
            if name == "pca":
                model = PCA(n_components=2, random_state=random_state)
                projections["pca"] = model.fit_transform(Z)
                projections["pca_var_ratio"] = model.explained_variance_ratio_.tolist()
            elif name in {"t-sne", "tsne"}:
                effective_perplexity = min(float(perplexity), max(1.0, len(Z) - 1.0))
                projections["tsne"] = TSNE(
                    n_components=2,
                    perplexity=effective_perplexity,
                    init="pca",
                    learning_rate="auto",
                    random_state=random_state,
                ).fit_transform(Z)
            elif name == "umap":
                try:
                    import umap
                except ImportError as exc:
                    raise ImportError("UMAP requires the optional dependency: pip install 'voice[deep-learning]'") from exc
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(n_neighbors, len(Z) - 1),
                    min_dist=0.1,
                    random_state=random_state,
                )
                projections["umap"] = reducer.fit_transform(Z)
            else:
                raise ValueError(f"Unknown projection method: {method}")
        projections["raw_embeddings"] = Z
        return projections

    @staticmethod
    def compute_metrics(Z: np.ndarray, y: np.ndarray) -> dict:
        Z, y = np.asarray(Z), np.asarray(y)
        if Z.ndim != 2 or y.ndim != 1 or not len(Z) or len(Z) != len(y) or not np.isfinite(Z).all():
            raise ValueError("Embeddings and labels must be finite and row-aligned")
        unique = np.unique(y)
        silhouette = None
        if 1 < len(unique) < len(y):
            sample_size = min(2000, len(y))
            silhouette = float(silhouette_score(
                Z, y, sample_size=sample_size if sample_size < len(y) else None,
                random_state=42,
            ))
        # Bound pairwise memory for large datasets, deterministically.
        sampled = np.random.default_rng(42).choice(len(Z), min(2000, len(Z)), replace=False)
        distances = pdist(Z[sampled], metric="euclidean")
        encoded_labels = np.unique(y[sampled], return_inverse=True)[1]
        pair_labels = pdist(encoded_labels.reshape(-1, 1), metric="hamming") == 0
        within = distances[pair_labels]
        between = distances[~pair_labels]
        ratio = float(within.mean() / (between.mean() + 1e-12)) if len(within) and len(between) else None
        return {
            "silhouette_score": silhouette,
            "mean_pairwise_distance": float(distances.mean()) if len(distances) else 0.0,
            "intra_inter_distance_ratio": ratio,
            "distance_sample_count": len(sampled),
        }

    @staticmethod
    def audit(Z, labels, subjects, *, method="PCA", perplexity=15, n_neighbors=15):
        """Combine the POC's word/subject audits without mixing separately fitted folds."""
        Z, labels, subjects = np.asarray(Z), np.asarray(labels), np.asarray(subjects)
        if subjects.ndim != 1 or len(subjects) != len(Z):
            raise ValueError("Subject identifiers must align with embedding rows")
        return {
            "projections": LatentSpaceAuditor.compute_projections(
                Z, methods=(method,), perplexity=perplexity, n_neighbors=n_neighbors
            ),
            "class_metrics": LatentSpaceAuditor.compute_metrics(Z, labels),
            "subject_metrics": LatentSpaceAuditor.compute_metrics(Z, subjects),
        }


class ClusteringMetrics:
    """POC-compatible metric entry points backed by Voice's shared implementation."""

    @staticmethod
    def compute_silhouette(Z, labels):
        return LatentSpaceAuditor.compute_metrics(Z, labels)["silhouette_score"]

    @staticmethod
    def compute_intra_inter_ratio(Z, labels):
        return LatentSpaceAuditor.compute_metrics(Z, labels)["intra_inter_distance_ratio"]
