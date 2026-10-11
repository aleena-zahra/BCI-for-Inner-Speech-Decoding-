import numpy as np

from voice.evaluation.latent_space import LatentSpaceAuditor


def test_latent_projection_and_metrics_accept_string_labels():
    rng = np.random.default_rng(12)
    embeddings = np.vstack([rng.normal(loc=-1, size=(12, 4)), rng.normal(loc=1, size=(12, 4))])
    labels = np.array(["alpha"] * 12 + ["beta"] * 12)
    projections = LatentSpaceAuditor.compute_projections(embeddings, methods=("PCA",))
    metrics = LatentSpaceAuditor.compute_metrics(embeddings, labels)
    assert projections["pca"].shape == (24, 2)
    assert 0 <= metrics["silhouette_score"] <= 1
    assert metrics["intra_inter_distance_ratio"] < 1

