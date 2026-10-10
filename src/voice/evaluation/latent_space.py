import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, pairwise_distances


class LatentSpaceAuditor:
    """Extracts high-dimensional latent vectors from trained models and projects them

    into 2D embedding space via PCA, t-SNE, and UMAP.
    """

    def __init__(self, model: nn.Module, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, X: np.ndarray, batch_size: int = 32) -> np.ndarray:
        """Extracts penultimate bottleneck feature representations for all trials.

        X shape: (N_trials, C_channels, T_timepoints)
        Returns Z: (N_trials, D_features)
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i : i + batch_size], dtype=torch.float32).to(self.device)
                
                # Extract latent features before the classification head
                if hasattr(self.model, "_forward_features"):
                    features = self.model._forward_features(batch)
                elif hasattr(self.model, "conv_spatial"):
                    # For EEGConformer or custom feature extractors
                    features = self.model.conv_spatial(batch)
                else:
                    features = batch

                features = features.view(features.size(0), -1)
                all_embeddings.append(features.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0)

    def compute_metrics(self, Z: np.ndarray, y: np.ndarray) -> dict:
        """Computes silhouette score and pairwise distances on latent embeddings."""
        sil = silhouette_score(Z, y) if len(np.unique(y)) > 1 else None
        dist_matrix = pairwise_distances(Z)
        return {
            "silhouette_score": sil,
            "mean_pairwise_distance": float(np.mean(dist_matrix)),
        }