"""
Step 5: Cross-Lingual Evaluation, Latent Space Dynamics & Interpretability
Follows the explainable BCI and latent auditing framework from Chris Bras (2024) and LaRocco et al. (2023).

Key Capabilities:
1. LatentSpaceAuditor: Extracts bottleneck feature embeddings from trained EEGNet and Conformer encoders.
   Computes PCA, t-SNE, and UMAP 2D projections.
2. ClusteringMetrics: Computes Silhouette Scores for linguistic target words vs. Subject Identity confounds
   (validating whether representations encode inner speech vs. speaker identity bias).
3. CrossLingualEvaluator: Zero-shot & fine-tuned transfer matrix between languages.
4. AttentionVisualizer: Multi-Head Self-Attention heatmap and Broca's area (F3/F7) saliency mapping.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.metrics import silhouette_score, pairwise_distances


class LatentSpaceAuditor:
    """
    Extracts high-dimensional latent vectors from trained models and projects them
    into 2D embedding space via PCA, t-SNE, and UMAP.
    """
    def __init__(self, model: nn.Module, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def extract_embeddings(self, X: np.ndarray, batch_size: int = 32):
        """
        Extracts penultimate bottleneck feature representations for all trials.
        X shape: (N_trials, C_channels, T_timepoints)
        Returns Z: (N_trials, D_features)
        """
        all_embeddings = []
        with torch.no_grad():
            for i in range(0, len(X), batch_size):
                batch = torch.tensor(X[i:i+batch_size], dtype=torch.float32).to(self.device)
                if hasattr(self.model, "extract_features"):
                    emb = self.model.extract_features(batch)
                    if isinstance(emb, tuple):
                        emb = emb[0]  # If returns (latent, attns)
                else:
                    # Fallback to direct forward if no explicit extract_features
                    emb = self.model(batch)
                
                all_embeddings.append(emb.detach().cpu().numpy())

        Z = np.vstack(all_embeddings)
        return Z

    def compute_projections(self, Z: np.ndarray, perplexity: int = 15, n_neighbors: int = 15):
        """
        Computes PCA, t-SNE, and UMAP 2D projections for the latent embeddings Z.
        """
        # 1. PCA
        pca = PCA(n_components=2, random_state=42)
        z_pca = pca.fit_transform(Z)
        pca_var_ratio = pca.explained_variance_ratio_

        # 2. t-SNE
        effective_perp = min(perplexity, max(2, len(Z) // 4))
        tsne = TSNE(n_components=2, perplexity=effective_perp, random_state=42, init="pca", learning_rate="auto")
        z_tsne = tsne.fit_transform(Z)

        # 3. UMAP
        effective_nn = min(n_neighbors, max(2, len(Z) - 1))
        reducer = umap.UMAP(n_components=2, n_neighbors=effective_nn, min_dist=0.1, random_state=42)
        z_umap = reducer.fit_transform(Z)

        return {
            "pca": z_pca,
            "pca_var_ratio": pca_var_ratio,
            "tsne": z_tsne,
            "umap": z_umap,
            "raw_embeddings": Z
        }


class ClusteringMetrics:
    """
    Computes quantitative cluster quality metrics to audit whether the model learned
    universal linguistic speech imagery or is overfitted to speaker identity.
    """
    @staticmethod
    def compute_silhouette(Z: np.ndarray, labels: np.ndarray):
        """
        Calculates Silhouette Score.
        Returns value between -1.0 and +1.0. Higher is better separation.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2 or len(Z) <= len(unique_labels):
            return 0.0
        try:
            return float(silhouette_score(Z, labels))
        except Exception:
            return 0.0

    @staticmethod
    def compute_intra_inter_ratio(Z: np.ndarray, labels: np.ndarray):
        """
        Calculates ratio of mean Intra-class distance to mean Inter-class distance.
        Lower ratio indicates tighter, more discriminative clusters.
        """
        unique_labels = np.unique(labels)
        if len(unique_labels) < 2:
            return 1.0

        dists = pairwise_distances(Z, metric="euclidean")
        intra_dists = []
        inter_dists = []

        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if labels[i] == labels[j]:
                    intra_dists.append(dists[i, j])
                else:
                    inter_dists.append(dists[i, j])

        mean_intra = np.mean(intra_dists) if intra_dists else 1.0
        mean_inter = np.mean(inter_dists) if inter_dists else 1.0

        return float(mean_intra / (mean_inter + 1e-8))


class CrossLingualEvaluator:
    """
    Evaluates zero-shot cross-lingual transfer and joint multi-language representations.
    """
    @staticmethod
    def evaluate_transfer(model: nn.Module, X_target: np.ndarray, y_target: np.ndarray, device: str = None):
        """
        Tests a model trained on Source Language (e.g. English) against Target Language (e.g. Spanish).
        """
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        with torch.no_grad():
            X_tensor = torch.tensor(X_target, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_target, dtype=torch.long)

            logits = model(X_tensor)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            targets = y_tensor.numpy()

            # For zero-shot cross-lingual, calculate Top-1 accuracy if class counts match
            acc = float(np.mean(preds == targets))

        return {
            "transfer_accuracy": acc,
            "predictions": preds,
            "targets": targets
        }
