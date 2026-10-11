
import numpy as np
import torch
from voice.models import EEGNet
from voice.evaluation import LatentSpaceAuditor, evaluate_classifier


def main() -> None:
    """Audit embeddings and predictions from an EEGNet model."""
    model = EEGNet(n_classes=2, n_channels=128, n_timepoints=256)
    signals = np.random.randn(50, 128, 256)
    labels = np.random.choice([0, 1], size=50)

    auditor = LatentSpaceAuditor(model)
    embeddings = auditor.extract_embeddings(signals)
    latent_metrics = auditor.compute_metrics(embeddings, labels)
    print("Latent-space silhouette score:", latent_metrics["silhouette_score"])

    with torch.no_grad():
        logits = model(torch.tensor(signals, dtype=torch.float32))
        predictions = torch.argmax(logits, dim=1).numpy()

    metrics = evaluate_classifier(labels, predictions)
    print("Test accuracy:", metrics["accuracy"])


if __name__ == "__main__":
    main()