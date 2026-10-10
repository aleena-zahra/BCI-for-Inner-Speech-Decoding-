import numpy as np
import torch
from voice.models import EEGNet
from voice.evaluation import LatentSpaceAuditor, evaluate_classifier

# Load or initialize model
model = EEGNet(n_classes=2, n_channels=128, n_timepoints=256)
model.eval()

# Dummy test set
X_test = np.random.randn(50, 128, 256)
y_test = np.random.choice([0, 1], size=50)

# 1. Audit Latent Space Embeddings
auditor = LatentSpaceAuditor(model)
Z = auditor.extract_embeddings(X_test)
metrics = auditor.compute_metrics(Z, y_test)

print("Latent Space Silhouette Score:", metrics["silhouette_score"])

# 2. Evaluate Performance Metrics
with torch.no_grad():
    logits = model(torch.tensor(X_test, dtype=torch.float32))
    preds = torch.argmax(logits, dim=1).numpy()

results = evaluate_classifier(y_test, preds)
print("Test Accuracy:", results["accuracy"])