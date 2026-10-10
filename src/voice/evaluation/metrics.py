import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_probs: np.ndarray = None) -> dict:
    """Calculates accuracy, confusion matrix, and optional ROC-AUC score."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    auc = None
    if y_probs is not None and len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)

    return {
        "accuracy": float(acc),
        "confusion_matrix": cm,
        "auc": float(auc) if auc is not None else None,
    }