from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import numpy as np
class CSPSVMClassifier:
    """Classical CSP + Linear SVM Baseline Classifier."""

    def __init__(self, C: float = 1.0):
        self.model = Pipeline([
            ('Scaler', StandardScaler()),
            ('SVM', SVC(kernel='linear', C=C, probability=True))
        ])

    def fit(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        y_pred = self.model.predict(X_test)
        y_scores = self.model.decision_function(X_test)
        
        return {
            "accuracy": accuracy_score(y_test, y_pred),
            "auc": roc_auc_score(y_test, y_scores) if len(np.unique(y_test)) == 2 else None,
            "confusion_matrix": confusion_matrix(y_test, y_pred)
        }