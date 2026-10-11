from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split

from voice.evaluation.features import MultiBandCSPExtractor
from voice.models.svm import CSPSVMClassifier
from voice.preprocessing.pipeline import EEGPreprocessor


class InnerSpeechPipeline:
    """High-level preprocessing, feature extraction, and SVM workflow."""

    def __init__(
        self,
        preprocessor: EEGPreprocessor | None = None,
        feature_extractor: MultiBandCSPExtractor | None = None,
        classifier: CSPSVMClassifier | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        self.preprocessor = preprocessor or EEGPreprocessor()
        self.feature_extractor = feature_extractor or MultiBandCSPExtractor()
        self.classifier = classifier or CSPSVMClassifier()
        self.test_size = test_size
        self.random_state = random_state
        self.is_fitted = False

    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "InnerSpeechPipeline":
        """Fit the complete pipeline on trials and labels."""
        processed_signals = self.preprocessor.fit_transform(signals)
        features = self.feature_extractor.fit_transform(processed_signals, labels)
        self.classifier.fit(features, labels)
        self.is_fitted = True
        return self

    def evaluate(self, signals: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Preprocess, transform, and evaluate new labeled trials."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate().")

        processed_signals = self.preprocessor.transform(signals)
        features = self.feature_extractor.transform(processed_signals)
        return self.classifier.evaluate(features, labels)

    def fit_evaluate(self, signals: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Split data, fit the pipeline, and return held-out metrics."""
        train_signals, test_signals, train_labels, test_labels = train_test_split(
            signals,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        self.fit(train_signals, train_labels)
        return self.evaluate(test_signals, test_labels)