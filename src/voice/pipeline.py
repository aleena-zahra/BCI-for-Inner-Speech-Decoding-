from __future__ import annotations

import copy
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
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
        self.training_features: np.ndarray | None = None

    def fit(self, signals: np.ndarray, labels: np.ndarray) -> "InnerSpeechPipeline":
        """Fit the complete pipeline on trials and labels."""
        processed_signals, labels, _ = self.preprocessor.transform_with_indices(signals, labels)
        features = self.feature_extractor.fit_transform(processed_signals, labels)
        self.classifier.fit(features, labels)
        self.training_features = features
        self.is_fitted = True
        return self

    def evaluate(self, signals: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
        """Preprocess, transform, and evaluate new labeled trials."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before evaluate().")
        if len(signals) != len(labels) or len(signals) == 0:
            raise ValueError("Evaluation signals and labels must be non-empty and aligned")

        processed_signals, labels, _ = self.preprocessor.transform_with_indices(signals, labels)
        features = self.feature_extractor.transform(processed_signals)
        return self.classifier.evaluate(features, labels)

    def transform_features(self, signals: np.ndarray) -> np.ndarray:
        """Return CSP features using only preprocessing and CSP state fitted on training data."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() before transforming features.")
        processed = self.preprocessor.transform_with_indices(signals)[0]
        return self.feature_extractor.transform(processed)

    def _fresh(self) -> "InnerSpeechPipeline":
        """Create an unfitted pipeline with the same configuration for one fold."""
        extractor = MultiBandCSPExtractor(
            n_components=self.feature_extractor.csps[0].n_components,
            sfreq=self.feature_extractor.sfreq,
            bands=self.feature_extractor.bands,
        )
        return InnerSpeechPipeline(
            preprocessor=copy.deepcopy(self.preprocessor),
            feature_extractor=extractor,
            classifier=CSPSVMClassifier(C=self.classifier.C),
            test_size=self.test_size,
            random_state=self.random_state,
        )

    @staticmethod
    def _summarize(y_true, y_pred, subjects, features, classes) -> dict[str, Any]:
        per_subject = {}
        for subject in np.unique(subjects):
            mask = subjects == subject
            per_subject[str(subject)] = float(accuracy_score(y_true[mask], y_pred[mask]))
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=classes),
            "classes": np.asarray(classes),
            "per_subject_accuracy": per_subject,
            "y_true": np.asarray(y_true),
            "y_pred": np.asarray(y_pred),
            "subjects": np.asarray(subjects),
            "features": np.asarray(features),
        }

    def evaluate_trial_holdout(
        self, signals: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> dict[str, Any]:
        """Stratified trial holdout; preprocessing and CSP are refit on train only."""
        signals, labels, subjects = np.asarray(signals), np.asarray(labels), np.asarray(subjects)
        if len(signals) != len(labels) or len(signals) != len(subjects):
            raise ValueError("Signals, labels, and subjects must have matching lengths")
        if len(np.unique(labels)) < 2 or np.min(np.unique(labels, return_counts=True)[1]) < 2:
            raise ValueError("Trial holdout needs at least two examples of every class")
        train_idx, test_idx = train_test_split(
            np.arange(len(labels)), test_size=self.test_size,
            random_state=self.random_state, stratify=labels,
        )
        model = self._fresh().fit(signals[train_idx], labels[train_idx])
        processed, y_test, kept = model.preprocessor.transform_with_indices(
            signals[test_idx], labels[test_idx]
        )
        features = model.feature_extractor.transform(processed)
        metrics = model.classifier.evaluate(features, y_test)
        result = self._summarize(
            y_test, model.classifier.model.predict(features),
            subjects[test_idx][kept], features, np.unique(labels),
        )
        result.update(metrics)
        return result

    def evaluate_leave_one_subject_out(
        self, signals: np.ndarray, labels: np.ndarray, subjects: np.ndarray
    ) -> dict[str, Any]:
        """Evaluate every subject using a model trained on all other subjects."""
        signals, labels, subjects = np.asarray(signals), np.asarray(labels), np.asarray(subjects)
        if len(signals) != len(labels) or len(signals) != len(subjects):
            raise ValueError("Signals, labels, and subjects must have matching lengths")
        unique_subjects = np.unique(subjects)
        if len(unique_subjects) < 2:
            raise ValueError("Leave-one-subject-out evaluation requires at least two subjects")
        all_classes = np.unique(labels)
        true_parts, pred_parts, subject_parts, feature_parts = [], [], [], []
        fold_scores = {}
        for subject in unique_subjects:
            test_mask = subjects == subject
            train_mask = ~test_mask
            if len(np.unique(labels[train_mask])) < 2:
                raise ValueError(f"Training data without subject {subject} has fewer than two classes")
            model = self._fresh().fit(signals[train_mask], labels[train_mask])
            processed, test_labels, kept = model.preprocessor.transform_with_indices(
                signals[test_mask], labels[test_mask]
            )
            test_features = model.feature_extractor.transform(processed)
            prediction = model.classifier.model.predict(test_features)
            fold_scores[str(subject)] = float(accuracy_score(test_labels, prediction))
            true_parts.append(test_labels)
            pred_parts.append(prediction)
            subject_parts.append(subjects[test_mask][kept])
            feature_parts.append(test_features)
        y_true = np.concatenate(true_parts)
        y_pred = np.concatenate(pred_parts)
        return self._summarize(
            y_true, y_pred, np.concatenate(subject_parts),
            np.concatenate(feature_parts), all_classes,
        ) | {"fold_accuracy": fold_scores}

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
