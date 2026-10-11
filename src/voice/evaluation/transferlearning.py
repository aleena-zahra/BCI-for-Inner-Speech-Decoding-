"""Explicit semantic-label evaluation for cross-dataset transfer experiments."""

from __future__ import annotations

from typing import Hashable, Mapping, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix


def evaluate_concept_mapped_transfer(
    target_labels: Sequence[Hashable],
    predicted_source_classes: Sequence[Hashable],
    *,
    target_to_concept: Mapping[Hashable, Hashable],
    source_class_to_concept: Mapping[Hashable, Hashable],
) -> dict:
    """Score target trials only after explicit, one-to-one semantic mapping.

    ``target_to_concept`` maps each target dataset's class label to a shared
    concept. ``source_class_to_concept`` maps the source model's output classes
    to the same concept names. No equality between dataset label IDs is assumed.
    """
    target = np.asarray(target_labels, dtype=object)
    predicted = np.asarray(predicted_source_classes, dtype=object)
    if target.ndim != 1 or predicted.ndim != 1 or len(target) != len(predicted) or not len(target):
        raise ValueError("Target labels and source predictions must be non-empty aligned vectors")
    if not target_to_concept or not source_class_to_concept:
        raise ValueError("Explicit target-label and source-class concept mappings are required")
    target_classes = set(target.tolist())
    missing_target = target_classes - set(target_to_concept)
    if missing_target:
        raise ValueError(f"Target labels have no concept mapping: {sorted(map(str, missing_target))}")
    unknown_predictions = set(predicted.tolist()) - set(source_class_to_concept)
    if unknown_predictions:
        raise ValueError(
            "Model predictions have no source concept mapping: "
            f"{sorted(map(str, unknown_predictions))}"
        )
    target_concepts = [target_to_concept[label] for label in target]
    source_concepts = [source_class_to_concept[label] for label in predicted]
    if len(set(target_to_concept[label] for label in target_classes)) != len(target_classes):
        raise ValueError("Target labels must map one-to-one to concepts for this evaluation")
    if len(set(source_class_to_concept.values())) != len(source_class_to_concept):
        raise ValueError("Source model classes must map one-to-one to concepts")
    concepts = sorted(set(target_concepts) | set(source_concepts), key=str)
    matrix = confusion_matrix(target_concepts, source_concepts, labels=concepts)
    per_concept = {
        str(concept): float(matrix[index, index] / matrix[index].sum())
        if matrix[index].sum() else None
        for index, concept in enumerate(concepts)
    }
    return {
        "concept_mapped_accuracy": float(accuracy_score(target_concepts, source_concepts)),
        "concepts": concepts,
        "confusion_matrix": matrix,
        "per_concept_accuracy": per_concept,
        "n_trials": len(target),
        "target_to_concept": dict(target_to_concept),
        "source_class_to_concept": dict(source_class_to_concept),
    }
