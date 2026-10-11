"""Shared, deterministic fold assignments for training and ICA review."""

import numpy as np
from sklearn.model_selection import train_test_split


def evaluation_folds(labels, subjects, *, strategy="trial", deep=False,
                     test_size=0.2, validation_size=0.15, seed=42):
    y, subject_ids = np.asarray(labels), np.asarray(subjects)
    if y.ndim != 1 or len(y) != len(subject_ids) or len(np.unique(y)) < 2:
        raise ValueError("Labels/subjects must be aligned and contain at least two classes")
    indices = np.arange(len(y))
    if strategy == "trial":
        if np.unique(y, return_counts=True)[1].min() < 2:
            raise ValueError("Trial holdout needs at least two examples of every class")
        trainval, test = train_test_split(indices, test_size=test_size, random_state=seed, stratify=y)
        raw_folds = [("trial holdout", trainval, test)]
    elif strategy == "loso":
        if len(np.unique(subject_ids)) < 2:
            raise ValueError("LOSO evaluation requires at least two subjects")
        raw_folds = [(str(subject), indices[subject_ids != subject], indices[subject_ids == subject])
                     for subject in np.unique(subject_ids)]
    else:
        raise ValueError("strategy must be 'trial' or 'loso'")
    folds = []
    for number, (name, trainval, test) in enumerate(raw_folds):
        if len(np.unique(y[trainval])) < 2:
            raise ValueError(f"Training data without subject {name} has fewer than two classes")
        if deep:
            train, validation = train_test_split(trainval, test_size=validation_size,
                                                random_state=seed + number, stratify=y[trainval])
        else:
            train, validation = trainval, np.array([], dtype=int)
        folds.append((name, train, validation, test))
    return folds
