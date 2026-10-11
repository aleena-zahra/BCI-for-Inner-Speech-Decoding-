"""Reusable PyTorch training and held-out evaluation for Voice deep models."""

from __future__ import annotations

from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


class EEGTrainer:
    """Train one torch classifier with an explicit validation set."""

    def __init__(
        self,
        model,
        *,
        device: str | None = None,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-2,
        batch_size: int = 32,
        seed: int = 42,
    ):
        try:
            import torch
        except ImportError as exc:
            raise ImportError("Deep models require PyTorch. Install with: pip install -e '.[deep-learning]'") from exc
        self.torch = torch
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.seed = seed
        self.history: list[dict[str, float]] = []

    def _loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool):
        torch = self.torch
        dataset = torch.utils.data.TensorDataset(
            torch.as_tensor(np.asarray(X, dtype=np.float32)),
            torch.as_tensor(np.asarray(y, dtype=np.int64)),
        )
        generator = torch.Generator().manual_seed(self.seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=min(self.batch_size, max(1, len(dataset))),
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=0,
        )

    def _run_epoch(self, loader, optimizer=None):
        torch = self.torch
        training = optimizer is not None
        self.model.train(training)
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)
        total_loss, targets, predictions = 0.0, [], []
        context = torch.enable_grad() if training else torch.no_grad()
        with context:
            for signals, labels in loader:
                signals, labels = signals.to(self.device), labels.to(self.device)
                if training:
                    optimizer.zero_grad(set_to_none=True)
                logits = self.model(signals)
                loss = criterion(logits, labels)
                if training:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
                    optimizer.step()
                total_loss += float(loss.detach()) * len(labels)
                targets.extend(labels.detach().cpu().numpy())
                predictions.extend(logits.argmax(dim=1).detach().cpu().numpy())
        return total_loss / max(1, len(loader.dataset)), float(accuracy_score(targets, predictions))

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_validation: np.ndarray,
        y_validation: np.ndarray,
        *,
        epochs: int = 10,
        progress_callback: Callable[[int, int, dict[str, float]], None] | None = None,
    ) -> list[dict[str, float]]:
        if epochs < 1 or not len(X_train) or not len(X_validation):
            raise ValueError("Training and validation sets must be non-empty and epochs positive")
        if X_train.ndim != 3 or X_validation.ndim != 3:
            raise ValueError("Deep models need (trials, channels, time) arrays")
        if X_train.shape[1:] != X_validation.shape[1:]:
            raise ValueError("Training and validation signal shapes must match")
        torch = self.torch
        train_loader = self._loader(X_train, y_train, shuffle=True)
        validation_loader = self._loader(X_validation, y_validation, shuffle=False)
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
        self.history = []
        for epoch in range(1, epochs + 1):
            train_loss, train_accuracy = self._run_epoch(train_loader, optimizer)
            validation_loss, validation_accuracy = self._run_epoch(validation_loader)
            scheduler.step()
            record = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
            self.history.append(record)
            if progress_callback is not None:
                progress_callback(epoch, epochs, record)
        return self.history

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        torch = self.torch
        self.model.eval()
        loader = self._loader(X, np.zeros(len(X), dtype=np.int64), shuffle=False)
        logits_all = []
        with torch.no_grad():
            for signals, _ in loader:
                logits_all.append(self.model(signals.to(self.device)).cpu().numpy())
        logits = np.concatenate(logits_all)
        shifted = logits - logits.max(axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
        return logits.argmax(axis=1), probabilities


def _new_model(
    name: str, n_classes: int, n_channels: int, n_timepoints: int,
    patch_len: int, stride: int, use_revin: bool,
):
    if name == "EEGNet":
        from voice.models.eegnet import EEGNet

        return EEGNet(
            n_classes=n_classes, n_channels=n_channels, n_timepoints=n_timepoints,
            use_revin=use_revin,
        )
    if name == "EEGConformer":
        from voice.models.eegconformer import EEGConformer

        return EEGConformer(
            n_classes=n_classes,
            n_channels=n_channels,
            n_timepoints=n_timepoints,
            patch_len=patch_len,
            stride=stride,
            use_revin=use_revin,
        )
    raise ValueError(f"Unsupported deep model: {name}")


def _class_metrics(y_true, y_pred, labels):
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels),
    }


def evaluate_deep_model(
    signals: np.ndarray,
    labels: np.ndarray,
    subjects: np.ndarray,
    *,
    model_name: str,
    strategy: str = "trial",
    epochs: int = 10,
    batch_size: int = 32,
    patch_len: int = 25,
    stride: int = 6,
    use_revin: bool = False,
    test_size: float = 0.2,
    validation_size: float = 0.15,
    seed: int = 42,
    progress_callback: Callable[[str, int, int, dict[str, float]], None] | None = None,
) -> dict:
    """Train one or more folds and return scores plus fold-aligned latents.

    ``strategy='loso'`` holds out each complete subject. ``strategy='trial'``
    makes a stratified trial holdout. Labels are encoded internally to 0..K-1;
    ``classes`` retains their original IDs.
    """
    X, y, subject_ids = np.asarray(signals), np.asarray(labels), np.asarray(subjects)
    if X.ndim != 3 or len(X) != len(y) or len(y) != len(subject_ids):
        raise ValueError("Signals, labels, and subjects must be aligned trial arrays")
    classes = np.unique(y)
    if len(classes) < 2:
        raise ValueError("At least two classes are required to train a deep model")
    label_to_index = {label: index for index, label in enumerate(classes.tolist())}
    encoded = np.asarray([label_to_index[label] for label in y], dtype=np.int64)
    all_indices = np.arange(len(y))
    if strategy == "trial":
        counts = np.unique(encoded, return_counts=True)[1]
        if counts.min() < 2:
            raise ValueError("Trial holdout needs at least two examples of every class")
        trainval, test = train_test_split(
            all_indices, test_size=test_size, random_state=seed, stratify=encoded
        )
        folds = [("trial holdout", trainval, test)]
    elif strategy == "loso":
        if len(np.unique(subject_ids)) < 2:
            raise ValueError("LOSO evaluation requires at least two subjects")
        folds = []
        for subject in np.unique(subject_ids):
            test = np.flatnonzero(subject_ids == subject)
            trainval = np.flatnonzero(subject_ids != subject)
            if len(np.unique(encoded[trainval])) < 2:
                raise ValueError(f"Training data without subject {subject} has fewer than two classes")
            folds.append((str(subject), trainval, test))
    else:
        raise ValueError("strategy must be 'trial' or 'loso'")

    models, histories, fold_latents = {}, {}, {}
    true_parts, pred_parts, subject_parts = [], [], []
    for fold_number, (fold_name, trainval, test) in enumerate(folds):
        train, validation = train_test_split(
            trainval,
            test_size=validation_size,
            random_state=seed + fold_number,
            stratify=encoded[trainval],
        )
        torch = __import__("torch")
        torch.manual_seed(seed + fold_number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + fold_number)
        model = _new_model(
            model_name, len(classes), X.shape[1], X.shape[2],
            patch_len, stride, use_revin,
        )
        trainer = EEGTrainer(model, batch_size=batch_size, seed=seed + fold_number)

        def report(epoch, total_epochs, record, current_fold=fold_name):
            if progress_callback:
                progress_callback(current_fold, epoch, total_epochs, record)

        trainer.fit(
            X[train], encoded[train], X[validation], encoded[validation],
            epochs=epochs, progress_callback=report,
        )
        predictions, probabilities = trainer.predict(X[test])
        fold_latents[fold_name] = extract_model_embeddings(model, X[test], device=str(trainer.device))
        models[fold_name] = model.to("cpu")
        histories[fold_name] = trainer.history
        true_parts.append(encoded[test])
        pred_parts.append(predictions)
        subject_parts.append(subject_ids[test])

    y_true_encoded = np.concatenate(true_parts)
    y_pred_encoded = np.concatenate(pred_parts)
    results = _class_metrics(y_true_encoded, y_pred_encoded, np.arange(len(classes)))
    per_subject = {}
    combined_subjects = np.concatenate(subject_parts)
    for subject in np.unique(combined_subjects):
        mask = combined_subjects == subject
        per_subject[str(subject)] = float(accuracy_score(y_true_encoded[mask], y_pred_encoded[mask]))
    results.update({
        "classes": classes,
        "y_true": classes[y_true_encoded],
        "y_pred": classes[y_pred_encoded],
        "subjects": combined_subjects,
        "per_subject_accuracy": per_subject,
        "fold_latents": fold_latents,
        "models": models,
        "histories": histories,
        "strategy": strategy,
        "model_name": model_name,
    })
    if strategy == "trial":
        results["features"] = fold_latents["trial holdout"]
        results["model"] = models["trial holdout"]
        results["history"] = histories["trial holdout"]
    return results


def extract_model_embeddings(model, signals: np.ndarray, *, batch_size: int = 64, device: str | None = None):
    """Extract the penultimate model representation in bounded-memory batches."""
    import torch

    selected_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = model.to(selected_device)
    model.eval()
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(signals), batch_size):
            batch = torch.as_tensor(np.asarray(signals[start : start + batch_size], dtype=np.float32)).to(selected_device)
            embeddings.append(model.extract_features(batch).detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)
