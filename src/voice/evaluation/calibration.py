"""Uncertainty Calibration and Risk-Sensitive Medical Safety Gating.

Author: Maimoona Hanan (23i-2544)
Project: V.O.I.C.E (FAST-NUCES FYP 2026-2027)

Module 3: Latent Auditing, Calibration, and Safety Deployment.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class TemperatureScaler(nn.Module):
    """Calibrates neural network logits via temperature scaling:

    p_i = softmax(z_i / T)
    """

    def __init__(self, initial_temp: float = 1.5):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * initial_temp)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.clamp(self.temperature, min=0.01)
        return logits / temp

    def fit(self, val_logits: torch.Tensor, val_labels: torch.Tensor, max_iter: int = 50) -> float:
        """Fits optimal temperature T on validation set using Negative Log-Likelihood."""
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)

        def eval_loss():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(val_logits), val_labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        return float(self.temperature.item())


class SafetyCalibrator:
    """Manages calibrated inference, confidence thresholding, and ECE calculation."""

    def __init__(self, threshold: float = 0.65, initial_temp: float = 1.0):
        self.threshold = float(threshold)
        self.scaler = TemperatureScaler(initial_temp=initial_temp)

    def compute_ece(
        self, 
        probs: np.ndarray, 
        labels: np.ndarray, 
        n_bins: int = 10
    ) -> Dict[str, Union[float, List[float]]]:
        """Calculates Expected Calibration Error (ECE) across confidence bins."""
        confidences = np.max(probs, axis=1)
        predictions = np.argmax(probs, axis=1)
        accuracies = (predictions == labels).astype(float)

        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        ece = 0.0
        mce = 0.0
        bin_accs, bin_confs, bin_counts = [], [], []

        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]

            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = np.mean(in_bin)

            if prop_in_bin > 0:
                acc_in_bin = float(np.mean(accuracies[in_bin]))
                conf_in_bin = float(np.mean(confidences[in_bin]))
                gap = abs(acc_in_bin - conf_in_bin)

                ece += gap * prop_in_bin
                mce = max(mce, gap)

                bin_accs.append(acc_in_bin)
                bin_confs.append(conf_in_bin)
                bin_counts.append(int(np.sum(in_bin)))
            else:
                bin_accs.append(0.0)
                bin_confs.append(float((bin_lower + bin_upper) / 2))
                bin_counts.append(0)

        return {
            "ece": float(ece),
            "mce": float(mce),
            "bin_accuracies": bin_accs,
            "bin_confidences": bin_confs,
            "bin_counts": bin_counts,
        }

    def predict_with_safety_rejection(
        self,
        logits: Union[np.ndarray, torch.Tensor],
        class_names: Optional[List[str]] = None
    ) -> Tuple[Optional[str], float, bool]:
        """Executes calibrated prediction with risk-sensitive safety abstention."""
        if isinstance(logits, np.ndarray):
            logits_tensor = torch.tensor(logits, dtype=torch.float32)
        else:
            logits_tensor = logits.clone().detach()

        if logits_tensor.ndim == 1:
            logits_tensor = logits_tensor.unsqueeze(0)

        with torch.no_grad():
            scaled_logits = self.scaler(logits_tensor)
            probs = torch.softmax(scaled_logits, dim=-1).squeeze(0).cpu().numpy()

        predicted_idx = int(np.argmax(probs))
        confidence = float(probs[predicted_idx])

        # Safety Rejection Logic: Abstain if confidence < tau
        is_accepted = bool(confidence >= self.threshold)

        if is_accepted:
            if class_names and predicted_idx < len(class_names):
                label = class_names[predicted_idx]
            else:
                label = str(predicted_idx)
        else:
            label = None  # Rejection / Abstain

        return label, confidence, is_accepted
