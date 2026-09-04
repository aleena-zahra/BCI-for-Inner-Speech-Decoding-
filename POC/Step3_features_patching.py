"""
Step 3: Feature Extraction, Temporal Patching & Representation Learning
Follows the tokenization and feature representation methodology from Zhou et al. (2025) and Nieto et al. (2022).

Key Features:
1. EEGPatchTokenizer: Slices (N_trials, C_channels, T_timepoints) into overlapping temporal patches
   (e.g., P=25 samples [100ms] with stride S=6 samples [24ms] at 250Hz).
2. SpectralFeatureExtractor: Computes Band Power across Delta (0.5-4Hz), Theta (4-8Hz), Alpha (8-13Hz),
   Beta (13-30Hz), and Gamma (30-100Hz), along with spectrograms and PSD.
3. RevIN: Reversible Instance Normalization to eliminate non-stationarity across sessions.
4. SubjectEmbedding: Learnable/per-subject scale embeddings for inter-subject alignment.
"""

import numpy as np
import scipy.signal
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


class EEGPatchTokenizer:
    """
    Slices multi-channel EEG continuous/epoch tensors into overlapping temporal tokens.
    
    As defined in Zhou et al. (2025) (LBLM):
    - Input Shape: (N_trials, C_channels, T_timepoints)
    - Patch Length (P): e.g. 25 samples (100 ms at 250 Hz)
    - Stride (S): e.g. 6 samples (24 ms at 250 Hz)
    - Output Shape: (N_trials, N_patches, C_channels, P) or flattened to (N_trials, N_patches, C_channels * P)
    """
    def __init__(self, patch_len=25, stride=6, flatten_channels=True):
        self.patch_len = patch_len
        self.stride = stride
        self.flatten_channels = flatten_channels

    def tokenize(self, X: np.ndarray):
        """
        Extracts overlapping patches from an EEG array of shape (N, C, T).
        """
        if isinstance(X, torch.Tensor):
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)

        if X_np.ndim != 3:
            raise ValueError(f"Expected 3D EEG array of shape (N_trials, C_channels, T_timepoints), got {X_np.shape}")

        N, C, T = X_np.shape
        if T < self.patch_len:
            raise ValueError(f"Time dimension ({T}) is smaller than patch length ({self.patch_len})")

        # Number of overlapping patches
        n_patches = (T - self.patch_len) // self.stride + 1

        patches = []
        for i in range(n_patches):
            start = i * self.stride
            end = start + self.patch_len
            patch = X_np[:, :, start:end]  # Shape: (N, C, P)
            patches.append(patch)

        # Stack across patch dimension: (N, N_patches, C, P)
        patches_tensor = np.stack(patches, axis=1)

        if self.flatten_channels:
            # Flatten to (N, N_patches, C * P)
            patches_tensor = patches_tensor.reshape(N, n_patches, C * self.patch_len)

        return patches_tensor, n_patches

    def get_token_metadata(self, total_timepoints: int, sfreq: float = 250.0):
        """Returns metadata about patch timing and token dimensions."""
        n_patches = (total_timepoints - self.patch_len) // self.stride + 1
        patch_duration_ms = (self.patch_len / sfreq) * 1000.0
        stride_duration_ms = (self.stride / sfreq) * 1000.0
        
        centers_s = [(i * self.stride + self.patch_len / 2.0) / sfreq for i in range(n_patches)]
        
        return {
            "n_patches": n_patches,
            "patch_len_samples": self.patch_len,
            "stride_samples": self.stride,
            "patch_duration_ms": patch_duration_ms,
            "stride_duration_ms": stride_duration_ms,
            "patch_centers_s": centers_s
        }


class SpectralFeatureExtractor:
    """
    Extracts frequency-band power and time-frequency representations with special focus
    on Gamma (30-100 Hz) phonemic speech imagery activations and Delta/Theta rhythm coupling.
    """
    BANDS = {
        "Delta": (0.5, 4.0),
        "Theta": (4.0, 8.0),
        "Alpha": (8.0, 13.0),
        "Beta": (13.0, 30.0),
        "Gamma": (30.0, 100.0)
    }

    def __init__(self, sfreq=250.0):
        self.sfreq = sfreq

    def compute_band_powers(self, X: np.ndarray):
        """
        Computes average band power for each frequency band.
        Input X shape: (N_trials, C_channels, T_timepoints)
        Output dict: {band_name: (N_trials, C_channels)}
        """
        N, C, T = X.shape
        band_powers = {}

        # Compute Welch PSD
        freqs, psd = scipy.signal.welch(X, fs=self.sfreq, nperseg=min(T, 128), axis=-1)
        # psd shape: (N, C, n_freqs)

        for band_name, (fmin, fmax) in self.BANDS.items():
            idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            if len(idx) > 0:
                band_power = np.mean(psd[:, :, idx], axis=-1)  # (N, C)
            else:
                band_power = np.zeros((N, C))
            band_powers[band_name] = band_power

        return band_powers, freqs, psd

    def compute_trial_spectrogram(self, trial_signal: np.ndarray, nperseg=32, noverlap=28):
        """
        Computes Short-Time Fourier Transform (STFT) spectrogram for a single channel trial.
        trial_signal: 1D array of shape (T,)
        """
        f, t, Zxx = scipy.signal.stft(trial_signal, fs=self.sfreq, nperseg=nperseg, noverlap=noverlap)
        spectrogram = np.abs(Zxx)
        return f, t, spectrogram


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (RevIN) as used in modern EEG sequence architectures.
    Eliminates non-stationarity by standardizing input sequences while retaining learnable affine parameters.
    """
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(num_features))
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x: torch.Tensor, mode: str = "norm"):
        """
        x shape: (Batch, Channels, Time) or (Batch, Time, Dim)
        mode: 'norm' to normalize, 'denorm' to reverse.
        """
        if mode == "norm":
            mean = torch.mean(x, dim=-1, keepdim=True)
            std = torch.std(x, dim=-1, keepdim=True) + self.eps
            self.mean = mean
            self.std = std
            x_norm = (x - mean) / std
            if self.affine:
                if x.ndim == 3 and x.shape[1] == self.num_features:
                    x_norm = x_norm * self.gamma.view(1, -1, 1) + self.beta.view(1, -1, 1)
                elif x.ndim == 3 and x.shape[2] == self.num_features:
                    x_norm = x_norm * self.gamma.view(1, 1, -1) + self.beta.view(1, 1, -1)
            return x_norm
        elif mode == "denorm":
            if self.affine:
                if x.ndim == 3 and x.shape[1] == self.num_features:
                    x = (x - self.beta.view(1, -1, 1)) / (self.gamma.view(1, -1, 1) + self.eps)
                elif x.ndim == 3 and x.shape[2] == self.num_features:
                    x = (x - self.beta.view(1, 1, -1)) / (self.gamma.view(1, 1, -1) + self.eps)
            return x * self.std + self.mean
        else:
            raise NotImplementedError(f"RevIN mode '{mode}' not supported.")
