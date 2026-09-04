"""
Step 4: Deep Learning Decoding Architectures & Training Engine
Implements the core benchmark models for inner speech decoding from literature:

1. EEGNet (Lawhern et al., 2018): Lightweight convolutional baseline (~8.54K params) with
   temporal, depthwise spatial, and pointwise separable convolutions.
2. EEGConformer (Song et al., 2022 / Zhou et al., 2025): Hybrid architecture combining
   Convolutional Patch Embedding with Multi-Head Self-Attention (MHSA) blocks.
3. EEGTrainer: Flexible PyTorch training engine with train/val splits, AdamW optimizer,
   learning rate scheduler, and real-time metric reporting for Streamlit.
"""

import math
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix


# ==============================================================================
# PyTorch EEG Dataset
# ==============================================================================
class EEGDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, subj: np.ndarray = None):
        """
        X: (N_trials, C_channels, T_timepoints)
        y: (N_trials,) class integers
        subj: (N_trials,) subject integers
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        self.subj = torch.tensor(subj if subj is not None else np.zeros(len(y)), dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.subj[idx]


# ==============================================================================
# 1. EEGNet Architecture (Lawhern et al., 2018)
# ==============================================================================
class Conv2dWithConstraint(nn.Conv2d):
    """Conv2d with maximum norm constraint on weights for regularization."""
    def __init__(self, *args, max_norm=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = max_norm

    def forward(self, x):
        if self.max_norm is not None:
            self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class EEGNet(nn.Module):
    """
    Standard EEGNet architecture (~8.54K parameters).
    Input Shape: (Batch, 1, Channels, Timepoints) or (Batch, Channels, Timepoints)
    """
    def __init__(
        self, 
        n_classes: int = 8, 
        n_channels: int = 72, 
        n_timepoints: int = 501,
        F1: int = 8, 
        D: int = 2, 
        F2: int = 16, 
        kernel_len: int = 64, 
        dropout_rate: float = 0.25
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.n_timepoints = n_timepoints
        self.F1 = F1
        self.D = D
        self.F2 = F2

        # Block 1: Temporal Conv -> Depthwise Spatial Conv
        self.conv1 = nn.Conv2d(1, F1, (1, kernel_len), padding=(0, kernel_len // 2), bias=False)
        self.bn1 = nn.BatchNorm2d(F1)
        self.depthwise = Conv2dWithConstraint(
            F1, F1 * D, (n_channels, 1), groups=F1, bias=False, max_norm=1.0
        )
        self.bn2 = nn.BatchNorm2d(F1 * D)
        self.pool1 = nn.AvgPool2d((1, 4))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2: Separable Conv (Depthwise + Pointwise)
        self.separable_depthwise = nn.Conv2d(
            F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False
        )
        self.separable_pointwise = nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d((1, 8))
        self.drop2 = nn.Dropout(dropout_rate)

        # Calculate flattened dimension dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_channels, n_timepoints)
            out = self._forward_features(dummy)
            self.flat_dim = out.view(1, -1).shape[1]

        # Classification Dense Head
        self.classifier = nn.Linear(self.flat_dim, n_classes)

    def _forward_features(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)
        
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depthwise(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.separable_depthwise(x)
        x = self.separable_pointwise(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)
        return x

    def extract_features(self, x):
        """Extracts penultimate latent representation vector for Stage 5 auditing."""
        feat_map = self._forward_features(x)
        latent_vector = feat_map.view(feat_map.size(0), -1)
        return latent_vector

    def forward(self, x):
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits


# ==============================================================================
# 2. EEG-Conformer Architecture (Song et al., 2022 / Zhou et al., 2025)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x):
        # x: (Batch, Seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ConformerEncoderBlock(nn.Module):
    """Single Conformer block with Multi-Head Self-Attention and Conv Feed-Forward."""
    def __init__(self, d_model: int = 64, n_heads: int = 4, ffn_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, return_attn: bool = False):
        # Multi-Head Attention Sublayer
        norm_x = self.norm1(x)
        attn_out, attn_weights = self.mha(norm_x, norm_x, norm_x, need_weights=True)
        x = x + self.drop1(attn_out)

        # Feed-Forward Sublayer
        x = x + self.ffn(self.norm2(x))

        if return_attn:
            return x, attn_weights
        return x


class EEGConformer(nn.Module):
    """
    EEG-Conformer combining Convolutional Temporal Patch Embedding with MHSA.
    Input Shape: (Batch, Channels, Timepoints)
    """
    def __init__(
        self, 
        n_classes: int = 8, 
        n_channels: int = 72, 
        n_timepoints: int = 501,
        d_model: int = 64, 
        n_heads: int = 4, 
        n_layers: int = 3, 
        patch_len: int = 25, 
        stride: int = 6, 
        dropout: float = 0.2
    ):
        super().__init__()
        self.n_classes = n_classes
        self.n_channels = n_channels
        self.d_model = d_model
        self.patch_len = patch_len
        self.stride = stride

        # 1. Temporal Patch Projection Layer
        self.patch_conv = nn.Conv2d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=(n_channels, patch_len),
            stride=(1, stride),
            bias=True
        )
        self.pos_encoder = PositionalEncoding(d_model=d_model, max_len=500)
        self.embed_drop = nn.Dropout(dropout)

        # 2. Conformer MHSA Transformer Encoder Blocks
        self.blocks = nn.ModuleList([
            ConformerEncoderBlock(d_model=d_model, n_heads=n_heads, ffn_dim=d_model * 2, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm_final = nn.LayerNorm(d_model)

        # 3. Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

    def extract_features(self, x, return_attns: bool = False):
        """
        Embeds input and returns latent representation vector + attention matrices.
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)  # (B, 1, C, T)

        # Conv Patch Embedding: (B, d_model, 1, N_patches) -> (B, N_patches, d_model)
        feat = self.patch_conv(x).squeeze(2).permute(0, 2, 1)
        feat = self.pos_encoder(feat)
        feat = self.embed_drop(feat)

        attns = []
        for block in self.blocks:
            if return_attns:
                feat, attn_w = block(feat, return_attn=True)
                attns.append(attn_w)
            else:
                feat = block(feat, return_attn=False)

        feat = self.norm_final(feat)
        # Global temporal average pooling to obtain trial latent vector: (B, d_model)
        latent_vector = torch.mean(feat, dim=1)

        if return_attns:
            return latent_vector, attns
        return latent_vector

    def forward(self, x):
        latent = self.extract_features(x)
        logits = self.classifier(latent)
        return logits


# ==============================================================================
# 3. PyTorch EEG Training & Evaluation Engine
# ==============================================================================
class EEGTrainer:
    """
    Manages end-to-end training, validation, metric evaluation, and model checkpoints.
    """
    def __init__(
        self, 
        model: nn.Module, 
        device: str = None, 
        lr: float = 1e-3, 
        weight_decay: float = 1e-2, 
        batch_size: int = 16
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = None

    def train_epoch(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []

        for X_batch, y_batch, _ in dataloader:
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(X_batch)
            loss = self.criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping for stable convergence
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0)
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(y_batch.detach().cpu().numpy())

        if self.scheduler:
            self.scheduler.step()

        epoch_loss = total_loss / len(dataloader.dataset)
        epoch_acc = accuracy_score(all_targets, all_preds)
        return epoch_loss, epoch_acc

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for X_batch, y_batch, _ in dataloader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                logits = self.model(X_batch)
                loss = self.criterion(logits, y_batch)

                total_loss += loss.item() * len(y_batch)
                probs = F.softmax(logits, dim=1).detach().cpu().numpy()
                preds = np.argmax(probs, axis=1)

                all_preds.extend(preds)
                all_targets.extend(y_batch.detach().cpu().numpy())
                all_probs.extend(probs)

        val_loss = total_loss / len(dataloader.dataset)
        val_acc = accuracy_score(all_targets, all_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted', zero_division=0)
        cm = confusion_matrix(all_targets, all_preds)

        return {
            "loss": val_loss,
            "accuracy": val_acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "confusion_matrix": cm,
            "predictions": np.array(all_preds),
            "targets": np.array(all_targets),
            "probabilities": np.array(all_probs)
        }

    def fit(
        self, 
        X: np.ndarray, 
        y: np.ndarray, 
        subj: np.ndarray = None, 
        epochs: int = 30, 
        val_split: float = 0.2, 
        progress_callback = None
    ):
        """
        Executes full training with Stratified split and real-time epoch telemetry.
        """
        # Stratified train/val split
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_split, random_state=42)
        train_idx, val_idx = next(sss.split(X, y))

        train_ds = EEGDataset(X[train_idx], y[train_idx], subj[train_idx] if subj is not None else None)
        val_ds = EEGDataset(X[val_idx], y[val_idx], subj[val_idx] if subj is not None else None)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, shuffle=False)

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs, eta_min=1e-5)

        history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": []
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self.train_epoch(train_loader)
            eval_res = self.evaluate(val_loader)
            val_loss, val_acc = eval_res["loss"], eval_res["accuracy"]
            elapsed = time.time() - t0

            history["train_loss"].append(tr_loss)
            history["train_acc"].append(tr_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            if progress_callback is not None:
                progress_callback(epoch, epochs, tr_loss, tr_acc, val_loss, val_acc, elapsed)

        # Final evaluation on full validation set
        final_metrics = self.evaluate(val_loader)
        final_metrics["history"] = history
        final_metrics["val_indices"] = val_idx
        return final_metrics
