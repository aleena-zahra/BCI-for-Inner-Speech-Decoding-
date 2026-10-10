import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Adds sinusoidal positional encoding to sequence embeddings."""

    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # Shape: (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, Seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class ConformerEncoderBlock(nn.Module):
    """Single Conformer block with Multi-Head Self-Attention and Conv Feed-Forward."""

    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        ffn_dim: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-Head Attention Branch
        res = x
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = res + attn_out

        # Feed-Forward Branch
        x = x + self.ffn(self.norm2(x))
        return x


class EEGConformer(nn.Module):
    """EEG-Conformer architecture for EEG classification.

    Combines CNN-based spatial-temporal feature extraction with
    Transformer-based sequence modeling.

    Parameters
    ----------
    n_classes : int, default=2
        Number of output classification classes.
    n_channels : int, default=128
        Number of input EEG channels.
    n_timepoints : int, default=256
        Number of time points per trial.
    d_model : int, default=64
        Embedding dimension for the Conformer sequence blocks.
    n_heads : int, default=4
        Number of heads in Multi-Head Self-Attention.
    n_layers : int, default=2
        Number of stacked Conformer encoder blocks.
    dropout : float, default=0.2
        Dropout rate applied across layers.
    """

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 128,
        n_timepoints: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Temporal & Spatial Convolutional Feature Extractor
        self.conv_spatial = nn.Sequential(
            nn.Conv2d(1, 16, (1, 15), padding=(0, 7), bias=False),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.Conv2d(16, d_model, (n_channels, 1), bias=False),
            nn.BatchNorm2d(d_model),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )

        # Positional Encoding & Conformer Sequence Blocks
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        self.conformer_blocks = nn.ModuleList(
            [
                ConformerEncoderBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    ffn_dim=d_model * 2,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )

        # Dense Classifier Head
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels, Time) or (Batch, 1, Channels, Time)
        if x.ndim == 3:
            x = x.unsqueeze(1)

        # CNN Feature Extraction -> (Batch, d_model, 1, Time_subsampled)
        x = self.conv_spatial(x)
        x = x.squeeze(2).transpose(1, 2)  # Reshape to (Batch, Sequence_length, d_model)

        # Conformer Sequence Modeling
        x = self.pos_encoder(x)
        for block in self.conformer_blocks:
            x = block(x)

        # Global Average Pooling along sequence time dimension
        latent = x.mean(dim=1)
        logits = self.classifier(latent)
        return logits