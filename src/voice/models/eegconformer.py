"""Convolutional patch encoder with lightweight Conformer blocks."""

from __future__ import annotations

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        scale = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        encoding = torch.zeros(max_len, d_model)
        encoding[:, 0::2] = torch.sin(position * scale)
        encoding[:, 1::2] = torch.cos(position * scale[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] > self.encoding.shape[1]:
            raise ValueError("Input contains more temporal patches than positional encoding supports")
        return x + self.encoding[:, : x.shape[1]]


class ConformerEncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = self.norm1(x)
        attended, _ = self.attn(normalized, normalized, normalized, need_weights=False)
        x = x + self.drop1(attended)
        return x + self.ffn(self.norm2(x))


class EEGConformer(nn.Module):
    """EEG classifier whose convolution projects each temporal patch to a token."""

    def __init__(
        self,
        n_classes: int = 2,
        n_channels: int = 128,
        n_timepoints: int = 256,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        patch_len: int = 25,
        stride: int = 6,
        dropout: float = 0.2,
    ):
        super().__init__()
        if patch_len < 2 or stride < 1 or patch_len > n_timepoints:
            raise ValueError("Patch length and stride must fit the input time dimension")
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_channels = n_channels
        self.patch_len = patch_len
        self.stride = stride
        self.patch_embedding = nn.Conv2d(
            1, d_model, kernel_size=(n_channels, patch_len), stride=(1, stride)
        )
        n_patches = (n_timepoints - patch_len) // stride + 1
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(1000, n_patches))
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [ConformerEncoderBlock(d_model, n_heads, dropout) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.n_channels:
            raise ValueError("EEGConformer expects (batch, channels, time) EEG tensors")
        tokens = self.patch_embedding(x).squeeze(2).transpose(1, 2)
        tokens = self.embed_dropout(self.pos_encoder(tokens))
        for block in self.blocks:
            tokens = block(tokens)
        return self.norm(tokens).mean(dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.extract_features(x))
