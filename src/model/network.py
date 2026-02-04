"""Full model: ticker embedding + GRU (from scratch) + head for next N days."""

from typing import Optional

import torch
import torch.nn as nn

from .gru import GRU


class StockGRUModel(nn.Module):
    """
    Multivariate multi-ticker GRU predictor.
    Input: ticker_id (B,), sequence (B, T, F).
    Output: (B, predict_days) — next N closing prices (normalized).
    """

    def __init__(
        self,
        num_tickers: int,
        num_features: int,
        seq_len: int,
        predict_days: int,
        hidden_size: int = 64,
        num_gru_layers: int = 2,
        embedding_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_tickers = num_tickers
        self.num_features = num_features
        self.seq_len = seq_len
        self.predict_days = predict_days
        self.hidden_size = hidden_size

        self.ticker_embed = nn.Embedding(num_tickers, embedding_dim)
        # GRU input: raw features (no concat embed per step to keep seq clean; we concat embed to final hidden)
        self.gru = GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            dropout=dropout,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size + embedding_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, predict_days),
        )

    def forward(
        self,
        ticker_id: torch.Tensor,
        sequence: torch.Tensor,
    ) -> torch.Tensor:
        # ticker_id: (B,), sequence: (B, T, F)
        B = sequence.size(0)
        embed = self.ticker_embed(ticker_id)  # (B, emb)
        out, h_last = self.gru(sequence)  # out (B, T, H), h_last (num_layers, B, H)
        h = h_last[-1]  # (B, H)
        h = torch.cat([h, embed], dim=1)  # (B, H+emb)
        return self.fc(h)  # (B, predict_days)
