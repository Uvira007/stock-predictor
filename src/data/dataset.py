"""Pytorch Datset for (ticker_id, sequence, target).
"""

from typing import List, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import get_settings
from .fetch import fetch_ohlcv
from .preprocess import normalize_ohlcv, build_sequences

class StockSequenceDataset(Dataset):
    """
    Multiavriate sequence dataset: (ticker_id, X, y)
    X: (seq_len, num_features), y: (predict_days)
    """

    def __init__(self,
                 tickers: List[str],
                 seq_len: int | None = None,
                 predict_days: int | None = None,
                 period: str | None = None,  
                 normalize_stats: Dict[str, Dict] | None = None,
    ):
        settings = get_settings()
        self.seq_len = seq_len or settings.seq_len
        self.predict_days = predict_days or settings.predict_days
        self.period = period or settings.yf_period
        self.tickers = tickers
        self.ticker_to_idx = (
            settings.ticker_to_idx if tickers == settings.tickers
            else {t : i for i, t in enumerate[str](tickers)}
        )
        self.feature_cols = settings.feature_cols
        raw = fetch_ohlcv(self.tickers, period = self.period)
        self.samples: List[Tuple[int, np.ndarray, np.ndarray]] = []
        self.stats_per_ticker: Dict[str, Dict] = normalize_stats or {}

        for ticker, df in raw.items():
            if ticker not in self.ticker_to_idx:
                continue
            tid = self.ticker_to_idx[ticker]
            stats = self.stats_per_ticker.get(ticker)
            arr, stats = normalize_ohlcv(df, self.feature_cols, stats)
            self.stats_per_ticker[ticker] = stats
            X, y = build_sequences(arr, self.seq_len, self.predict_days,
                                   target_col_idx=3)
            for j in range(len(X)):
                self.samples.append((tid, X[j], y[j]))

    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tid, X, y = self.samples[idx]
        return(
            torch.tensor(tid, dtype=torch.long),
            torch.tensor(X, dtype = torch.float32),
            torch.tensor(y, dtype = torch.float32)
        )

    def __getstats__(self) -> Dict[str, Dict]:
        return self.stats_per_ticker