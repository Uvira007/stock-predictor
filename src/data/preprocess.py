"""
Normalize OHLCV data and build (sequence, target) for GRU
"""

from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]

def normalize_ohlcv(df: pd.DataFrame,
                    feature_cols: List[str] | None = None,
                    stats: Dict | None = None) -> Tuple[np.ndarray, dict]:
    """Z-score normalization for OHLCV, log-scale Volume
    Returns (array shape [T, F], stats dict for inverse later)
    """
    feature_cols = feature_cols or FEATURE_COLS

    # Log volume to reduce scale
    if "Volume" in df.columns:
        df["Volume"] = np.log1p(df["Volume"].values)

    arr = df.values.astype(np.float32)
    if stats is None:
        mean = arr.mean(axis = 0)
        std = arr.std(axis = 0)
        std[std == 0] = 1.0
        stats = {"mean": mean.tolist(), "std": std.tolist()}
    else:
        mean = np.array(stats["mean"], dtype=np.float32)
        std = np.array(stats["std"], dtype = np.float32)

    arr = (arr - mean) / std
    return arr, stats

def build_sequences(
        arr: np.ndarray,
        seq_len: int,
        predict_days: int,
        target_col_idx: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build sequences of shape (N, seq_len, Feature) and targets (N, predict_days)
    target_col_idx=3 is Close
    """
    T, F = arr.shape
    if T < seq_len + predict_days:
        return np.empty((0, seq_len, F)), np.empty((0, predict_days))
    
    X_list, y_list = [], []
    for i in range(T - seq_len - predict_days + 1):
        X_list.append(arr[i : i+ seq_len])
        y_list.append(arr[i + seq_len : i + seq_len + predict_days, target_col_idx])
    return np.stack(X_list), np.stack(y_list)