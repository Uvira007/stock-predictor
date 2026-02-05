"""Run prediction for a ticker using latest Yahoo data and saved model."""

from pathlib import Path
from typing import List

import numpy as np
import torch

from ..config import get_settings
from ..data.fetch import fetch_ohlcv
from ..data.preprocess import normalize_ohlcv
from .train import load_model
from .network import StockGRUModel


def predict(
    ticker: str,
    n_days: int | None = None,
    models_dir: Path | str | None = None,
    period: str | None = None,
) -> dict:
    """
    Fetch latest OHLCV for ticker, run model, return predictions (denormalized).
    n_days: how many days to predict (must be <= model's predict_days).
    Returns dict with keys: ticker, predictions (list of floats), last_date, model_predict_days.
    """
    settings = get_settings()
    models_dir = Path(models_dir or settings.models_dir)
    period = period or settings.yf_period

    model, config, stats = load_model(models_dir=models_dir)
    device = next(model.parameters()).device
    tickers = config["tickers"]
    if ticker not in tickers:
        raise ValueError(
            f"Ticker '{ticker}' not in trained tickers: {tickers}. Retrain with this ticker or choose from {tickers}."
        )
    ticker_id = tickers.index(ticker)
    seq_len = config["seq_len"]
    predict_days = config["predict_days"]
    n_days = n_days or predict_days
    if n_days > predict_days:
        n_days = predict_days

    raw = fetch_ohlcv([ticker], period=period)
    if not raw or ticker not in raw:
        raise ValueError(f"No data returned for ticker {ticker}.")
    df = raw[ticker]
    ticker_stats = stats.get(ticker)
    if not ticker_stats:
        raise ValueError(f"No normalization stats for ticker {ticker}. Retrain model.")
    arr, _ = normalize_ohlcv(df, settings.feature_cols, stats=ticker_stats)
    if len(arr) < seq_len:
        raise ValueError(
            f"Not enough history for {ticker}: need {seq_len} days, got {len(arr)}."
        )
    # Last seq_len days
    seq = arr[-seq_len:].astype(np.float32)
    # Close index for denormalization
    close_idx = 3
    mean_close = np.array(ticker_stats["mean"], dtype=np.float32)[close_idx]
    std_close = np.array(ticker_stats["std"], dtype=np.float32)[close_idx]

    x = torch.tensor(seq[np.newaxis, ...], dtype=torch.float32, device=device)
    tid = torch.tensor([ticker_id], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model(tid, x)
    pred_norm = out[0].cpu().numpy()[:n_days]
    pred_price = pred_norm * std_close + mean_close
    # If we log-scaled volume, close was not log-scaled, so denorm is correct
    last_date = str(df.index[-1].date()) if hasattr(df.index[-1], "date") else str(df.index[-1])

    return {
        "ticker": ticker,
        "predictions": pred_price.tolist(),
        "last_date": last_date,
        "model_predict_days": predict_days,
        "n_days_returned": n_days,
    }
