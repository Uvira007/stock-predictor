"""
Fetch OHLCV data from yahoo finance (Free API)
"""

from typing import List, Dict
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    yf = None

from ..config import get_settings

FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]

def fetch_ohlcv(
        tickers: List[str],
        period: str | None = None,
        interval: str | None = None,
        ) -> Dict[str, pd.DataFrame]:
    """
    Download historical OHLCV data for one or more tickers
    
    :param tickers: List of tickers to fetch ohlcv
    :type tickers: List[str]
    :param period: lookback period for each ticker
    :type period: str | None
    :param interval: Interval to get the historical value
    :type interval: str | None
    :return: DataFrame with DateTimeIndex and columns for each ticker
    :rtype: Dict[str, DataFrame]
    """    
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")
    
    settings = get_settings()
    feature_cols = settings.feature_cols
    period = period or settings.yf_period
    interval = interval or settings.yf_interval

    out = {}
    for ticker in tickers:
        t = yf.Ticker(ticker)
        df = t.history(period = period, interval = interval, auto_adjust = True)
        if df.empty or len(df) < 2:
            continue
        cols = [c for c in feature_cols if c in df.columns]
        if len(cols) != len(feature_cols):
            continue
        df = df[feature_cols].copy()
        df.columns = feature_cols
        df = df.dropna()
        out[ticker] = df
    return out
