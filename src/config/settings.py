"""
Configuration: tickers, hyperparameters, paths.
"""

from pathlib import Path
from typing import List

# default tickers for training (single model learns all)
DEFAULT_TICKERS: list[str] = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "META",
    "NVDA",
    "TSLA",
    "JPM",
    "V",
    "JNJ",
]

# Paths realtive to stock-predictor/
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Model training defaults
SEQ_LEN = 60 # lookback days
PREDICT_DAYS = 21 # Next N days (use 1 for next-day only)
FEATURE_COLS = ["Open","High", "Low", "Close", "Volume"]
HIDDEN_SIZE = 64
NUM_GRU_LAYERS = 2
EMBEDDING_DIM = 16
DROPOUT = 0.2
TRAIN_SPLIT = 0.85
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 1e-3

# Yahoo finanace
YF_PERIOD = "5y" # 5 years history for training
YF_INERVAL = "1d"

class Settings:
    """
    Runtime settings (can be overridden by env or API).
    """
    def __init__(self,
                 tickers: list[str] | None = None,
                 seq_len: int = SEQ_LEN,
                 predict_days: int = PREDICT_DAYS,
                 hidden_size: int = HIDDEN_SIZE,
                 epochs: int = EPOCHS,
                 batch_size: int = BATCH_SIZE,
                 num_gru_layers: int = NUM_GRU_LAYERS,
                 embedding_dim: int = EMBEDDING_DIM,
                 dropout: float = DROPOUT,
                 models_dir: Path | None = None,
                 data_dir: Path | None = None,
                 ):
        self.tickers = tickers or DEFAULT_TICKERS.copy()
        self.seq_len = seq_len
        self.predict_days = predict_days
        self.hidden_size = hidden_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_gru_layers = num_gru_layers
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.models_dir = models_dir or MODELS_DIR
        self.data_dir = data_dir or DATA_DIR
        self.learning_rate = LEARNING_RATE
        self.train_split = TRAIN_SPLIT
        self.yf_period = YF_PERIOD
        self.yf_interval = YF_INERVAL
        self.feature_cols = FEATURE_COLS.copy()

    @property
    def num_tickers(self) -> int:
        return len(self.tickers)
    
    @property
    def ticker_to_idx(self) -> dict:
        return {t:i for i, t in enumerate[str](self.tickers)}

    @property
    def num_features(self) -> int:
        return len(self.feature_cols)

def get_settings() -> Settings:    
    return Settings()

