# Stock GRU Predictor

End-to-end stock price predictor using a **GRU built from scratch** (no `nn.GRU`), multivariate OHLCV inputs, and a single model for all tickers (ticker embedding). Data from **Yahoo Finance** (free). Export to **PyTorch** and **ONNX** for use outside this repo.

## Features

- **GRU from scratch**: Custom `GRUCell` and multi-layer `GRU` in `src/model/gru.py`
- **Multivariate**: Open, High, Low, Close, Volume (normalized; log-scale volume)
- **Single model for all tickers**: Ticker embedding; one training run, one checkpoint
- **Next N days**: Predict next 1–21 days closing price (configurable)
- **Isolated backend**: FastAPI API; Streamlit UI calls the API
- **Retrain option**: Optional retrain with latest data (use cautiously; can take several minutes)
- **Export**: PyTorch (`.pt` + `config.json` + `normalize_stats.json`) and ONNX

## Project structure

```
stock_predictor/
  src/
    config/       # Tickers, hyperparameters, paths
    data/         # yfinance fetch, normalize, sequence dataset
    model/        # GRU from scratch, StockGRUModel, train, inference
    api/          # FastAPI: predict, retrain, tickers
    export/       # PyTorch bundle + ONNX export
  app/
    streamlit_app.py
  scripts/
    train_model.py   # CLI: train and save
    export_model.py  # CLI: export ONNX + bundle
    run_api.py       # Run FastAPI server
  models/         # Saved model.pt, config.json, normalize_stats.json, model.onnx (created)
  requirements.txt
```

## Setup

From the `stock_predictor` directory:

```bash
cd stock_predictor
python -m venv .venv
.venv\Scripts\activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

## Run end-to-end

### 1. Train the model (once)

```bash
python scripts/train_model.py
```

This downloads ~5 years of OHLCV for default tickers (AAPL, MSFT, GOOGL, …), trains the GRU, and saves to `models/`:
- `model.pt` — PyTorch state dict + config
- `config.json` — Tickers, seq_len, predict_days, etc.
- `normalize_stats.json` — Per-ticker mean/std for inference

### 2. Start the backend API

```bash
python scripts/run_api.py
```

API runs at `http://127.0.0.1:8000`. Endpoints:
- `GET /tickers` — List supported tickers
- `POST /predict` — Body: `{"ticker": "AAPL", "n_days": 5}` → predictions
- `POST /retrain` — Start retrain in background (use cautiously)
- `GET /model/status` — Whether a model is loaded

### 3. Start the Streamlit UI

In a second terminal (same venv):

```bash
streamlit run app/streamlit_app.py
```

Select ticker, choose “Predict next N days” (1 = next-day), click **Predict**. Optionally use **Retrain** in the sidebar (can take several minutes).

## Export model for use outside this repo

```bash
python scripts/export_model.py
```

This writes/keeps in `models/`:
- `model.pt`, `config.json`, `normalize_stats.json` (PyTorch bundle)
- `model.onnx` (ONNX)

### Loading the PyTorch model elsewhere

Copy the `models/` folder (or the three files above) to your project. You need the same model definition. Minimal example:

```python
import json
import torch
from pathlib import Path

# Copy StockGRUModel definition (from src/model/network.py and gru.py) or install this package
from src.model.network import StockGRUModel

models_dir = Path("path/to/models")
with open(models_dir / "config.json") as f:
    config = json.load(f)
ckpt = torch.load(models_dir / "model.pt", map_location="cpu", weights_only=True)
model = StockGRUModel(
    num_tickers=len(config["tickers"]),
    num_features=config["num_features"],
    seq_len=config["seq_len"],
    predict_days=config["predict_days"],
    hidden_size=config["hidden_size"],
    num_gru_layers=config["num_gru_layers"],
    embedding_dim=config["embedding_dim"],
    dropout=config["dropout"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
# Use normalize_stats.json for input normalization when building sequences
```

### Using the ONNX model

```python
import onnxruntime as rt
import numpy as np

session = rt.InferenceSession("models/model.onnx")
ticker_id = np.array([0], dtype=np.int64)  # 0 = first ticker in config
sequence = np.random.randn(1, 60, 5).astype(np.float32)  # (1, seq_len, 5)
out = session.run(["predictions"], {"ticker_id": ticker_id, "sequence": sequence})
# out[0].shape = (1, predict_days)
```

Inputs must be normalized the same way (use `normalize_stats.json` for the chosen ticker).

## Configuration

Edit `src/config/settings.py` to change:
- `DEFAULT_TICKERS` — Tickers used for training
- `SEQ_LEN` — Lookback days (default 60)
- `PREDICT_DAYS` — Next N days (default 21)
- `YF_PERIOD` — Yahoo history period (default `"5y"`)

## Data

Uses [yfinance](https://pypi.org/project/yfinance/) (free). No API key. For research/education; not for live trading.
