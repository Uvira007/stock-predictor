"""Training loop and model persistence (load/save with config and stats)."""

import json
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ..config import get_settings, DEFAULT_TICKERS
from ..data import StockSequenceDataset
from .network import StockGRUModel


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    n = 0
    for ticker_id, seq, target in loader:
        ticker_id = ticker_id.to(device)
        seq = seq.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        out = model(ticker_id, seq)
        loss = criterion(out, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    n = 0
    for ticker_id, seq, target in loader:
        ticker_id = ticker_id.to(device)
        seq = seq.to(device)
        target = target.to(device)
        out = model(ticker_id, seq)
        loss = criterion(out, target)
        total_loss += loss.item()
        n += 1
    return total_loss / max(n, 1)


def train_model(
    tickers: List[str] | None = None,
    seq_len: int | None = None,
    predict_days: int | None = None,
    hidden_size: int | None = None,
    num_gru_layers: int | None = None,
    embedding_dim: int | None = None,
    dropout: float | None = None,
    batch_size: int | None = None,
    epochs: int | None = None,
    lr: float | None = None,
    train_split: float | None = None,
    device: torch.device | None = None,
    models_dir: Path | None = None,
    period: str | None = None,
) -> Dict[str, Any]:
    """
    Build dataset, train StockGRUModel, save .pt + config + stats.
    Returns dict with train/val losses and paths.
    """
    settings = get_settings()
    tickers = tickers or settings.tickers
    seq_len = seq_len or settings.seq_len
    predict_days = predict_days or settings.predict_days
    hidden_size = hidden_size or settings.hidden_size
    num_gru_layers = num_gru_layers or settings.num_gru_layers
    embedding_dim = embedding_dim or settings.embedding_dim
    dropout = dropout or settings.dropout
    batch_size = batch_size or getattr(settings, "batch_size", 32)
    epochs = epochs or getattr(settings, "epochs", 50)
    lr = lr or getattr(settings, "learning_rate", 1e-3)
    train_split = train_split or getattr(settings, "train_split", 0.85)
    models_dir = models_dir or settings.models_dir
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dataset = StockSequenceDataset(
        tickers=tickers,
        seq_len=seq_len,
        predict_days=predict_days,
        period=period or settings.yf_period,
    )
    if len(dataset) == 0:
        raise ValueError("No samples after loading data. Check tickers and period.")

    n = len(dataset)
    n_train = int(n * train_split)
    n_val = n - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    num_features = len(dataset.feature_cols)
    model = StockGRUModel(
        num_tickers=len(tickers),
        num_features=num_features,
        seq_len=seq_len,
        predict_days=predict_days,
        hidden_size=hidden_size,
        num_gru_layers=num_gru_layers,
        embedding_dim=embedding_dim,
        dropout=dropout,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep+1}/{epochs}  train_loss={tl:.6f}  val_loss={vl:.6f}")

    # Persist: model state, config, normalization stats
    config = {
        "tickers": tickers,
        "seq_len": seq_len,
        "predict_days": predict_days,
        "num_features": num_features,
        "hidden_size": hidden_size,
        "num_gru_layers": num_gru_layers,
        "embedding_dim": embedding_dim,
        "dropout": dropout,
    }
    stats = dataset.get_stats()

    model_path = models_dir / "model.pt"
    config_path = models_dir / "config.json"
    stats_path = models_dir / "normalize_stats.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        model_path,
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return {
        "model_path": str(model_path),
        "config_path": str(config_path),
        "stats_path": str(stats_path),
        "history": history,
        "tickers": tickers,
    }

def finetune_model(
        period: str = "6mo",
        batch_size: int | None = None,
        epochs: int | None = None,
        lr: float | None = None,
        train_split: float | None = None,
        device: torch.device | None = None,
        models_dir: Path | None = None,
) -> Dict[str, Any]:
    """
    Load existing model, freeze ticker_ember and gru weights, fine-tune only fc on recent data.
    Uses saved normalize_stats so inputs match the base model. Saves updated weights.
    Returns dict with paths and history
    """
    settings = get_settings()
    models_dir = Path(models_dir or settings.models_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = batch_size or getattr(settings, "batch_size", 32)
    epochs = epochs or 15
    lr = lr or 1e-4
    train_split = train_split or getattr(settings, "train_split", .85)

    if not (models_dir / "model.pt").exists():
        raise FileNotFoundError("No existing model.pt found. Train a model first")
    
    model, config, stats = load_model(models_dir=models_dir, device=device)
    model.train() # set the model to training mode

    # Freeze ticker_embed and gru; train only fc
    for p in model.ticker_embed.parameters():
        p.requires_grad = False
    for p in model.gru.parameters():
        p.requires_grad = False

    tickers = config["tickers"]
    seq_len = config["seq_len"]
    predict_days = config["predict_days"]

    dataset = StockSequenceDataset(
        tickers=tickers,
        seq_len=seq_len,
        predict_days=predict_days,
        period=period,
        normalize_stats=stats,
    )
    if len(dataset) == 0:
        raise ValueError(
            f"No samples for period {period}. Need enough data for seq_len={seq_len} + predict_days={predict_days}."
                         )
    
    n = len(dataset)
    n_train = int(n * train_split)
    n_val = n- n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader[Any](train_ds, batch_size=batch_size, shuffle=True,
                                   num_workers=0)
    val_loader = DataLoader[Any](val_ds, batch_size=batch_size, shuffle=False,
                                 num_workers=0)
    
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {"train_loss": [], "val_loss": []}
    for ep in range(epochs):
        tl = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl = validate(model, val_loader, criterion, device)
        history["train_loss"].append(tl)
        history["val_loss"].append(vl)
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"Finetune Epoch {ep+1}/{epochs}  train loss={tl:.6f} val loss={vl:.6f}")

    model_path = models_dir / "model.pt"
    config_path = models_dir / "config.json"
    stats_path = models_dir / "normalize_stats.json"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
        },
        model_path
    )
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return{
        "model_path": str(model_path),
        "config_path": str(config_path),
        "stats_path": str(stats_path),
        "history": history,
        "tickers": tickers,
        "period": period
    }



def load_model(
    models_dir: Path | str | None = None,
    device: torch.device | None = None,
) -> tuple[StockGRUModel, dict, dict]:
    """
    Load trained model, config, and normalize stats.
    Returns (model, config, stats).
    """
    settings = get_settings()
    models_dir = Path(models_dir or settings.models_dir)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(models_dir / "model.pt", map_location=device, weights_only=True)
    with open(models_dir / "config.json") as f:
        config = json.load(f)
    with open(models_dir / "normalize_stats.json") as f:
        stats = json.load(f)

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
    model.to(device)
    model.eval()
    return model, config, stats
