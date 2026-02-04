"""Export trained model to PyTorch (.pt) and ONNX for use outside this repo."""

import json
from pathlib import Path
from typing import Optional

import torch

from ..config import get_settings
from ..model.train import load_model
from ..model.network import StockGRUModel


def export_pytorch_bundle(
    models_dir: Path | str | None = None,
    output_dir: Path | str | None = None,
) -> dict:
    """
    Copy/save model.pt, config.json, normalize_stats.json to output_dir
    so another repo can load with load_model(output_dir).
    Returns paths written.
    """
    settings = get_settings()
    models_dir = Path(models_dir or settings.models_dir)
    output_dir = Path(output_dir or models_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for name in ["model.pt", "config.json", "normalize_stats.json"]:
        src = models_dir / name
        if not src.exists():
            raise FileNotFoundError(f"Missing {src}. Train the model first.")
        dst = output_dir / name
        if src.resolve() != dst.resolve():
            import shutil
            shutil.copy2(src, dst)

    return {
        "model_pt": str(output_dir / "model.pt"),
        "config_json": str(output_dir / "config.json"),
        "normalize_stats_json": str(output_dir / "normalize_stats.json"),
    }


def export_onnx(
    models_dir: Path | str | None = None,
    output_path: Path | str | None = None,
    batch_size: int = 1,
    opset_version: int = 14,
) -> str:
    """
    Export loaded model to ONNX with fixed batch and seq_len.
    Inputs: ticker_id (B,), sequence (B, seq_len, num_features).
    Output: (B, predict_days).
    """
    settings = get_settings()
    models_dir = Path(models_dir or settings.models_dir)
    model, config, _ = load_model(models_dir=models_dir)
    model.eval()

    seq_len = config["seq_len"]
    num_features = config["num_features"]
    num_tickers = config["num_tickers"]

    ticker_id = torch.randint(0, num_tickers, (batch_size,))
    sequence = torch.randn(batch_size, seq_len, num_features)

    out_path = output_path or (models_dir / "model.onnx")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (ticker_id, sequence),
        str(out_path),
        input_names=["ticker_id", "sequence"],
        output_names=["predictions"],
        dynamic_axes={
            "ticker_id": {0: "batch"},
            "sequence": {0: "batch"},
            "predictions": {0: "batch"},
        },
        opset_version=opset_version,
    )
    return str(out_path)
