"""CLI: Export trained model to PyTorch bundle and ONNX"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export import export_onnx, export_pytorch_bundle
from src.config import get_settings

if __name__ == "__main__":
    train_model(
        period = "5y",
        epochs=50,
        batch_size=32,
    )
    print("Training complete. Model saved under stocks-predictor/models/")