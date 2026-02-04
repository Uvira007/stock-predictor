"""CLI: train the GRU model and save to models/."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model.train import train_model

if __name__ == "__main__":
    train_model(
        period = "5y",
        epochs=50,
        batch_size=32,
    )
    print("Training complete. Model saved under stocks-predictor/models/")