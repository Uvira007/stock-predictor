"""CLI: Export trained model to PyTorch bundle and ONNX"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.export import export_onnx, export_pytorch_bundle
from src.config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    models_dir = Path(settings.models_dir)
    if not (models_dir / "model.pt").exists():
        print("Mo model.pt found. Train first: python scripts/train_model.py")
        sys.exit(1)
    export_pytorch_bundle(models_dir=models_dir, output_dir=models_dir)
    onnx_path = export_onnx(models_dir=models_dir)
    print(f"PyTorch bundle: {models_dir}")
    print(f"ONNX: {onnx_path}")