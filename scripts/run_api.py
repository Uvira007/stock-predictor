"""
Run FasAPI backend (Uvicorn)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import uvicorn

from src.api.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port = 8000)