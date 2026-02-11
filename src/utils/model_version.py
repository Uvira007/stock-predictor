"""
Model version metadata: Read/write model_version.json in the models directory.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional


# File name for storing the model version metadata (models/model_version.json)
MODEL_VERSION_FILENAME = "model_version.json"

def write_model_version(models_dir: Path, updated_by: str) -> None:
    """
    Write or update model version.json with najor.minor version (incrementing),
    - train/retrain: bump major version, set minor to 0 (e.g. 1.0 -> 2.0)
    - finetune: bump minor version only (e.g. 1.0 -> 1.1, 1.1 -> 1.2) 
    last_updated (ISO UTC),
    updated_by should be one of: "train", "retrain", "finetune"
    """
    path = Path(models_dir) / MODEL_VERSION_FILENAME
    major, minor = 1, 0
    if path.exists():
        try:
            data = json.loads(path.read_text())
            major = int(data.get("version_major", 1))
            minor = int(data.get("version_minor", 0))
        except(json.JSONDecodeError, ValueError) as e:
            print(f"Unable to parse json file: {str(e)}")
    if updated_by in ("train", "retrain"):
        if path.exists():
            major += 1
        minor = 0
    elif updated_by == "finetune":
        minor += 1
    data = {
        "version_major": major,
        "version_minor": minor,
        "version": f"{major}.{minor}",
        "last_updated": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_by": updated_by,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent = 2))


def get_model_version(models_dir: Optional[Path]=None) -> Optional[Dict[str, Any]]:
    """
    Read model_version.json if present.
    Return dict with version, last_updated, updated_by or None
    """
    if models_dir is None:
        from ..config import get_settings
        models_dir = Path(get_settings().models_dir)
    path = models_dir / MODEL_VERSION_FILENAME
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        # Normalize: if old format had integer "version", expose as major.minor
        if "version_major" not in data and "version" in data:
            v = data["version"]
            if isinstance(v, int):
                data["version_major"] = v
                data["version_minor"] = 0
                data["version"] = f"{v}.0"
        return data
    except (json.JSONDecodeError, OSError):
        return None
