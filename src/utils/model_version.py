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
    Write or update model version.json with version (incrementing), 
    last_updated (ISO UTC),
    updated_by should be one of: "train", "retrain", "finetune"
    """
    path = Path(models_dir) / MODEL_VERSION_FILENAME
    version = 1
    if path.exists():
        try:
            data = json.loads(path.read_text())
            version = int(data.get("version", 0)) + 1
        except(json.JSONDecodeError, ValueError) as e:
            print(f"Unable to parse json file: {str(e)}")
    
    data = {
        "version": version,
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
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
