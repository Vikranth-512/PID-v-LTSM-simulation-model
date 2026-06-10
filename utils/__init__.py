"""
Shared utilities: config loading, reproducibility, experiment metadata, naming.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml

from utils.naming import (
    get_trajectory_index_padding,
    list_trajectory_files,
    parse_trajectory_index,
    trajectory_filename,
)

__all__ = [
    "load_config",
    "set_seed",
    "ensure_dirs",
    "save_experiment_metadata",
    "get_trajectory_index_padding",
    "list_trajectory_files",
    "parse_trajectory_index",
    "trajectory_filename",
]


def load_config(path: Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path) as f:
        if path.suffix in (".yaml", ".yml"):
            return yaml.safe_load(f)
        return json.load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(config: Dict[str, Any]) -> Dict[str, Path]:
    paths = {}
    for key, rel in config.get("paths", {}).items():
        p = Path(rel)
        p.mkdir(parents=True, exist_ok=True)
        paths[key] = p
    return paths


def save_experiment_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    extra: Dict[str, Any] | None = None,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config,
        **(extra or {}),
    }
    path = output_dir / "experiment_metadata.json"
    with open(path, "w") as f:
        json.dump(meta, f, indent=2, default=str)
    return path
