"""
Centralized trajectory CSV filename helpers.

Padding width is controlled by config `naming.trajectory_index_padding`.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

# Matches trajectory_<digits>.csv or trajectory_<digits>_labeled.csv (any padding width)
_TRAJECTORY_RAW_RE = re.compile(r"^trajectory_(\d+)\.csv$")
_TRAJECTORY_LABELED_RE = re.compile(r"^trajectory_(\d+)_labeled\.csv$")


def get_trajectory_index_padding(config: Dict[str, Any], default: int = 5) -> int:
    """Single source of truth accessor with safe fallback."""
    padding = config.get("naming", {}).get("trajectory_index_padding", default)
    return max(1, int(padding))


def trajectory_filename(
    idx: int,
    padding: int = 5,
    labeled: bool = False,
    suffix: str = ".csv",
) -> str:
    """
    Build a trajectory CSV filename with zero-padded index.

    Examples (padding=3):
        trajectory_filename(7) -> trajectory_007.csv
        trajectory_filename(7, labeled=True) -> trajectory_007_labeled.csv
    """
    if idx < 0:
        raise ValueError(f"trajectory index must be non-negative, got {idx}")
    core = f"trajectory_{idx:0{padding}d}"
    if labeled:
        core += "_labeled"
    return core + suffix


def parse_trajectory_index(filename: str) -> Optional[int]:
    """
    Extract numeric index from a trajectory filename.

    Accepts any digit width (backward compatible with 3- or 5-digit files).
    """
    name = Path(filename).name
    m = _TRAJECTORY_LABELED_RE.match(name) or _TRAJECTORY_RAW_RE.match(name)
    return int(m.group(1)) if m else None


def list_trajectory_files(
    directory: Path,
    labeled: bool = False,
) -> List[Path]:
    """
    List trajectory CSV paths sorted by numeric index (not string order).

    Raw: trajectory_*.csv excluding *_labeled.csv
    Labeled: trajectory_*_labeled.csv
    """
    directory = Path(directory)
    pattern = _TRAJECTORY_LABELED_RE if labeled else _TRAJECTORY_RAW_RE
    files: List[Path] = []
    for path in directory.iterdir():
        if not path.is_file():
            continue
        if labeled:
            if pattern.match(path.name):
                files.append(path)
        else:
            if _TRAJECTORY_RAW_RE.match(path.name):
                files.append(path)
    return sorted(files, key=lambda p: int(pattern.match(p.name).group(1)))
