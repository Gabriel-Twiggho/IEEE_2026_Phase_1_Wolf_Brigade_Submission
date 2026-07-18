from __future__ import annotations

from pathlib import Path


def ensure_debug_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path

