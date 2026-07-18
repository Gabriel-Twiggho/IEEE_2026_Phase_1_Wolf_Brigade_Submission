"""Environment parsing helpers shared by the flat controller modules."""

from __future__ import annotations

import os
from pathlib import Path


CONTROLLER_DIR = Path(__file__).resolve().parent
REPO_ROOT = CONTROLLER_DIR.parents[1]
SIM_LOGS_DIR = CONTROLLER_DIR / "sim_logs"
SLAM_LITE_DIR = CONTROLLER_DIR / "slam_lite"
VICTIM_DIAGNOSTICS_DIR = CONTROLLER_DIR / "victim_diagnostics"
GROUND_VICTIM_MODEL_PATH = CONTROLLER_DIR / "models" / "ground_v2.pt"
LIVE_MAP_VIEWER_PATH = CONTROLLER_DIR / "diagnostics" / "live_map_viewer.py"
DRONE_MAP_IMAGE_PATH = SIM_LOGS_DIR / "map_estimate.png"
DRONE_MAP_INFO_PATH = SIM_LOGS_DIR / "map_estimate_info.json"
DRONE_MAP_VICTIMS_PATH = SIM_LOGS_DIR / "victim_location_estimates.csv"


def env_float(name: str, default: float) -> float:
    text = os.environ.get(name)
    if text is None or text == "":
        return default
    try:
        return float(text)
    except ValueError:
        print(f"[CONFIG] Ignoring invalid {name}={text!r}; using {default}")
        return default


def env_int(name: str, default: int) -> int:
    text = os.environ.get(name)
    if text is None or text == "":
        return default
    try:
        return int(text)
    except ValueError:
        print(f"[CONFIG] Ignoring invalid {name}={text!r}; using {default}")
        return default


def env_flag(name: str, default: bool = False) -> bool:
    text = os.environ.get(name)
    if text is None or text == "":
        return default
    return text.strip().lower() in ("1", "true", "yes", "on")


# Compatibility names used by the extracted modules.
_env_float = env_float
_env_int = env_int
_env_flag = env_flag
