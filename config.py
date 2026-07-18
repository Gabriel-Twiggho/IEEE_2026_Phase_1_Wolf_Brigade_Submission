"""Environment parsing helpers shared by the flat controller modules."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile


CONTROLLER_DIR = Path(__file__).resolve().parent


def resolve_runtime_path(
    value: str | Path,
    base: Path = CONTROLLER_DIR,
) -> Path:
    """Resolve a user-supplied runtime path without depending on process CWD."""

    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def _competition_root() -> Path:
    configured = os.environ.get("SAR_COMPETITION_ROOT")
    if configured:
        return resolve_runtime_path(configured)

    # This is the layout used by the official evaluator:
    # <competition-root>/controllers/proposed_solution.
    if CONTROLLER_DIR.parent.name == "controllers":
        return CONTROLLER_DIR.parent.parent.resolve()

    for candidate in CONTROLLER_DIR.parents:
        if (candidate / "controllers").is_dir() and (candidate / "recordings").is_dir():
            return candidate.resolve()

    # Standalone checkouts remain self-contained. A separate recordings folder
    # can be selected with SAR_RECORDINGS_DIR.
    return CONTROLLER_DIR


REPO_ROOT = _competition_root()
RECORDINGS_DIR = resolve_runtime_path(
    os.environ.get("SAR_RECORDINGS_DIR", REPO_ROOT / "recordings"),
    REPO_ROOT,
)
SIM_LOGS_DIR = CONTROLLER_DIR / "sim_logs"
SLAM_LITE_DIR = CONTROLLER_DIR / "slam_lite"
VICTIM_DIAGNOSTICS_DIR = CONTROLLER_DIR / "victim_diagnostics"
GROUND_VICTIM_MODEL_PATH = CONTROLLER_DIR / "models" / "ground_v2.pt"
_live_map_viewer = os.environ.get("SAR_LIVE_MAP_VIEWER_PATH")
LIVE_MAP_VIEWER_PATH = (
    resolve_runtime_path(_live_map_viewer) if _live_map_viewer else None
)
DRONE_MAP_IMAGE_PATH = SIM_LOGS_DIR / "map_estimate.png"
DRONE_MAP_INFO_PATH = SIM_LOGS_DIR / "map_estimate_info.json"
DRONE_MAP_VICTIMS_PATH = SIM_LOGS_DIR / "victim_location_estimates.csv"
RUNTIME_CACHE_DIR = resolve_runtime_path(
    os.environ.get(
        "SAR_RUNTIME_CACHE_DIR",
        Path(tempfile.gettempdir()) / "wolf_brigade_sar",
    ),
    CONTROLLER_DIR,
)
MATPLOTLIB_CACHE_DIR = RUNTIME_CACHE_DIR / "matplotlib"
ULTRALYTICS_CONFIG_DIR = RUNTIME_CACHE_DIR / "ultralytics"


def configure_ml_runtime_environment() -> None:
    """Point third-party ML caches at a portable, writable temp location."""

    for env_name, default_path in (
        ("MPLCONFIGDIR", MATPLOTLIB_CACHE_DIR),
        ("YOLO_CONFIG_DIR", ULTRALYTICS_CONFIG_DIR),
    ):
        configured_path = os.environ.get(env_name) or default_path
        runtime_path = resolve_runtime_path(configured_path)
        runtime_path.mkdir(parents=True, exist_ok=True)
        os.environ[env_name] = str(runtime_path)


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
