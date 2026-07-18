from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from .altitude import normalise_altitude_mode


PROPOSED_SOLUTION_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROPOSED_SOLUTION_DIR.parents[1]
TOOLS_DIR = REPO_ROOT / "tools"
CONFIG_DIR = PROPOSED_SOLUTION_DIR / "extraction" / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "extraction_config.json"
TASK_CONFIG_SECTIONS = {
    "drone_path": "path",
    "victim_locator": "victims",
    "map_builder": "walls",
}


def ensure_tool_imports() -> None:
    """Make the repository tools directory importable for reused extraction code."""
    tools_text = str(TOOLS_DIR)
    if tools_text not in sys.path:
        sys.path.insert(0, tools_text)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in {path}")
    return data


def load_pipeline_config(path: Path) -> dict[str, Any]:
    config = load_json(path)
    for task_name, section_name in TASK_CONFIG_SECTIONS.items():
        task_configs = config.get("task_configs", {})
        task_config_path = task_configs.get(task_name) if isinstance(task_configs, dict) else None
        if task_config_path is None:
            continue

        resolved = resolve_project_path(task_config_path)
        if resolved is None:
            continue
        config[section_name] = load_json(resolved)
    altitude_mode = config.get("altitude_mode")
    if altitude_mode is not None:
        altitude_mode = normalise_altitude_mode(altitude_mode)
        config["altitude_mode"] = altitude_mode
        config.setdefault("path", {})["altitude_mode"] = altitude_mode
        config.setdefault("walls", {})["altitude_mode"] = altitude_mode
        config["walls"]["camera_altitude_source"] = altitude_mode
    return config


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def resolve_project_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return (PROPOSED_SOLUTION_DIR / path).resolve()


def output_path(config: dict[str, Any], key: str) -> Path:
    outputs = config.get("outputs", {})
    if key not in outputs:
        raise RuntimeError(f"Missing output path config for {key!r}")
    resolved = resolve_project_path(outputs[key])
    assert resolved is not None
    return resolved


def configured_path(config: dict[str, Any], key: str) -> Path:
    paths = config.get("paths", {})
    if key not in paths:
        raise RuntimeError(f"Missing path config for {key!r}")
    resolved = resolve_project_path(paths[key])
    assert resolved is not None
    return resolved


def infer_run_label(video: Path, config: dict[str, Any]) -> str:
    del config
    return video.stem.replace("_flyover", "")


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
