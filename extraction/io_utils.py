from __future__ import annotations

import json
from pathlib import Path, PureWindowsPath
from typing import Any

from config import CONTROLLER_DIR, RECORDINGS_DIR
from .altitude import normalise_altitude_mode


PROPOSED_SOLUTION_DIR = CONTROLLER_DIR
CONFIG_DIR = PROPOSED_SOLUTION_DIR / "extraction" / "config"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "extraction_config.json"
TASK_CONFIG_SECTIONS = {
    "drone_path": "path",
    "victim_locator": "victims",
    "map_builder": "walls",
}


def _unique_paths(paths: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for path in paths:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        unique.append(resolved)
    return unique


def _validated_recordings_relative(
    relative: str,
    shortcut: str,
) -> Path:
    if not relative:
        raise RuntimeError(
            f"{shortcut} must include a recording name, such as "
            f"{shortcut.rstrip('/')}/small_world."
        )

    path = Path(relative)
    windows_path = PureWindowsPath(relative)
    if (
        path.is_absolute()
        or windows_path.is_absolute()
        or bool(windows_path.drive)
        or ".." in path.parts
        or ".." in windows_path.parts
    ):
        raise RuntimeError(
            f"The {shortcut} shortcut must stay inside the configured "
            "recordings directory. Use an explicit path for files elsewhere."
        )
    return path


def _recording_base_paths(value: str | Path) -> list[Path]:
    text = str(value).strip()
    if not text:
        raise RuntimeError("Recording input cannot be empty.")

    # `/recordings/...` is a portable virtual prefix for the official
    # competition recordings directory, not a machine-root filesystem path.
    portable_text = text.replace("\\", "/")
    if portable_text in ("/recordings", "/recordings/"):
        _validated_recordings_relative("", "/recordings/")
    if portable_text.startswith("/recordings/"):
        relative = portable_text[len("/recordings/") :]
        path = _validated_recordings_relative(relative, "/recordings/")
        return [(RECORDINGS_DIR / path).resolve()]

    path = Path(text).expanduser()
    if path.is_absolute():
        return [path.resolve()]

    portable_lower = portable_text.lower()
    if portable_lower in ("recordings", "recordings/"):
        _validated_recordings_relative("", "recordings/")
    if portable_lower.startswith("recordings/"):
        relative = portable_text[len("recordings/") :]
        recording_path = _validated_recordings_relative(relative, "recordings/")
        return [(RECORDINGS_DIR / recording_path).resolve()]

    parts = path.parts
    is_explicit_relative = portable_text.startswith(("./", "../"))
    if is_explicit_relative or len(parts) > 1:
        return [(Path.cwd() / path).resolve()]

    if path.suffix:
        return _unique_paths([RECORDINGS_DIR / path, Path.cwd() / path])
    return [(RECORDINGS_DIR / path).resolve()]


def _recording_file_candidates(
    value: str | Path,
    expected_suffix: str,
) -> list[Path]:
    suffix = expected_suffix if expected_suffix.startswith(".") else f".{expected_suffix}"
    candidates: list[Path] = []

    for base in _recording_base_paths(value):
        if base.suffix.lower() == suffix.lower():
            candidates.append(base)
        elif not base.suffix:
            if base.name.lower().endswith("_flyover"):
                candidates.append(base.with_suffix(suffix))
            else:
                candidates.append(base.with_name(f"{base.name}_flyover{suffix}"))
                candidates.append(base.with_suffix(suffix))

    return _unique_paths(candidates)


def resolve_recording_file(
    value: str | Path,
    expected_suffix: str,
    label: str,
) -> Path:
    expected_suffix = (
        expected_suffix
        if expected_suffix.startswith(".")
        else f".{expected_suffix}"
    )
    supplied_suffix = Path(str(value).replace("\\", "/")).suffix
    if supplied_suffix and supplied_suffix.lower() != expected_suffix.lower():
        raise RuntimeError(
            f"{label.capitalize()} must be a {expected_suffix} file; "
            f"received {str(value)!r}."
        )

    candidates = _recording_file_candidates(value, expected_suffix)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    attempted = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise RuntimeError(
        f"Could not resolve {label} input {str(value)!r}. Looked for:\n"
        f"{attempted}\n"
        f"Expected official recordings under {RECORDINGS_DIR}. "
        "Set SAR_RECORDINGS_DIR to override that location."
    )


def resolve_recording_inputs(
    video_value: str | Path,
    imu_value: str | Path | None = None,
) -> tuple[Path, Path]:
    video = resolve_recording_file(video_value, ".mp4", "flyover video")

    if imu_value is not None:
        imu = resolve_recording_file(imu_value, ".csv", "IMU CSV")
        return video, imu

    imu = video.with_suffix(".csv")
    if not imu.is_file():
        raise RuntimeError(
            f"Could not infer the IMU CSV for {video}. Expected {imu}. "
            "Pass --imu explicitly if it has a different name or location."
        )
    return video, imu


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
    return video.stem.removesuffix("_flyover")


def ensure_parent(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path
