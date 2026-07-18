from __future__ import annotations

from typing import Any

import numpy as np


VALID_ALTITUDE_MODES = {"aruco", "imu"}


def normalise_altitude_mode(value: Any) -> str:
    mode = str(value or "aruco").strip().lower()
    if mode in ("imu_csv", "csv"):
        mode = "imu"
    if mode in ("path_csv", "path"):
        mode = "aruco"
    if mode not in VALID_ALTITUDE_MODES:
        raise RuntimeError(f"Unknown altitude_mode={mode!r}; use 'aruco' or 'imu'.")
    return mode


def estimate_altitude_from_imu_csv(
    imu_rows: list[dict[str, float]],
    config: dict[str, Any],
    default_end_seconds: float,
) -> tuple[float, str] | None:
    if not imu_rows:
        return None

    end_seconds = float(config.get("imu_altitude_end_seconds", default_end_seconds))
    rows = [row for row in imu_rows if float(row.get("timestamp", 0.0)) <= end_seconds]
    if len(rows) < 3:
        return None

    gravity_config = config.get("imu_altitude_gravity_mps2")
    gravity_mode = str(config.get("imu_altitude_gravity_mode", "zero_velocity_at_end")).strip().lower()
    if gravity_config is not None:
        gravity = float(gravity_config)
        gravity_source = "config"
    elif gravity_mode in ("zero_velocity_at_end", "zero_velocity", "max_height"):
        gravity = _mean_acc_z_over_time(rows)
        gravity_source = f"zero_velocity_at_{end_seconds:.2f}s"
    elif gravity_mode in ("standard", "earth"):
        gravity = 9.80665
        gravity_source = "standard"
    else:
        raise RuntimeError(
            "Unknown imu_altitude_gravity_mode="
            f"{gravity_mode!r}; use 'zero_velocity_at_end' or 'standard'."
        )

    z_m = 0.0
    velocity_mps = 0.0
    prev_t = float(rows[0].get("timestamp", 0.0))
    for row in rows:
        t = float(row.get("timestamp", prev_t))
        dt = max(0.0, min(t - prev_t, 0.25))
        velocity_mps += (float(row.get("acc_z", gravity)) - gravity) * dt
        z_m += velocity_mps * dt
        prev_t = t

    if not np.isfinite(z_m) or z_m <= 0.0:
        return None

    return (
        float(z_m),
        (
            "imu_csv_acc_z_double_integral_"
            f"{gravity_source}_g={gravity:.5f}_end={end_seconds:.2f}s"
        ),
    )


def _mean_acc_z_over_time(rows: list[dict[str, float]]) -> float:
    weighted_acc = 0.0
    total_dt = 0.0
    prev_t = float(rows[0].get("timestamp", 0.0))
    for row in rows:
        t = float(row.get("timestamp", prev_t))
        dt = max(0.0, min(t - prev_t, 0.25))
        weighted_acc += float(row.get("acc_z", 9.80665)) * dt
        total_dt += dt
        prev_t = t
    if total_dt <= 0.0:
        return 9.80665
    return weighted_acc / total_dt
