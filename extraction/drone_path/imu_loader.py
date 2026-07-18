from __future__ import annotations

import csv
from pathlib import Path


_FLOAT_FIELDS = {
    "timestamp",
    "acc_x",
    "acc_y",
    "acc_z",
    "w_x",
    "w_y",
    "w_z",
    "comp_x",
    "comp_y",
    "comp_z",
}


def load_imu(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, float]] = []
        for line_number, row in enumerate(reader, start=2):
            parsed: dict[str, float] = {}
            for field in _FLOAT_FIELDS:
                value = row.get(field)
                if value is None or value == "":
                    raise RuntimeError(f"Missing IMU field {field!r} at {path}:{line_number}")
                parsed[field] = float(value)
            rows.append(parsed)
    if not rows:
        raise RuntimeError(f"No IMU rows found in {path}")
    return rows
