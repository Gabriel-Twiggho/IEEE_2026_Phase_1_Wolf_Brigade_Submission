from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


def write_official_victim_estimates(victims: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as f:
        writer = csv.writer(f)
        for victim in victims:
            writer.writerow([f"{float(victim['x_m']):.6f}", f"{float(victim['y_m']):.6f}"])

