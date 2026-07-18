from __future__ import annotations

import math
from typing import Any


def is_duplicate_victim(
    existing: dict[str, Any],
    candidate: dict[str, Any],
    merge_distance_m: float,
) -> bool:
    return (
        math.hypot(
            float(existing["x_m"]) - float(candidate["x_m"]),
            float(existing["y_m"]) - float(candidate["y_m"]),
        )
        <= merge_distance_m
    )


def update_existing_victim(existing: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    count = int(existing.get("detection_count", 1)) + 1
    previous_count = count - 1
    existing["x_m"] = (float(existing["x_m"]) * previous_count + float(candidate["x_m"])) / count
    existing["y_m"] = (float(existing["y_m"]) * previous_count + float(candidate["y_m"])) / count
    existing["detection_count"] = count
    existing["best_confidence"] = max(
        float(existing.get("best_confidence", 0.0)),
        float(candidate.get("confidence", candidate.get("best_confidence", 0.0))),
    )
    return existing
