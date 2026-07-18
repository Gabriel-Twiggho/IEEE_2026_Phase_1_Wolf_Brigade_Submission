from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def load_camera_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open() as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise RuntimeError(f"Expected JSON object in camera intrinsics file: {path}")

    try:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        dist_coeffs = np.asarray(data["distortion_coefficients"], dtype=np.float64)
    except KeyError as exc:
        raise RuntimeError(f"Missing camera intrinsics field {exc.args[0]!r} in {path}") from exc

    if camera_matrix.shape != (3, 3):
        raise RuntimeError(f"Expected camera_matrix to be 3x3 in {path}")
    if dist_coeffs.ndim != 1:
        dist_coeffs = dist_coeffs.reshape(-1)
    if dist_coeffs.size not in (4, 5, 8, 12, 14):
        raise RuntimeError(
            f"Expected 4, 5, 8, 12, or 14 distortion coefficients in {path}; "
            f"got {dist_coeffs.size}"
        )

    return camera_matrix, dist_coeffs
