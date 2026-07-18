from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def enforce_official_map(path: Path, size_px: int = 600) -> None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"Could not read generated map: {path}")
    if image.shape[:2] != (size_px, size_px):
        image = cv2.resize(image, (size_px, size_px), interpolation=cv2.INTER_NEAREST)

    binary = np.where(image < 128, 0, 255).astype(np.uint8)
    rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), rgb)

