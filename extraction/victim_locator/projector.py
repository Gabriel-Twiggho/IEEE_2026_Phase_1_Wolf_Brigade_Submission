from __future__ import annotations

import math
from typing import Any

import numpy as np


def project_bbox_to_origin_xy(
    bbox_xyxy: tuple[float, float, float, float],
    pose: dict[str, float],
    camera_matrix: np.ndarray,
    default_height_m: float,
    image_forward_sign: float = -1.0,
    image_left_sign: float = -1.0,
) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox_xyxy
    u = (x1 + x2) / 2.0
    v = (y1 + y2) / 2.0

    fx = float(camera_matrix[0, 0])
    fy = float(camera_matrix[1, 1])
    cx = float(camera_matrix[0, 2])
    cy = float(camera_matrix[1, 2])

    forward = image_forward_sign * ((v - cy) / fy) * default_height_m
    left = image_left_sign * ((u - cx) / fx) * default_height_m
    yaw = float(pose.get("yaw_rad", 0.0))
    dx = forward * math.cos(yaw) - left * math.sin(yaw)
    dy = forward * math.sin(yaw) + left * math.cos(yaw)
    return float(pose.get("x_m", 0.0) + dx), float(pose.get("y_m", 0.0) + dy)


__all__ = ["project_bbox_to_origin_xy"]
