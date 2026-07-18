from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def paste_wall_probability(
    wall_sum: np.ndarray,
    obs_weight: np.ndarray,
    prob_image: np.ndarray,
    centre_xy: tuple[float, float],
    valid_mask: np.ndarray,
) -> None:
    h, w = prob_image.shape[:2]
    cx, cy = centre_xy
    x1 = int(round(cx - w / 2.0))
    y1 = int(round(cy - h / 2.0))
    x2 = x1 + w
    y2 = y1 + h

    canvas_h, canvas_w = wall_sum.shape[:2]
    ix1 = max(0, x1)
    iy1 = max(0, y1)
    ix2 = min(canvas_w, x2)
    iy2 = min(canvas_h, y2)
    if ix1 >= ix2 or iy1 >= iy2:
        return

    sx1 = ix1 - x1
    sy1 = iy1 - y1
    sx2 = sx1 + (ix2 - ix1)
    sy2 = sy1 + (iy2 - iy1)
    src_prob = prob_image[sy1:sy2, sx1:sx2]
    src_valid = valid_mask[sy1:sy2, sx1:sx2]

    wall_region = wall_sum[iy1:iy2, ix1:ix2]
    obs_region = obs_weight[iy1:iy2, ix1:ix2]
    wall_region[src_valid] += src_prob[src_valid]
    obs_region[src_valid] += 1.0


def make_probability_image(wall_sum: np.ndarray, obs_weight: np.ndarray) -> np.ndarray:
    prob = np.zeros_like(wall_sum, dtype=np.float32)
    valid = obs_weight > 0
    prob[valid] = wall_sum[valid] / obs_weight[valid]
    return np.clip(prob, 0.0, 1.0)


def crop_float_to_observed(
    image: np.ndarray,
    observed: np.ndarray,
    padding_px: int,
) -> tuple[np.ndarray, dict[str, int]]:
    ys, xs = np.where(observed)
    if len(xs) == 0 or len(ys) == 0:
        h, w = image.shape[:2]
        return image, {"x1": 0, "y1": 0, "x2": w, "y2": h, "unchanged": 1}

    h, w = image.shape[:2]
    x1 = max(0, int(xs.min()) - padding_px)
    x2 = min(w, int(xs.max()) + padding_px + 1)
    y1 = max(0, int(ys.min()) - padding_px)
    y2 = min(h, int(ys.max()) + padding_px + 1)
    return image[y1:y2, x1:x2].copy(), {
        "x1": int(x1),
        "y1": int(y1),
        "x2": int(x2),
        "y2": int(y2),
        "original_width": int(w),
        "original_height": int(h),
        "cropped_width": int(x2 - x1),
        "cropped_height": int(y2 - y1),
        "unchanged": 0,
    }


def save_float_png(path: Path, image: np.ndarray) -> None:
    out = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), out)


def remove_small_components(binary: np.ndarray, min_area_px: int) -> np.ndarray:
    if min_area_px <= 0:
        return binary

    binary_u8 = binary.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_u8, connectivity=8)
    keep = np.zeros_like(binary_u8, dtype=np.uint8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            keep[labels == label] = 1
    return keep.astype(bool)


def clean_wall_binary(
    wall_binary: np.ndarray,
    close_px: int,
    open_px: int,
    dilate_px: int,
    min_wall_area_px: int,
) -> np.ndarray:
    img = wall_binary.astype(np.uint8)
    if open_px > 0:
        k = 2 * open_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    img = remove_small_components(img.astype(bool), min_wall_area_px).astype(np.uint8)

    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if dilate_px > 0:
        k = 2 * dilate_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img = cv2.dilate(img, kernel, iterations=1)

    return img.astype(bool)


def make_black_white_map(wall_binary: np.ndarray) -> np.ndarray:
    out = np.full(wall_binary.shape[:2], 255, dtype=np.uint8)
    out[wall_binary] = 0
    return out


def orient_wall_binary_for_supervisor(wall_binary: np.ndarray, orientation: str) -> np.ndarray:
    orientation = orientation.strip().lower()
    if orientation in ("origin", "origin_xy", "navigation"):
        return wall_binary
    if orientation in ("supervisor", "webots", "webots_xy"):
        return np.rot90(wall_binary, k=-1).copy()
    raise ValueError(f"Unknown map output orientation: {orientation!r}")


def center_wall_binary_bbox(wall_binary: np.ndarray) -> tuple[np.ndarray, dict[str, int | float]]:
    ys, xs = np.where(wall_binary)
    h, w = wall_binary.shape[:2]
    if len(xs) == 0 or len(ys) == 0:
        return wall_binary, {
            "applied": 0,
            "dx_px": 0,
            "dy_px": 0,
            "reason": "no_wall_pixels",
        }

    bbox_cx = (float(xs.min()) + float(xs.max())) / 2.0
    bbox_cy = (float(ys.min()) + float(ys.max())) / 2.0
    image_cx = (w - 1) / 2.0
    image_cy = (h - 1) / 2.0
    dx = int(round(image_cx - bbox_cx))
    dy = int(round(image_cy - bbox_cy))
    if dx == 0 and dy == 0:
        return wall_binary, {
            "applied": 0,
            "dx_px": 0,
            "dy_px": 0,
            "bbox_center_px": [bbox_cx, bbox_cy],
            "image_center_px": [image_cx, image_cy],
        }

    transform = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]], dtype=np.float32)
    shifted = cv2.warpAffine(
        wall_binary.astype(np.uint8),
        transform,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ).astype(bool)
    return shifted, {
        "applied": 1,
        "dx_px": dx,
        "dy_px": dy,
        "bbox_center_px": [bbox_cx, bbox_cy],
        "image_center_px": [image_cx, image_cy],
    }
