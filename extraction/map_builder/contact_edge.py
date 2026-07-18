from __future__ import annotations

import cv2
import numpy as np


def nearest_true_pixel(mask: np.ndarray, x: int, y: int) -> tuple[int, int] | None:
    ys, xs = np.where(mask)
    if xs.size == 0:
        return None
    dx = xs.astype(np.float32) - float(x)
    dy = ys.astype(np.float32) - float(y)
    idx = int(np.argmin(dx * dx + dy * dy))
    return int(xs[idx]), int(ys[idx])


def flood_fill_free_component(free_mask: np.ndarray, seed_xy: tuple[int, int]) -> np.ndarray:
    h, w = free_mask.shape[:2]
    sx, sy = seed_xy
    flood = free_mask.astype(np.uint8).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (int(sx), int(sy)), 2)
    return flood == 2


def border_connected_free_component(free_mask: np.ndarray) -> np.ndarray:
    h, w = free_mask.shape[:2]
    flood = free_mask.astype(np.uint8).copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    for x in range(w):
        if flood[0, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, 0), 2)
        if flood[h - 1, x] == 1:
            cv2.floodFill(flood, flood_mask, (x, h - 1), 2)

    for y in range(h):
        if flood[y, 0] == 1:
            cv2.floodFill(flood, flood_mask, (0, y), 2)
        if flood[y, w - 1] == 1:
            cv2.floodFill(flood, flood_mask, (w - 1, y), 2)

    return flood == 2


def remove_small_binary_components(binary: np.ndarray, min_component_area_px: int) -> np.ndarray:
    if min_component_area_px <= 0:
        return binary.astype(np.uint8)

    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
    keep = np.zeros_like(binary, dtype=np.uint8)
    for label in range(1, n):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_component_area_px:
            keep[labels == label] = 1
    return keep


def extract_contact_wall_edge_probability(
    wall_prob: np.ndarray,
    wall_threshold: float,
    band_px: int,
    close_px: int,
    min_component_area_px: int,
    contact_direction: str,
    line_close_px: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only wall probability on the visible floor-contact side of wall masks."""
    if wall_prob.ndim != 2:
        raise ValueError("wall_prob must be HxW")
    if contact_direction not in ("toward_center", "away_from_center"):
        raise ValueError(f"Unknown contact_direction: {contact_direction}")

    h, w = wall_prob.shape[:2]
    band_px = max(1, int(band_px))
    close_px = max(0, int(close_px))
    min_component_area_px = max(0, int(min_component_area_px))
    line_close_px = max(0, int(line_close_px))

    wall_bin = (wall_prob >= float(wall_threshold)).astype(np.uint8)
    if close_px > 0:
        k = 2 * close_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        wall_bin = cv2.morphologyEx(wall_bin, cv2.MORPH_CLOSE, kernel)

    wall_bin = remove_small_binary_components(wall_bin, min_component_area_px)
    free = wall_bin == 0

    image_cx = int(round((w - 1) / 2.0))
    image_cy = int(round((h - 1) / 2.0))
    if contact_direction == "toward_center":
        seed = (image_cx, image_cy) if free[image_cy, image_cx] else nearest_true_pixel(free, image_cx, image_cy)
        if seed is None:
            empty_prob = np.zeros_like(wall_prob, dtype=np.float32)
            return empty_prob, np.zeros_like(wall_prob, dtype=bool)
        contact_free = flood_fill_free_component(free, seed)
    else:
        contact_free = border_connected_free_component(free)
        if not np.any(contact_free):
            empty_prob = np.zeros_like(wall_prob, dtype=np.float32)
            return empty_prob, np.zeros_like(wall_prob, dtype=bool)

    grow_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    grown = contact_free.astype(np.uint8)
    edge = np.zeros_like(wall_bin, dtype=np.uint8)
    for _ in range(band_px):
        grown = cv2.dilate(grown, grow_kernel, iterations=1)
        ring = (grown > 0) & (wall_bin > 0) & (edge == 0)
        edge[ring] = 1

    if line_close_px > 0 and np.any(edge):
        k = 2 * line_close_px + 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(edge, cv2.MORPH_CLOSE, close_kernel)

        allowed = contact_free.astype(np.uint8)
        for _ in range(band_px + line_close_px):
            allowed = cv2.dilate(allowed, grow_kernel, iterations=1)
        allowed_wall_band = (allowed > 0) & (wall_bin > 0)
        edge = ((closed > 0) & allowed_wall_band).astype(np.uint8)

    edge_prob = np.zeros_like(wall_prob, dtype=np.float32)
    edge_mask = edge.astype(bool)
    edge_prob[edge_mask] = wall_prob[edge_mask]
    return edge_prob.astype(np.float32), edge_mask
