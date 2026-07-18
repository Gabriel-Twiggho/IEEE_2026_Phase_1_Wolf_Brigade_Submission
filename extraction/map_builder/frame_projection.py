from __future__ import annotations

import math

import cv2
import numpy as np


DEFAULT_HORIZONTAL_FOV_DEG = 116.65822873287782
DEFAULT_VERTICAL_FOV_DEG = 84.71718860723323


def interp_rows(rows: list[dict[str, float]], x_key: str, x: float, y_key: str) -> float:
    xs = np.array([r[x_key] for r in rows], dtype=np.float64)
    ys = np.array([r[y_key] for r in rows], dtype=np.float64)
    return float(np.interp(x, xs, ys))


def interp_path(rows: list[dict[str, float]], frame_idx: int, key: str) -> float:
    return interp_rows(rows, "frame", float(frame_idx), key)


def build_imu_yaw_rows(imu_rows: list[dict[str, float]], fps: float, mode: str) -> list[dict[str, float]]:
    if not imu_rows:
        return []

    yaw_rows: list[dict[str, float]] = []
    if mode == "compass":
        for i, row in enumerate(imu_rows):
            timestamp = row.get("timestamp", i / fps)
            yaw = math.atan2(row.get("comp_x", 0.0), row.get("comp_y", 1.0))
            yaw_rows.append({"frame": float(round(timestamp * fps)), "timestamp": timestamp, "yaw_rad": yaw})
        return yaw_rows

    if mode == "gyro":
        yaw = 0.0
        prev_t = imu_rows[0].get("timestamp", 0.0)
        for i, row in enumerate(imu_rows):
            timestamp = row.get("timestamp", i / fps)
            dt = max(0.0, timestamp - prev_t)
            yaw += row.get("w_z", 0.0) * dt
            prev_t = timestamp
            yaw_rows.append({"frame": float(round(timestamp * fps)), "timestamp": timestamp, "yaw_rad": yaw})
        return yaw_rows

    return []


def yaw_at_frame(
    frame_idx: int,
    path_rows: list[dict[str, float]],
    imu_rows: list[dict[str, float]],
    imu_yaw_rows: list[dict[str, float]],
    source: str,
) -> float:
    if source == "path":
        return interp_path(path_rows, frame_idx, "yaw_rad")

    if source == "imu_row_compass" and imu_rows:
        idx = min(max(frame_idx, 0), len(imu_rows) - 1)
        row = imu_rows[idx]
        return math.atan2(row.get("comp_x", 0.0), row.get("comp_y", 1.0))

    if source == "imu_row_gyro" and imu_rows:
        idx = min(max(frame_idx, 0), len(imu_rows) - 1)
        yaw = 0.0
        prev_t = imu_rows[0].get("timestamp", 0.0)
        for i in range(1, idx + 1):
            t = imu_rows[i].get("timestamp", i)
            dt = max(0.0, t - prev_t)
            yaw += imu_rows[i].get("w_z", 0.0) * dt
            prev_t = t
        return yaw

    if source in ("imu_compass", "imu_gyro") and imu_yaw_rows:
        frames = np.array([r["frame"] for r in imu_yaw_rows], dtype=np.float64)
        yaws = np.unwrap(np.array([r["yaw_rad"] for r in imu_yaw_rows], dtype=np.float64))
        return float(np.interp(float(frame_idx), frames, yaws))

    return interp_path(path_rows, frame_idx, "yaw_rad")


def estimate_altitude_from_path(path_rows: list[dict[str, float]], start_time_s: float) -> float | None:
    height_key = "altitude_m" if any("altitude_m" in row for row in path_rows) else "aruco_height_m"
    heights_after_start = [
        row[height_key]
        for row in path_rows
        if height_key in row and row.get("timestamp", row.get("time_s", 0.0)) >= start_time_s
    ]
    if heights_after_start:
        return float(np.median(np.array(heights_after_start, dtype=np.float64)))

    all_heights = [row[height_key] for row in path_rows if height_key in row]
    if all_heights:
        return float(np.median(np.array(all_heights, dtype=np.float64)))

    return None


def camera_footprint_from_fov(
    altitude_m: float,
    horizontal_fov_deg: float,
    vertical_fov_deg: float,
) -> tuple[float, float]:
    horizontal_fov_rad = math.radians(horizontal_fov_deg)
    vertical_fov_rad = math.radians(vertical_fov_deg)
    width_m = 2.0 * altitude_m * math.tan(horizontal_fov_rad / 2.0)
    height_m = 2.0 * altitude_m * math.tan(vertical_fov_rad / 2.0)
    return width_m, height_m


def world_to_canvas(
    x_m: float,
    y_m: float,
    origin_px: tuple[float, float],
    pixels_per_meter: float,
) -> tuple[float, float]:
    ox, oy = origin_px
    canvas_x = ox - y_m * pixels_per_meter
    canvas_y = oy - x_m * pixels_per_meter
    return float(canvas_x), float(canvas_y)


def rotate_bound_size(width: int, height: int, angle_rad: float) -> tuple[int, int]:
    angle_deg = math.degrees(angle_rad)
    centre = (width / 2.0, height / 2.0)
    rot = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)
    cos_a = abs(rot[0, 0])
    sin_a = abs(rot[0, 1])
    new_w = int(height * sin_a + width * cos_a)
    new_h = int(height * cos_a + width * sin_a)
    return max(1, new_w), max(1, new_h)


def crop_center_size_after_rotation(width: int, height: int, fraction: float) -> tuple[int, int]:
    if fraction >= 0.999:
        return width, height
    return max(1, int(math.ceil(width * fraction))), max(1, int(math.ceil(height * fraction)))


def crop_center_fraction_2d(
    image: np.ndarray,
    valid_mask: np.ndarray,
    fraction: float,
) -> tuple[np.ndarray, np.ndarray]:
    if fraction >= 0.999:
        return image, valid_mask
    if fraction <= 0.0:
        raise ValueError("center_crop_fraction must be > 0")

    h, w = image.shape[:2]
    crop_w = max(1, int(round(w * fraction)))
    crop_h = max(1, int(round(h * fraction)))
    x1 = max(0, (w - crop_w) // 2)
    y1 = max(0, (h - crop_h) // 2)
    return image[y1 : y1 + crop_h, x1 : x1 + crop_w], valid_mask[y1 : y1 + crop_h, x1 : x1 + crop_w]


def rotate_bound_float_with_mask(prob: np.ndarray, angle_rad: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = prob.shape[:2]
    angle_deg = math.degrees(angle_rad)
    centre = (w / 2.0, h / 2.0)
    rot = cv2.getRotationMatrix2D(centre, angle_deg, 1.0)

    cos_a = abs(rot[0, 0])
    sin_a = abs(rot[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)

    rot[0, 2] += new_w / 2.0 - centre[0]
    rot[1, 2] += new_h / 2.0 - centre[1]

    rotated_prob = cv2.warpAffine(
        prob.astype(np.float32),
        rot,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0.0,
    )
    source_mask = np.full((h, w), 255, dtype=np.uint8)
    rotated_mask = cv2.warpAffine(
        source_mask,
        rot,
        (new_w, new_h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return rotated_prob, rotated_mask > 0


def compute_canvas_from_sampled_frames(
    path_rows: list[dict[str, float]],
    sample_frames: list[int],
    paste_w: int,
    paste_h: int,
    pixels_per_meter: float,
    centre_crop_fraction: float,
    yaw_source: str,
    imu_rows: list[dict[str, float]],
    imu_yaw_rows: list[dict[str, float]],
    yaw_sign: float,
    safety_padding_px: int,
) -> tuple[int, int, tuple[float, float], dict[str, float]]:
    min_canvas_x = float("inf")
    max_canvas_x = float("-inf")
    min_canvas_y = float("inf")
    max_canvas_y = float("-inf")
    max_frame_w = 0
    max_frame_h = 0

    for frame_idx in sample_frames:
        x_m = interp_path(path_rows, frame_idx, "x_m")
        y_m = interp_path(path_rows, frame_idx, "y_m")
        yaw_rad = yaw_at_frame(frame_idx, path_rows, imu_rows, imu_yaw_rows, yaw_source)
        unrotate_rad = -yaw_sign * yaw_rad

        rotated_w, rotated_h = rotate_bound_size(paste_w, paste_h, unrotate_rad)
        frame_w, frame_h = crop_center_size_after_rotation(rotated_w, rotated_h, centre_crop_fraction)
        max_frame_w = max(max_frame_w, frame_w)
        max_frame_h = max(max_frame_h, frame_h)

        raw_canvas_x = -y_m * pixels_per_meter
        raw_canvas_y = -x_m * pixels_per_meter
        min_canvas_x = min(min_canvas_x, raw_canvas_x - frame_w / 2.0)
        max_canvas_x = max(max_canvas_x, raw_canvas_x + frame_w / 2.0)
        min_canvas_y = min(min_canvas_y, raw_canvas_y - frame_h / 2.0)
        max_canvas_y = max(max_canvas_y, raw_canvas_y + frame_h / 2.0)

    if not np.isfinite(min_canvas_x):
        raise RuntimeError("No sampled frames available for canvas bounds.")

    canvas_w = int(math.ceil(max_canvas_x - min_canvas_x)) + 2 * safety_padding_px
    canvas_h = int(math.ceil(max_canvas_y - min_canvas_y)) + 2 * safety_padding_px
    origin_px = (-min_canvas_x + safety_padding_px, -min_canvas_y + safety_padding_px)
    info = {
        "raw_min_canvas_x": float(min_canvas_x),
        "raw_max_canvas_x": float(max_canvas_x),
        "raw_min_canvas_y": float(min_canvas_y),
        "raw_max_canvas_y": float(max_canvas_y),
        "max_sampled_frame_width_px": int(max_frame_w),
        "max_sampled_frame_height_px": int(max_frame_h),
        "safety_padding_px": int(safety_padding_px),
    }
    return canvas_w, canvas_h, origin_px, info


def project_highres_to_final(
    highres_prob: np.ndarray,
    highres_obs: np.ndarray,
    origin_px: tuple[float, float],
    highres_pixels_per_meter: float,
    output_size_px: int,
    output_resolution_m: float,
    projection_mode: str = "center",
    projection_supersample: int = 1,
    center_world_m: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    out_h = int(output_size_px)
    out_w = int(output_size_px)
    cx = (out_w - 1) / 2.0
    cy = (out_h - 1) / 2.0

    yy_base, xx_base = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
    center_x_m, center_y_m = center_world_m

    def remap_at(xx: np.ndarray, yy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_m = center_x_m + (cy - yy) * output_resolution_m
        y_m = center_y_m + (cx - xx) * output_resolution_m
        map_x = origin_px[0] - y_m * highres_pixels_per_meter
        map_y = origin_px[1] - x_m * highres_pixels_per_meter

        prob = cv2.remap(
            highres_prob.astype(np.float32),
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        obs = cv2.remap(
            highres_obs.astype(np.float32),
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        return prob, obs

    if projection_mode == "center" or projection_supersample <= 1:
        return remap_at(xx_base, yy_base)
    if projection_mode != "max":
        raise ValueError(f"Unknown projection_mode={projection_mode!r}; use 'center' or 'max'.")

    n = max(1, int(projection_supersample))
    offsets = (np.arange(n, dtype=np.float32) + 0.5) / n - 0.5
    final_prob = np.zeros((out_h, out_w), dtype=np.float32)
    final_obs = np.zeros((out_h, out_w), dtype=np.float32)
    for offset_y in offsets:
        for offset_x in offsets:
            prob, obs = remap_at(xx_base + offset_x, yy_base + offset_y)
            final_prob = np.maximum(final_prob, prob)
            final_obs = np.maximum(final_obs, obs)
    return final_prob, final_obs
