from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..altitude import estimate_altitude_from_imu_csv, normalise_altitude_mode
from .camera_model import load_camera_intrinsics
from .imu_loader import load_imu
from ..io_utils import ensure_parent
from ..progress import ProgressBar


def build_drone_path(
    video: Path,
    imu: Path,
    intrinsics: Path,
    world: str,
    config: dict[str, Any],
    output_csv: Path,
    plot_path: Path | None,
    output_json: Path | None = None,
) -> tuple[list[dict[str, float]], list[dict[str, float]], dict[str, Any]]:
    camera_matrix, dist_coeffs = load_camera_intrinsics(intrinsics)
    fps = _video_fps(video)
    imu_rows = load_imu(imu)
    raw_path = _integrate_imu_xy(imu_rows)
    path_cfg = config.get("path", {})
    altitude_mode = normalise_altitude_mode(path_cfg.get("altitude_mode", config.get("altitude_mode", "aruco")))

    samples: list[dict[str, float]] = []
    aruco_height = None
    altitude_m = None
    altitude_source = "none"
    if altitude_mode == "aruco":
        samples = _aruco_samples(
            video=video,
            camera_matrix=camera_matrix,
            dist_coeffs=dist_coeffs,
            frame_step=int(path_cfg.get("aruco_frame_step", 5)),
            start_seconds=float(path_cfg.get("aruco_scan_start_seconds", 0.0)),
            end_seconds=path_cfg.get("aruco_scan_end_seconds", 10.0),
            marker_size_m=float(path_cfg.get("aruco_marker_size_m", 0.25)),
            dictionaries=list(path_cfg.get("aruco_dictionaries", [])),
        )
        aruco_height = _robust_aruco_height(
            samples=samples,
            stable_after_seconds=float(path_cfg.get("aruco_height_stable_after_seconds", 3.0)),
            min_height_m=float(path_cfg.get("aruco_min_height_m", 1.0)),
            max_height_m=float(path_cfg.get("aruco_max_height_m", 4.0)),
        )
        altitude_m = aruco_height
        altitude_source = "aruco"
    else:
        estimate = estimate_altitude_from_imu_csv(
            imu_rows=imu_rows,
            config=path_cfg,
            default_end_seconds=float(path_cfg.get("imu_altitude_end_seconds", 3.5)),
        )
        if estimate is not None:
            altitude_m, altitude_source = estimate
        print(
            "[mission] ArUco scan skipped: altitude_mode=imu; "
            f"using IMU altitude {altitude_m if altitude_m is not None else 'unavailable'}"
        )
    scale = _fit_imu_scale(raw_path, samples)

    path_rows = _write_path(
        path=raw_path,
        samples=samples,
        scale=scale,
        fps=fps,
        altitude_m=altitude_m,
        altitude_mode=altitude_mode,
        output=ensure_parent(output_csv),
    )
    if plot_path is not None:
        _write_plot(path_rows, ensure_parent(plot_path))

    summary = {
        "video": str(video),
        "imu": str(imu),
        "intrinsics": str(intrinsics),
        "run_label": world,
        "fps": float(fps),
        "altitude_mode": altitude_mode,
        "altitude_m": altitude_m,
        "altitude_source": altitude_source,
        "imu_rows": len(imu_rows),
        "path_rows": len(path_rows),
        "aruco_scan_frame_step": int(path_cfg.get("aruco_frame_step", 5)),
        "aruco_scan_start_seconds": float(path_cfg.get("aruco_scan_start_seconds", 0.0)),
        "aruco_scan_end_seconds": path_cfg.get("aruco_scan_end_seconds", 10.0),
        "aruco_marker_size_m": float(path_cfg.get("aruco_marker_size_m", 0.25)),
        "aruco_sample_count": len(samples),
        "aruco_pose_sample_count": sum(1 for sample in samples if "camera_height_m" in sample),
        "aruco_anchor": _summarize_aruco_anchor(samples),
        "robust_aruco_height_m": aruco_height,
        "imu_to_aruco_scale": float(scale),
        "output_csv": str(output_csv),
        "plot": str(plot_path) if plot_path is not None else None,
    }
    if output_json is not None:
        ensure_parent(output_json).write_text(json.dumps(summary, indent=2) + "\n")
    return path_rows, samples, summary


def _video_fps(video: Path) -> float:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    return fps if fps > 0.0 else 30.0


def _heading_from_compass(row: dict[str, float]) -> float:
    comp_x = row.get("comp_x", 0.0)
    comp_y = row.get("comp_y", 1.0)
    return math.atan2(comp_x, comp_y)


def _integrate_imu_xy(rows: list[dict[str, float]]) -> list[dict[str, float]]:
    path: list[dict[str, float]] = []
    x_m = 0.0
    y_m = 0.0
    vx = 0.0
    vy = 0.0
    previous_t = rows[0]["timestamp"]

    for index, row in enumerate(rows):
        t = row["timestamp"]
        dt = max(0.0, min(t - previous_t, 0.25))
        yaw = _heading_from_compass(row)

        body_ax = row.get("acc_x", 0.0)
        body_ay = row.get("acc_y", 0.0)
        world_ax = body_ax * math.cos(yaw) - body_ay * math.sin(yaw)
        world_ay = body_ax * math.sin(yaw) + body_ay * math.cos(yaw)

        vx += world_ax * dt
        vy += world_ay * dt
        x_m += vx * dt
        y_m += vy * dt

        path.append(
            {
                "frame": float(index),
                "time_s": float(t),
                "x_m": float(x_m),
                "y_m": float(y_m),
                "yaw_rad": float(yaw),
            }
        )
        previous_t = t

    return path


def _aruco_samples(
    video: Path,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    frame_step: int,
    start_seconds: float,
    end_seconds: Any,
    marker_size_m: float,
    dictionaries: list[str],
) -> list[dict[str, float]]:
    if not hasattr(cv2, "aruco"):
        print("[mission] ArUco scan skipped: this OpenCV build does not include cv2.aruco")
        return []

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    start_frame = max(0, int(round(start_seconds * fps)))
    if end_seconds is None:
        end_frame = frame_count - 1
    else:
        end_frame = min(frame_count - 1, int(round(float(end_seconds) * fps)))
    sample_frames = list(range(start_frame, end_frame + 1, max(1, frame_step)))
    dictionary_names = dictionaries or ["DICT_4X4_50", "DICT_4X4_100", "DICT_5X5_50", "DICT_5X5_100"]
    aruco_dictionaries = _load_aruco_dictionaries(dictionary_names)
    if not aruco_dictionaries:
        print("[mission] ArUco scan skipped: no configured ArUco dictionaries were available")
        return []

    print(
        "[mission] ArUco scan: "
        f"{len(sample_frames)} frames from {start_frame / fps:.1f}s to {end_frame / fps:.1f}s"
    )
    progress = ProgressBar("aruco scan", len(sample_frames))
    samples: list[dict[str, float]] = []
    try:
        for progress_index, frame_index in enumerate(sample_frames, start=1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = cap.read()
            if not ok:
                progress.update(progress_index)
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detection = _detect_aruco(gray, aruco_dictionaries)
            if detection is not None:
                dictionary_name, corners, ids = detection
                marker_poses = _estimate_marker_poses(
                    corners=corners,
                    ids=ids,
                    marker_size_m=marker_size_m,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                )
                heights = [pose["height_m"] for pose in marker_poses if np.isfinite(pose["height_m"])]
                anchor = _image_anchor_from_markers(corners, ids)
                sample: dict[str, Any] = {
                    "frame": float(frame_index),
                    "time_s": float(frame_index / fps),
                    "marker_count": float(len(ids)),
                    "dictionary": dictionary_name,
                    "marker_ids": [int(value) for value in ids],
                    "marker_poses": marker_poses,
                }
                if heights:
                    sample["camera_height_m"] = float(np.median(np.array(heights, dtype=np.float64)))
                if anchor is not None:
                    sample.update(anchor)
                samples.append(
                    sample
                )
            progress.update(progress_index)
    finally:
        cap.release()
        progress.finish()

    print(f"[mission] ArUco scan: found markers in {len(samples)} sampled frames")
    return samples


def _load_aruco_dictionaries(names: list[str]) -> list[tuple[str, object]]:
    dictionaries: list[tuple[str, object]] = []
    for name in names:
        dictionary_id = getattr(cv2.aruco, name, None)
        if dictionary_id is None:
            continue
        dictionaries.append((name, cv2.aruco.getPredefinedDictionary(dictionary_id)))
    return dictionaries


def _detect_aruco(gray: np.ndarray, dictionaries: list[tuple[str, object]]) -> tuple[str, list[np.ndarray], np.ndarray] | None:
    parameters = cv2.aruco.DetectorParameters()
    for name, dictionary in dictionaries:
        if hasattr(cv2.aruco, "ArucoDetector"):
            detector = cv2.aruco.ArucoDetector(dictionary, parameters)
            corners, ids, _ = detector.detectMarkers(gray)
        else:
            corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary, parameters=parameters)
        if ids is not None and len(ids) > 0:
            return name, corners, ids.reshape(-1)
    return None


def _estimate_marker_poses(
    corners: list[np.ndarray],
    ids: np.ndarray,
    marker_size_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[dict[str, Any]]:
    half = marker_size_m / 2.0
    object_points = np.array(
        [
            [-half, half, 0.0],
            [half, half, 0.0],
            [half, -half, 0.0],
            [-half, -half, 0.0],
        ],
        dtype=np.float32,
    )
    solve_flag = getattr(cv2, "SOLVEPNP_IPPE_SQUARE", cv2.SOLVEPNP_ITERATIVE)
    poses: list[dict[str, Any]] = []
    for marker_corners, marker_id in zip(corners, ids):
        image_points = np.asarray(marker_corners, dtype=np.float32).reshape(4, 2)
        ok, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            dist_coeffs,
            flags=solve_flag,
        )
        if not ok:
            continue
        t = np.asarray(tvec, dtype=np.float64).reshape(3)
        height_m = abs(float(t[2]))
        if not np.isfinite(height_m) or height_m <= 0.0:
            height_m = float(np.linalg.norm(t))
        poses.append(
            {
                "marker_id": int(marker_id),
                "height_m": height_m,
                "camera_tvec_m": [float(value) for value in t],
                "camera_rvec": [float(value) for value in np.asarray(rvec, dtype=np.float64).reshape(3)],
                "center_px": [float(value) for value in image_points.mean(axis=0)],
            }
        )
    return poses


def _image_anchor_from_markers(corners: list[np.ndarray], ids: np.ndarray) -> dict[str, Any] | None:
    centers = {
        int(marker_id): np.asarray(marker_corners, dtype=np.float64).reshape(4, 2).mean(axis=0)
        for marker_corners, marker_id in zip(corners, ids)
    }
    if 0 not in centers:
        return None

    anchor: dict[str, Any] = {
        "origin_marker_center_px": [float(value) for value in centers[0]],
    }
    if 2 in centers:
        x_vec = centers[2] - centers[0]
        anchor["x_axis_angle_image_rad"] = float(math.atan2(float(x_vec[1]), float(x_vec[0])))
        anchor["origin_to_marker2_px"] = float(np.linalg.norm(x_vec))
    if 1 in centers:
        y_vec = centers[1] - centers[0]
        anchor["y_axis_angle_image_rad"] = float(math.atan2(float(y_vec[1]), float(y_vec[0])))
        anchor["origin_to_marker1_px"] = float(np.linalg.norm(y_vec))
    return anchor


def _summarize_aruco_anchor(samples: list[dict[str, Any]]) -> dict[str, Any]:
    heights = [float(sample["camera_height_m"]) for sample in samples if "camera_height_m" in sample]
    with_origin = [sample for sample in samples if "origin_marker_center_px" in sample]
    summary: dict[str, Any] = {
        "height_sample_count": len(heights),
        "origin_anchor_sample_count": len(with_origin),
    }
    if heights:
        summary["median_camera_height_m"] = float(np.median(np.array(heights, dtype=np.float64)))
        summary["min_camera_height_m"] = float(np.min(np.array(heights, dtype=np.float64)))
        summary["max_camera_height_m"] = float(np.max(np.array(heights, dtype=np.float64)))
    if with_origin:
        latest = with_origin[-1]
        summary["latest_origin_marker_center_px"] = latest.get("origin_marker_center_px")
        summary["latest_x_axis_angle_image_rad"] = latest.get("x_axis_angle_image_rad")
        summary["latest_y_axis_angle_image_rad"] = latest.get("y_axis_angle_image_rad")
    return summary


def _robust_aruco_height(
    samples: list[dict[str, Any]],
    stable_after_seconds: float,
    min_height_m: float,
    max_height_m: float,
) -> float | None:
    candidates = [
        float(sample["camera_height_m"])
        for sample in samples
        if "camera_height_m" in sample
        and float(sample.get("time_s", 0.0)) >= stable_after_seconds
        and min_height_m <= float(sample["camera_height_m"]) <= max_height_m
    ]
    if len(candidates) < 3:
        candidates = [
            float(sample["camera_height_m"])
            for sample in samples
            if "camera_height_m" in sample
            and min_height_m <= float(sample["camera_height_m"]) <= max_height_m
        ]
    if len(candidates) < 3:
        return None

    values = np.array(candidates, dtype=np.float64)
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    if iqr > 0:
        keep = (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)
        values = values[keep]
    if values.size == 0:
        return None
    return float(np.median(values))


def _fit_imu_scale(path: list[dict[str, float]], samples: list[dict[str, float]]) -> float:
    del path
    return 1.0 if not samples else float(samples[0].get("scale", 1.0))


def _write_path(
    path: list[dict[str, float]],
    samples: list[dict[str, float]],
    scale: float,
    fps: float,
    altitude_m: float | None,
    altitude_mode: str,
    output: Path,
) -> list[dict[str, float]]:
    del samples
    rows: list[dict[str, float]] = []
    for row in path:
        scaled = dict(row)
        scaled["timestamp"] = float(row["time_s"])
        scaled["frame"] = float(row["time_s"]) * fps
        scaled["x_m"] = float(row["x_m"]) * scale
        scaled["y_m"] = float(row["y_m"]) * scale
        if altitude_m is not None:
            scaled["altitude_m"] = float(altitude_m)
            scaled["altitude_source"] = altitude_mode
            scaled["aruco_height_m"] = float(altitude_m)
        rows.append(scaled)

    with output.open("w", newline="") as f:
        fieldnames = ["frame", "timestamp", "time_s", "x_m", "y_m", "yaw_rad"]
        if altitude_m is not None:
            fieldnames.extend(["altitude_m", "altitude_source", "aruco_height_m"])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _write_plot(path_rows: list[dict[str, float]], output: Path) -> None:
    image = np.full((600, 600, 3), 255, dtype=np.uint8)
    if path_rows:
        points = np.asarray([[row["x_m"], row["y_m"]] for row in path_rows], dtype=np.float64)
        span = np.ptp(points, axis=0)
        max_span = max(float(span.max()), 1.0)
        centered = points - points.mean(axis=0)
        pixels = centered * (460.0 / max_span)
        pixels[:, 0] += 300.0
        pixels[:, 1] = 300.0 - pixels[:, 1]
        cv_points = np.round(pixels).astype(np.int32)
        cv2.polylines(image, [cv_points], isClosed=False, color=(0, 80, 220), thickness=2)
        cv2.circle(image, tuple(cv_points[0]), 5, (0, 180, 0), -1)
        cv2.circle(image, tuple(cv_points[-1]), 5, (0, 0, 220), -1)
    cv2.imwrite(str(output), image)
