from __future__ import annotations

import csv
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..drone_path.camera_model import load_camera_intrinsics
from ..io_utils import ensure_parent
from ..progress import ProgressBar
from ..video_reader import read_cached_frame


def _load_yolo_class():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/sar-matplotlib")
    os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/sar-ultralytics")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Unable to import Axes3D\..*",
            category=UserWarning,
            module=r"matplotlib\.projections",
        )
        from ultralytics import YOLO

    return YOLO


def detect_victims(
    video: Path,
    model_path: Path,
    path_rows: list[dict[str, float]],
    aruco_samples: list[dict[str, float]],
    intrinsics: Path,
    config: dict[str, Any],
    detailed_csv: Path | None,
    detailed_json: Path | None,
    plot_path: Path | None,
    annotated_dir: Path | None,
    frame_cache: dict[int, object] | None = None,
) -> list[dict[str, Any]]:
    camera_matrix, _ = load_camera_intrinsics(intrinsics)
    victim_cfg = config.get("victims", {})
    return _detect_victims_in_video(
        video=video,
        model_path=model_path,
        path_rows=path_rows,
        samples=aruco_samples,
        camera_matrix=camera_matrix,
        victim_output=detailed_csv,
        victim_summary=detailed_json,
        victim_plot=plot_path,
        confidence_threshold=float(victim_cfg.get("confidence_threshold", 0.70)),
        skip_seconds=float(victim_cfg.get("skip_seconds", 3.5)),
        victim_frame_step=int(victim_cfg.get("frame_step", 10)),
        merge_distance_m=float(victim_cfg.get("merge_distance_m", 0.90)),
        default_height_m=float(victim_cfg.get("default_height_m", 2.5)),
        max_aruco_height_delta_m=float(victim_cfg.get("max_aruco_height_delta_m", 0.20)),
        image_forward_sign=float(victim_cfg.get("image_forward_sign", -1.0)),
        image_left_sign=float(victim_cfg.get("image_left_sign", -1.0)),
        annotated_dir=annotated_dir,
        max_detections=int(victim_cfg.get("max_detections", 20)),
        min_victim_detections=int(victim_cfg.get("min_detections", 15)),
        device=str(victim_cfg.get("device", "cuda")),
        frame_cache=frame_cache,
    )


def victim_sample_frames(frame_count: int, victim_cfg: dict[str, Any]) -> list[int]:
    frame_step = max(1, int(victim_cfg.get("frame_step", 10)))
    fps = float(victim_cfg.get("_video_fps", 30.0))
    start_frame = max(0, int(round(float(victim_cfg.get("skip_seconds", 3.5)) * fps)))
    return list(range(start_frame, frame_count, frame_step))


def _detect_victims_in_video(
    video: Path,
    model_path: Path,
    path_rows: list[dict[str, float]],
    samples: list[dict[str, float]],
    camera_matrix: np.ndarray,
    victim_output: Path | None,
    victim_summary: Path | None,
    victim_plot: Path | None,
    confidence_threshold: float,
    skip_seconds: float,
    victim_frame_step: int,
    merge_distance_m: float,
    default_height_m: float,
    max_aruco_height_delta_m: float,
    image_forward_sign: float,
    image_left_sign: float,
    annotated_dir: Path | None,
    max_detections: int,
    min_victim_detections: int,
    device: str,
    frame_cache: dict[int, object] | None,
) -> list[dict[str, Any]]:
    del samples
    try:
        YOLO = _load_yolo_class()
    except ImportError:
        _write_victim_artifacts([], [], victim_output, victim_summary, victim_plot)
        return []

    model = YOLO(str(model_path))
    device = _resolve_yolo_device(device)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frame_step = max(1, int(victim_frame_step))
    sampled_frames = list(range(max(0, int(round(skip_seconds * fps))), frame_count, frame_step))
    progress = ProgressBar("victim frames", len(sampled_frames))
    detections: list[dict[str, Any]] = []
    victims: list[dict[str, Any]] = []

    if annotated_dir is not None:
        annotated_dir.mkdir(parents=True, exist_ok=True)

    try:
        for progress_index, frame_index in enumerate(sampled_frames, start=1):
            frame = read_cached_frame(frame_cache, frame_index)
            if frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                ok, frame = cap.read()
            else:
                ok = True
            if not ok:
                progress.update(progress_index)
                continue
            result = model.predict(frame, conf=confidence_threshold, verbose=False, device=device)[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                progress.update(progress_index)
                continue

            for box_index, box in enumerate(boxes):
                confidence = float(box.conf[0])
                if confidence < confidence_threshold:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box.xyxy[0]]
                u = (x1 + x2) / 2.0
                v = (y1 + y2) / 2.0
                pose = _pose_at_time(path_rows, frame_index / fps)
                projection_height_m = _projection_height(
                    pose=pose,
                    default_height_m=default_height_m,
                    max_aruco_height_delta_m=max_aruco_height_delta_m,
                )
                x_m, y_m = _project_pixel_to_origin(
                    u=u,
                    v=v,
                    pose=pose,
                    camera_matrix=camera_matrix,
                    default_height_m=projection_height_m,
                    image_forward_sign=image_forward_sign,
                    image_left_sign=image_left_sign,
                )
                detection = {
                    "frame": frame_index,
                    "time_s": frame_index / fps,
                    "confidence": confidence,
                    "bbox_x1": x1,
                    "bbox_y1": y1,
                    "bbox_x2": x2,
                    "bbox_y2": y2,
                    "x_m": x_m,
                    "y_m": y_m,
                }
                detections.append(detection)
                _merge_detection(victims, detection, merge_distance_m)

                if annotated_dir is not None:
                    cv2.rectangle(
                        frame,
                        (int(x1), int(y1)),
                        (int(x2), int(y2)),
                        (0, 220, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"{confidence:.2f}",
                        (int(x1), max(0, int(y1) - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 220, 0),
                        2,
                    )
                    cv2.imwrite(str(annotated_dir / f"frame_{frame_index:06d}_{box_index}.jpg"), frame)
            progress.update(progress_index)
    finally:
        cap.release()
        progress.finish()

    filtered = [
        victim
        for victim in victims
        if int(victim.get("detection_count", 0)) >= max(1, min_victim_detections)
    ]
    filtered.sort(key=lambda item: float(item.get("best_confidence", 0.0)), reverse=True)
    filtered = filtered[:max(0, max_detections)]
    _write_victim_artifacts(filtered, detections, victim_output, victim_summary, victim_plot)
    return filtered


def _resolve_yolo_device(requested: str) -> str:
    if requested.startswith("cuda"):
        try:
            import torch
        except Exception:
            cuda_available = False
        else:
            cuda_available = bool(torch.cuda.is_available())

        if not cuda_available:
            print(
                "[WARN] CUDA was requested for victim detection, but CUDA is not available. "
                "Falling back to CPU for this run. To make CPU explicit, set "
                "`device`: `cpu` in extraction/config/victim_locator_config.json.",
                file=sys.stderr,
            )
            return "cpu"
    return requested


def _projection_height(
    pose: dict[str, float],
    default_height_m: float,
    max_aruco_height_delta_m: float,
) -> float:
    del max_aruco_height_delta_m
    height = pose.get("altitude_m", pose.get("aruco_height_m"))
    if height is None:
        return default_height_m
    return float(height)


def _pose_at_time(path_rows: list[dict[str, float]], time_s: float) -> dict[str, float]:
    if not path_rows:
        return {"x_m": 0.0, "y_m": 0.0, "yaw_rad": 0.0}
    return min(path_rows, key=lambda row: abs(float(row.get("time_s", 0.0)) - time_s))


def _project_pixel_to_origin(
    u: float,
    v: float,
    pose: dict[str, float],
    camera_matrix: np.ndarray,
    default_height_m: float,
    image_forward_sign: float,
    image_left_sign: float,
) -> tuple[float, float]:
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


def _merge_detection(
    victims: list[dict[str, Any]],
    detection: dict[str, Any],
    merge_distance_m: float,
) -> None:
    for victim in victims:
        distance = math.hypot(
            float(victim["x_m"]) - float(detection["x_m"]),
            float(victim["y_m"]) - float(detection["y_m"]),
        )
        if distance <= merge_distance_m:
            count = int(victim["detection_count"]) + 1
            victim["x_m"] = (
                float(victim["x_m"]) * int(victim["detection_count"]) + float(detection["x_m"])
            ) / count
            victim["y_m"] = (
                float(victim["y_m"]) * int(victim["detection_count"]) + float(detection["y_m"])
            ) / count
            victim["detection_count"] = count
            victim["best_confidence"] = max(float(victim["best_confidence"]), float(detection["confidence"]))
            return

    victims.append(
        {
            "x_m": float(detection["x_m"]),
            "y_m": float(detection["y_m"]),
            "best_confidence": float(detection["confidence"]),
            "detection_count": 1,
        }
    )


def _write_victim_artifacts(
    victims: list[dict[str, Any]],
    detections: list[dict[str, Any]],
    detailed_csv: Path | None,
    detailed_json: Path | None,
    plot_path: Path | None,
) -> None:
    if detailed_csv is not None:
        ensure_parent(detailed_csv)
        with detailed_csv.open("w", newline="") as f:
            fieldnames = [
                "frame",
                "time_s",
                "confidence",
                "bbox_x1",
                "bbox_y1",
                "bbox_x2",
                "bbox_y2",
                "x_m",
                "y_m",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for detection in detections:
                writer.writerow(detection)

    if detailed_json is not None:
        ensure_parent(detailed_json).write_text(
            json.dumps({"victims": victims, "detections": detections}, indent=2)
            + "\n"
        )

    if plot_path is not None:
        _write_victim_plot(victims, plot_path)


def _write_victim_plot(victims: list[dict[str, Any]], output: Path) -> None:
    ensure_parent(output)
    image = np.full((600, 600, 3), 255, dtype=np.uint8)
    if victims:
        points = np.asarray([[victim["x_m"], victim["y_m"]] for victim in victims], dtype=np.float64)
        span = max(float(np.ptp(points, axis=0).max()), 1.0)
        centered = points - points.mean(axis=0)
        pixels = centered * (460.0 / span)
        pixels[:, 0] += 300.0
        pixels[:, 1] = 300.0 - pixels[:, 1]
        for index, point in enumerate(np.round(pixels).astype(np.int32), start=1):
            cv2.circle(image, tuple(point), 7, (0, 0, 220), -1)
            cv2.putText(
                image,
                str(index),
                (int(point[0]) + 9, int(point[1]) - 9),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
    cv2.imwrite(str(output), image)
