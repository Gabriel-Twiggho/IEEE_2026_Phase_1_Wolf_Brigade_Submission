from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..altitude import estimate_altitude_from_imu_csv, normalise_altitude_mode
from .contact_edge import extract_contact_wall_edge_probability
from .frame_projection import (
    DEFAULT_HORIZONTAL_FOV_DEG,
    DEFAULT_VERTICAL_FOV_DEG,
    build_imu_yaw_rows,
    camera_footprint_from_fov,
    crop_center_fraction_2d,
    estimate_altitude_from_path,
    interp_path,
    rotate_bound_float_with_mask,
    world_to_canvas,
    yaw_at_frame,
)
from .map_exporter import enforce_official_map
from .wall_fuser import (
    center_wall_binary_bbox,
    clean_wall_binary,
    crop_float_to_observed,
    make_black_white_map,
    make_probability_image,
    orient_wall_binary_for_supervisor,
    paste_wall_probability,
    save_float_png,
)
from .wall_projector import compute_canvas_from_sampled_frames, project_highres_to_final
from .wall_segmenter import (
    default_device,
    infer_model_input_wh,
    infer_wall_probability,
    load_model_metadata,
    load_segformer,
    resolve_wall_class_id,
)
from ..drone_path.imu_loader import load_imu
from ..io_utils import ensure_parent
from ..progress import ProgressBar
from ..video_reader import read_cached_frame


def build_wall_map(
    video: Path,
    imu: Path,
    path_csv: Path,
    model_path: Path,
    world: str,
    config: dict[str, Any],
    map_output: Path,
    info_output: Path,
    debug_dir: Path | None = None,
    frame_cache: dict[int, object] | None = None,
    sample_frames_override: list[int] | None = None,
) -> dict[str, Any]:
    wall_cfg = config.get("walls", {})
    path_rows = _load_path_csv(path_csv)
    imu_rows = load_imu(imu) if imu.exists() else []

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        sample_rate = float(wall_cfg.get("sample_rate", 1.5))
        skip_seconds = float(wall_cfg.get("skip_seconds", 2.5))
        end_seconds_value = wall_cfg.get("end_seconds")
        end_seconds = None if end_seconds_value is None else float(end_seconds_value)
        start_frame = int(round(skip_seconds * fps))
        end_frame = frame_count - 1 if end_seconds is None else min(frame_count - 1, int(round(end_seconds * fps)))
        frame_step = max(1, int(round(fps / max(sample_rate, 0.001))))
        if sample_frames_override is None:
            sample_frames = list(range(start_frame, end_frame + 1, frame_step))
        else:
            sample_frames = sorted(
                {
                    int(frame_idx)
                    for frame_idx in sample_frames_override
                    if start_frame <= int(frame_idx) <= end_frame
                }
            )
            frame_step = _median_frame_delta(sample_frames)
        if not sample_frames:
            raise RuntimeError("No sampled frames selected for wall map extraction.")

        yaw_source = str(wall_cfg.get("yaw_source", "imu_row_compass"))
        yaw_sign = float(wall_cfg.get("yaw_sign", -1.0))
        if yaw_source == "imu_compass":
            imu_yaw_rows = build_imu_yaw_rows(imu_rows, fps, "compass")
        elif yaw_source == "imu_gyro":
            imu_yaw_rows = build_imu_yaw_rows(imu_rows, fps, "gyro")
        else:
            imu_yaw_rows = []

        camera_altitude_m, _altitude_source = _camera_altitude(path_rows, imu_rows, wall_cfg, skip_seconds)
        pixels_per_meter = float(wall_cfg.get("pixels_per_meter", 140.0))
        footprint_width_m, footprint_height_m = camera_footprint_from_fov(
            camera_altitude_m,
            float(wall_cfg.get("camera_horizontal_fov_deg", DEFAULT_HORIZONTAL_FOV_DEG)),
            float(wall_cfg.get("camera_vertical_fov_deg", DEFAULT_VERTICAL_FOV_DEG)),
        )
        footprint_width_m *= float(wall_cfg.get("footprint_scale", 0.85))
        footprint_height_m *= float(wall_cfg.get("footprint_scale", 0.85))
        paste_w = max(1, int(round(footprint_width_m * pixels_per_meter)))
        paste_h = max(1, int(round(footprint_height_m * pixels_per_meter)))

        source_aspect = video_w / max(video_h, 1)
        footprint_aspect = paste_w / max(paste_h, 1)
        if abs(source_aspect - footprint_aspect) > 0.05:
            paste_h = max(1, int(round(paste_w / source_aspect)))
            footprint_height_m = paste_h / pixels_per_meter

        center_crop_fraction = float(wall_cfg.get("center_crop_fraction", 0.75))
        canvas_w, canvas_h, origin_px, _canvas_bounds_info = compute_canvas_from_sampled_frames(
            path_rows=path_rows,
            sample_frames=sample_frames,
            paste_w=paste_w,
            paste_h=paste_h,
            pixels_per_meter=pixels_per_meter,
            centre_crop_fraction=center_crop_fraction,
            yaw_source=yaw_source,
            imu_rows=imu_rows,
            imu_yaw_rows=imu_yaw_rows,
            yaw_sign=yaw_sign,
            safety_padding_px=int(wall_cfg.get("canvas_safety_padding_px", 300)),
        )

        metadata = load_model_metadata(model_path)
        model_input_width, model_input_height, _model_input_source = infer_model_input_wh(
            metadata=metadata,
            cli_width=_optional_int(wall_cfg.get("model_input_width")),
            cli_height=_optional_int(wall_cfg.get("model_input_height")),
            cli_square_size=_optional_int(wall_cfg.get("model_input_size")),
        )
        device = default_device(str(wall_cfg.get("device", "auto")))
        processor, model, _processor_source = load_segformer(model_path, device)
        wall_class_id = resolve_wall_class_id(
            model,
            _optional_int(wall_cfg.get("wall_class_id")),
            str(wall_cfg.get("wall_label", "wall")),
        )
        vote_mode = str(wall_cfg.get("vote_mode", "floor_contact")).strip().lower()
        if vote_mode not in ("floor_contact", "full_wall"):
            raise RuntimeError(f"Unknown walls.vote_mode={vote_mode!r}; use 'floor_contact' or 'full_wall'.")

        wall_sum = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        obs_weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
        debug_prefix = None
        frame_debug_dir = None
        save_frame_debug = (
            debug_dir is not None
            and bool(wall_cfg.get("save_frame_debug", False))
        )
        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_prefix = debug_dir / f"{world}_segformer_wall"
            if save_frame_debug:
                frame_debug_dir = debug_dir / f"{world}_segformer_frame_debug"
                frame_debug_dir.mkdir(parents=True, exist_ok=True)

        print(
            "[mission] wall map: "
            f"{len(sample_frames)} frames, canvas {canvas_w}x{canvas_h}, "
            f"model input {model_input_width}x{model_input_height}, device {device}, "
            f"vote mode {vote_mode}"
        )
        progress = ProgressBar("wall map frames", len(sample_frames))
        used_frames = 0
        saved_debug_frames = 0
        max_frame_debug = int(wall_cfg.get("max_frame_debug", 8))

        for frame_idx in sample_frames:
            frame = read_cached_frame(frame_cache, frame_idx)
            if frame is None:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ok, frame = cap.read()
                if not ok:
                    continue

            x_m = interp_path(path_rows, frame_idx, "x_m")
            y_m = interp_path(path_rows, frame_idx, "y_m")
            yaw_rad = yaw_at_frame(frame_idx, path_rows, imu_rows, imu_yaw_rows, yaw_source)
            unrotate_rad = -yaw_sign * yaw_rad
            wall_prob = infer_wall_probability(
                frame_bgr=frame,
                processor=processor,
                model=model,
                device=device,
                wall_class_id=wall_class_id,
                model_input_width=model_input_width,
                model_input_height=model_input_height,
                use_fp16=bool(wall_cfg.get("fp16", True)),
            )

            vote_prob = wall_prob
            contact_edge_binary: np.ndarray | None = None
            if vote_mode == "floor_contact":
                vote_prob, contact_edge_binary = extract_contact_wall_edge_probability(
                    wall_prob=wall_prob,
                    wall_threshold=float(wall_cfg.get("frame_wall_threshold", 0.55)),
                    band_px=int(wall_cfg.get("contact_edge_band_px", 5)),
                    close_px=int(wall_cfg.get("contact_edge_close_px", 2)),
                    min_component_area_px=int(wall_cfg.get("contact_edge_min_component_area_px", 80)),
                    contact_direction=str(wall_cfg.get("contact_edge_direction", "toward_center")),
                    line_close_px=int(wall_cfg.get("contact_line_close_px", 4)),
                )

            vote_prob_metric = cv2.resize(vote_prob, (paste_w, paste_h), interpolation=cv2.INTER_LINEAR)
            vote_prob_aligned, valid_mask = rotate_bound_float_with_mask(vote_prob_metric, unrotate_rad)
            vote_prob_aligned, valid_mask = crop_center_fraction_2d(
                vote_prob_aligned,
                valid_mask,
                center_crop_fraction,
            )
            placement_center = world_to_canvas(x_m, y_m, origin_px, pixels_per_meter)
            paste_wall_probability(wall_sum, obs_weight, vote_prob_aligned, placement_center, valid_mask)

            if (
                save_frame_debug
                and frame_debug_dir is not None
                and saved_debug_frames < max_frame_debug
            ):
                save_float_png(frame_debug_dir / f"frame_{frame_idx:06d}_wall_prob.png", wall_prob)
                if contact_edge_binary is not None:
                    cv2.imwrite(
                        str(frame_debug_dir / f"frame_{frame_idx:06d}_full_wall_binary.png"),
                        ((wall_prob >= float(wall_cfg.get("frame_wall_threshold", 0.55))).astype(np.uint8) * 255),
                    )
                    cv2.imwrite(
                        str(frame_debug_dir / f"frame_{frame_idx:06d}_contact_edge_binary.png"),
                        (contact_edge_binary.astype(np.uint8) * 255),
                    )
                    save_float_png(frame_debug_dir / f"frame_{frame_idx:06d}_contact_edge_prob.png", vote_prob)
                save_float_png(frame_debug_dir / f"frame_{frame_idx:06d}_vote_prob_aligned.png", vote_prob_aligned)
                saved_debug_frames += 1

            used_frames += 1
            progress.update(used_frames)

        progress.finish()
        if used_frames == 0:
            raise RuntimeError("No frames were processed for wall map extraction.")

        highres_prob = make_probability_image(wall_sum, obs_weight)
        if debug_prefix is not None:
            observed_mask = obs_weight >= float(
                wall_cfg.get("min_observations", 1.0)
            )
            cropped_prob, _crop_info = crop_float_to_observed(
                highres_prob,
                observed_mask,
                int(wall_cfg.get("debug_crop_padding_px", 160)),
            )
            cropped_obs, _ = crop_float_to_observed(
                (obs_weight > 0).astype(np.float32),
                observed_mask,
                int(wall_cfg.get("debug_crop_padding_px", 160)),
            )
            save_float_png(
                debug_prefix.with_name(
                    debug_prefix.name + "_highres_prob_cropped.png"
                ),
                cropped_prob,
            )
            save_float_png(
                debug_prefix.with_name(
                    debug_prefix.name + "_highres_observed_cropped.png"
                ),
                cropped_obs,
            )

        output_size_px = int(wall_cfg.get("output_size_px", 600))
        output_resolution_m = float(wall_cfg.get("output_resolution_m", 0.05))
        project_prob = highres_prob
        preproject_dilate_px = int(wall_cfg.get("preproject_dilate_px", 0))
        if preproject_dilate_px > 0:
            k = preproject_dilate_px * 2 + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
            project_prob = cv2.dilate(highres_prob.astype(np.float32), kernel)

        projection_mode = str(wall_cfg.get("projection_mode", "max" if vote_mode == "floor_contact" else "center"))
        projection_supersample = int(wall_cfg.get("projection_supersample", 7 if projection_mode == "max" else 1))
        projection_center_m, projection_center_info = _estimate_projection_center_m(
            highres_prob=project_prob,
            highres_obs=obs_weight,
            origin_px=origin_px,
            pixels_per_meter=pixels_per_meter,
            threshold=float(wall_cfg.get("projection_center_threshold", wall_cfg.get("wall_threshold", 0.55))),
            min_observations=float(wall_cfg.get("min_observations", 1.0)),
            enabled=bool(wall_cfg.get("center_projection_on_walls", True)),
        )
        final_prob, final_obs = project_highres_to_final(
            highres_prob=project_prob,
            highres_obs=obs_weight,
            origin_px=origin_px,
            highres_pixels_per_meter=pixels_per_meter,
            output_size_px=output_size_px,
            output_resolution_m=output_resolution_m,
            projection_mode=projection_mode,
            projection_supersample=projection_supersample,
            center_world_m=projection_center_m,
        )
        final_observed = final_obs >= float(wall_cfg.get("min_observations", 1.0))
        final_wall_raw = (final_prob >= float(wall_cfg.get("wall_threshold", 0.55))) & final_observed
        final_wall_clean = clean_wall_binary(
            wall_binary=final_wall_raw,
            close_px=int(wall_cfg.get("close_px", 1)),
            open_px=int(wall_cfg.get("open_px", 0)),
            dilate_px=int(wall_cfg.get("dilate_px", 1)),
            min_wall_area_px=int(wall_cfg.get("min_wall_area_px", 12)),
        )
        output_orientation = str(wall_cfg.get("output_orientation", "origin"))
        if output_orientation.strip().lower() not in ("origin", "origin_xy", "navigation"):
            raise RuntimeError(
                "Robot map info export requires walls.output_orientation to be "
                "'origin', 'origin_xy', or 'navigation'."
            )
        exported_wall = orient_wall_binary_for_supervisor(final_wall_clean, output_orientation)
        center_map_on_walls = bool(wall_cfg.get("center_map_on_walls", True))
        center_info: dict[str, Any] = {"applied": 0, "reason": "disabled"}
        if center_map_on_walls:
            exported_wall, center_info = center_wall_binary_bbox(exported_wall)

        drone_clear_info: dict[str, Any] = {"applied": 0, "reason": "disabled"}
        if bool(wall_cfg.get("clear_drone_path_walls", True)):
            exported_wall, drone_clear_info = _clear_drone_path_from_wall_binary(
                wall_binary=exported_wall,
                path_rows=path_rows,
                center_world_m=projection_center_m,
                center_shift_info=center_info,
                resolution_m=output_resolution_m,
                canvas_size_px=output_size_px,
                clear_radius_m=float(wall_cfg.get("drone_path_clear_radius_m", 0.20)),
            )

        final_map = make_black_white_map(exported_wall)
        if debug_prefix is not None:
            save_float_png(
                debug_prefix.with_name(debug_prefix.name + "_final_prob.png"),
                final_prob,
            )
            cv2.imwrite(
                str(
                    debug_prefix.with_name(
                        debug_prefix.name + "_final_wall_raw.png"
                    )
                ),
                make_black_white_map(final_wall_raw),
            )
            cv2.imwrite(
                str(
                    debug_prefix.with_name(
                        debug_prefix.name
                        + "_final_wall_clean_origin_orientation.png"
                    )
                ),
                make_black_white_map(final_wall_clean),
            )
            cv2.imwrite(
                str(
                    debug_prefix.with_name(
                        debug_prefix.name
                        + "_final_wall_drone_path_cleared.png"
                    )
                ),
                final_map,
            )
            cv2.imwrite(
                str(
                    debug_prefix.with_name(
                        debug_prefix.name + "_final_wall_clean.png"
                    )
                ),
                final_map,
            )
        ensure_parent(map_output)
        cv2.imwrite(str(map_output), final_map)
        enforce_official_map(map_output, output_size_px)

        info = _build_robot_map_info(
            wall_binary=exported_wall,
            center_world_m=projection_center_m,
            center_shift_info=center_info,
            resolution_m=output_resolution_m,
            canvas_size_px=output_size_px,
        )
        info["drone_path_wall_clear"] = drone_clear_info
        ensure_parent(info_output).write_text(json.dumps(info, indent=2) + "\n")
        return {
            "map_output": str(map_output),
            "info": str(info_output),
            "run_label": world,
            "used_frames": int(used_frames),
        }
    finally:
        cap.release()


def _build_robot_map_info(
    wall_binary: np.ndarray,
    center_world_m: tuple[float, float],
    center_shift_info: dict[str, Any],
    resolution_m: float,
    canvas_size_px: int,
) -> dict[str, Any]:
    image_x_sign = -1.0
    image_y_sign = -1.0
    pixels_per_metre = 1.0 / resolution_m
    dx_px = float(center_shift_info.get("dx_px", 0.0) or 0.0)
    dy_px = float(center_shift_info.get("dy_px", 0.0) or 0.0)

    center_x_m = float(center_world_m[0]) - dy_px * resolution_m / image_y_sign
    center_y_m = float(center_world_m[1]) - dx_px * resolution_m / image_x_sign
    x_min_m, x_max_m, y_min_m, y_max_m = _wall_world_bounds(
        wall_binary=wall_binary,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        canvas_size_px=canvas_size_px,
        resolution_m=resolution_m,
        image_x_sign=image_x_sign,
        image_y_sign=image_y_sign,
    )
    origin_px = _world_to_pixel(
        x_m=0.0,
        y_m=0.0,
        center_x_m=center_x_m,
        center_y_m=center_y_m,
        canvas_size_px=canvas_size_px,
        pixels_per_metre=pixels_per_metre,
        image_x_sign=image_x_sign,
        image_y_sign=image_y_sign,
    )

    return {
        "resolution": float(resolution_m),
        "pixels_per_metre": float(pixels_per_metre),
        "canvas_size": int(canvas_size_px),
        "center_x_m": float(center_x_m),
        "center_y_m": float(center_y_m),
        "x_min_m": float(x_min_m),
        "x_max_m": float(x_max_m),
        "y_min_m": float(y_min_m),
        "y_max_m": float(y_max_m),
        "image_x_sign": image_x_sign,
        "image_y_sign": image_y_sign,
        "world_frame": {
            "x_axis": "positive forward/front",
            "y_axis": "positive left",
            "origin": "drone start pose",
        },
        "image_frame": {
            "px_axis": "positive right",
            "py_axis": "positive down",
        },
        "world_to_pixel_formula": {
            "px": "canvas_size/2 + image_x_sign * (y_m - center_y_m) * pixels_per_metre",
            "py": "canvas_size/2 + image_y_sign * (x_m - center_x_m) * pixels_per_metre",
        },
        "pixel_to_world_formula": {
            "x_m": "center_x_m + ((py - canvas_size/2) * resolution) / image_y_sign",
            "y_m": "center_y_m + ((px - canvas_size/2) * resolution) / image_x_sign",
        },
        "origin_px": [int(origin_px[0]), int(origin_px[1])],
    }


def _clear_drone_path_from_wall_binary(
    wall_binary: np.ndarray,
    path_rows: list[dict[str, float]],
    center_world_m: tuple[float, float],
    center_shift_info: dict[str, Any],
    resolution_m: float,
    canvas_size_px: int,
    clear_radius_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if not path_rows:
        return wall_binary, {"applied": 0, "reason": "no_path_rows"}
    radius_px = max(0, int(round(float(clear_radius_m) / float(resolution_m))))
    if radius_px <= 0:
        return wall_binary, {
            "applied": 0,
            "reason": "non_positive_clear_radius",
            "clear_radius_m": float(clear_radius_m),
        }

    image_x_sign = -1.0
    image_y_sign = -1.0
    dx_px = float(center_shift_info.get("dx_px", 0.0) or 0.0)
    dy_px = float(center_shift_info.get("dy_px", 0.0) or 0.0)
    center_x_m = float(center_world_m[0]) - dy_px * resolution_m / image_y_sign
    center_y_m = float(center_world_m[1]) - dx_px * resolution_m / image_x_sign
    pixels_per_metre = 1.0 / resolution_m

    points: list[tuple[int, int]] = []
    for row in path_rows:
        try:
            point = _world_to_pixel(
                x_m=float(row["x_m"]),
                y_m=float(row["y_m"]),
                center_x_m=center_x_m,
                center_y_m=center_y_m,
                canvas_size_px=canvas_size_px,
                pixels_per_metre=pixels_per_metre,
                image_x_sign=image_x_sign,
                image_y_sign=image_y_sign,
            )
        except (KeyError, TypeError, ValueError):
            continue
        if not points or points[-1] != point:
            points.append(point)

    if not points:
        return wall_binary, {"applied": 0, "reason": "no_valid_path_points"}

    clear_mask = np.zeros(wall_binary.shape[:2], dtype=np.uint8)
    thickness_px = max(1, 2 * radius_px + 1)
    if len(points) == 1:
        cv2.circle(clear_mask, points[0], radius_px, 1, thickness=-1)
    else:
        pts = np.array(points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(clear_mask, [pts], isClosed=False, color=1, thickness=thickness_px)
        cv2.circle(clear_mask, points[0], radius_px, 1, thickness=-1)
        cv2.circle(clear_mask, points[-1], radius_px, 1, thickness=-1)

    cleared = wall_binary.copy()
    before = int(np.count_nonzero(cleared))
    cleared[clear_mask.astype(bool)] = False
    removed = before - int(np.count_nonzero(cleared))
    return cleared, {
        "applied": 1,
        "clear_radius_m": float(clear_radius_m),
        "clear_radius_px": int(radius_px),
        "path_points": int(len(points)),
        "removed_wall_pixels": int(max(0, removed)),
    }


def _world_to_pixel(
    x_m: float,
    y_m: float,
    center_x_m: float,
    center_y_m: float,
    canvas_size_px: int,
    pixels_per_metre: float,
    image_x_sign: float,
    image_y_sign: float,
) -> tuple[int, int]:
    px = (
        canvas_size_px / 2.0
        + image_x_sign * (y_m - center_y_m) * pixels_per_metre
    )
    py = (
        canvas_size_px / 2.0
        + image_y_sign * (x_m - center_x_m) * pixels_per_metre
    )
    return int(round(px)), int(round(py))


def _wall_world_bounds(
    wall_binary: np.ndarray,
    center_x_m: float,
    center_y_m: float,
    canvas_size_px: int,
    resolution_m: float,
    image_x_sign: float,
    image_y_sign: float,
) -> tuple[float, float, float, float]:
    wall_ys, wall_xs = np.where(wall_binary)
    if len(wall_xs) == 0 or len(wall_ys) == 0:
        wall_xs = np.array([0, canvas_size_px - 1], dtype=np.float64)
        wall_ys = np.array([0, canvas_size_px - 1], dtype=np.float64)

    centre = canvas_size_px / 2.0
    wall_ys_f = wall_ys.astype(np.float64)
    wall_xs_f = wall_xs.astype(np.float64)
    x_values = center_x_m + ((wall_ys_f - centre) * resolution_m) / image_y_sign
    y_values = center_y_m + ((wall_xs_f - centre) * resolution_m) / image_x_sign
    return (
        float(np.min(x_values)),
        float(np.max(x_values)),
        float(np.min(y_values)),
        float(np.max(y_values)),
    )


def wall_sample_frames(
    video_fps: float,
    frame_count: int,
    wall_cfg: dict[str, Any],
    source_frames: list[int] | None = None,
) -> list[int]:
    sample_rate = float(wall_cfg.get("sample_rate", 1.5))
    skip_seconds = float(wall_cfg.get("skip_seconds", 2.5))
    end_seconds_value = wall_cfg.get("end_seconds")
    end_seconds = None if end_seconds_value is None else float(end_seconds_value)
    start_frame = int(round(skip_seconds * video_fps))
    end_frame = frame_count - 1 if end_seconds is None else min(frame_count - 1, int(round(end_seconds * video_fps)))

    frame_source = str(wall_cfg.get("frame_source", "victim")).strip().lower()
    if source_frames is not None and frame_source in ("victim", "shared", "victim_subset"):
        stride = max(1, int(wall_cfg.get("victim_frame_stride", wall_cfg.get("shared_frame_stride", 2))))
        eligible = [
            int(frame_idx)
            for frame_idx in source_frames
            if start_frame <= int(frame_idx) <= end_frame
        ]
        return eligible[::stride]

    frame_step = max(1, int(round(video_fps / max(sample_rate, 0.001))))
    return list(range(start_frame, end_frame + 1, frame_step))


def _median_frame_delta(frames: list[int]) -> int:
    if len(frames) < 2:
        return 0
    deltas = np.diff(np.array(frames, dtype=np.int64))
    return int(round(float(np.median(deltas))))


def _load_path_csv(path_csv: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"frame", "x_m", "y_m"}
        missing = required - fieldnames
        if missing:
            raise RuntimeError(f"Path CSV is missing columns {sorted(missing)} in {path_csv}")

        for row in reader:
            timestamp_text = row.get("timestamp") or row.get("time_s") or "0"
            parsed = {
                "frame": float(row["frame"]),
                "timestamp": float(timestamp_text),
                "x_m": float(row["x_m"]),
                "y_m": float(row["y_m"]),
                "yaw_rad": float(row.get("yaw_rad", 0.0) or 0.0),
            }
            for optional_key in ("altitude_m", "aruco_height_m", "aruco_x_m", "aruco_y_m"):
                value = row.get(optional_key)
                if value not in (None, ""):
                    parsed[optional_key] = float(value)
            rows.append(parsed)

    if not rows:
        raise RuntimeError(f"No path rows found in {path_csv}")
    return rows


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _camera_altitude(
    path_rows: list[dict[str, float]],
    imu_rows: list[dict[str, float]],
    wall_cfg: dict[str, Any],
    skip_seconds: float,
) -> tuple[float, str]:
    configured = wall_cfg.get("camera_altitude_m")
    if configured is not None:
        return float(configured), "config"

    source_mode = normalise_altitude_mode(wall_cfg.get("altitude_mode", wall_cfg.get("camera_altitude_source", "aruco")))
    estimated = estimate_altitude_from_path(path_rows, skip_seconds)
    if estimated is not None:
        return estimated, "path_csv_altitude_m_median"

    if source_mode == "imu":
        imu_estimate = estimate_altitude_from_imu_csv(imu_rows, wall_cfg, skip_seconds)
        if imu_estimate is not None:
            altitude_m, detail = imu_estimate
            return altitude_m, detail
        fallback = float(wall_cfg.get("fallback_camera_altitude_m", 2.5))
        return fallback, "fallback_camera_altitude_m_imu_csv_unavailable"

    fallback = float(wall_cfg.get("fallback_camera_altitude_m", 2.5))
    return fallback, "fallback_camera_altitude_m"


def _estimate_projection_center_m(
    highres_prob: np.ndarray,
    highres_obs: np.ndarray,
    origin_px: tuple[float, float],
    pixels_per_meter: float,
    threshold: float,
    min_observations: float,
    enabled: bool,
) -> tuple[tuple[float, float], dict[str, Any]]:
    if not enabled:
        return (0.0, 0.0), {"source": "origin", "enabled": 0}

    evidence = (highres_prob >= threshold) & (highres_obs >= min_observations)
    ys, xs = np.where(evidence)
    if len(xs) == 0 or len(ys) == 0:
        return (
            (0.0, 0.0),
            {
                "source": "origin",
                "enabled": 1,
                "reason": "no_wall_evidence",
                "threshold": float(threshold),
            },
        )

    bbox_cx_px = (float(xs.min()) + float(xs.max())) / 2.0
    bbox_cy_px = (float(ys.min()) + float(ys.max())) / 2.0
    center_y_m = (float(origin_px[0]) - bbox_cx_px) / pixels_per_meter
    center_x_m = (float(origin_px[1]) - bbox_cy_px) / pixels_per_meter
    return (
        (center_x_m, center_y_m),
        {
            "source": "highres_wall_bbox",
            "enabled": 1,
            "threshold": float(threshold),
            "bbox_px": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
            "bbox_center_px": [float(bbox_cx_px), float(bbox_cy_px)],
        },
    )
