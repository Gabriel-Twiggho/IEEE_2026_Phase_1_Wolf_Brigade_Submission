#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from extraction.debug_viz import ensure_debug_dir
from extraction.drone_path import build_drone_path
from extraction.io_utils import (
    DEFAULT_CONFIG_PATH,
    configured_path,
    infer_run_label,
    load_pipeline_config,
    output_path,
    resolve_recording_inputs,
)
from extraction.map_builder import build_wall_map
from extraction.map_builder import wall_sample_frames
from extraction.progress import ProgressBar
from extraction.victim_locator import detect_victims
from extraction.victim_locator import victim_sample_frames
from extraction.victim_locator import write_official_victim_estimates
from extraction.video_reader import extract_frame_memory_cache
from extraction.video_reader import read_video_info


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the pre-mission flyover extraction pipeline.",
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help=(
            "Flyover video filename or path. Examples: "
            "recordings/YOUR_VIDEO_FILE.mp4, "
            "/recordings/YOUR_VIDEO_FILE.mp4, or an explicit .mp4 path."
        ),
    )
    parser.add_argument(
        "--imu",
        type=str,
        default=None,
        help="Optional IMU CSV path; defaults to the CSV beside the resolved video.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Optional pipeline config path; relative paths use the current directory.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Write diagnostic plots, detection details, and wall-map intermediates.",
    )
    return parser.parse_args(argv)


def _print_reload_banner() -> None:
    title = "MISSION EXTRACTION COMPLETE"
    instruction = "Please reload the world and run the Simulation"
    width = max(len(title), len(instruction)) + 4
    border = "+" + "-" * (width - 2) + "+"
    print(f"\n{border}")
    print(f"| {title.center(width - 4)} |")
    print(f"| {instruction.center(width - 4)} |")
    print(border)


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_pipeline_config(config_path)

    try:
        video, imu = resolve_recording_inputs(args.video, args.imu)
    except RuntimeError as exc:
        raise SystemExit(f"mission_extraction.py: error: {exc}") from None

    run_label = infer_run_label(video, config)
    video_info = read_video_info(video)
    debug_dir = (
        ensure_debug_dir(configured_path(config, "debug"))
        if args.debug
        else None
    )

    intrinsics = configured_path(config, "camera_intrinsics")
    drone_model = configured_path(config, "drone_victim_model")
    wall_model = configured_path(config, "wall_model")

    if not intrinsics.exists():
        raise RuntimeError(f"Camera intrinsics not found: {intrinsics}")
    if not drone_model.exists():
        raise RuntimeError(f"Drone victim model not found: {drone_model}")
    if not wall_model.exists():
        raise RuntimeError(f"Wall model folder not found: {wall_model}")

    path_csv = output_path(config, "drone_path_csv")
    path_plot = output_path(config, "drone_path_plot") if args.debug else None
    victims_csv = output_path(config, "victim_estimates")
    detailed_victims_csv = (
        output_path(config, "detailed_victims_csv") if args.debug else None
    )
    detailed_victims_json = (
        output_path(config, "detailed_victims_json") if args.debug else None
    )
    victim_plot = output_path(config, "victim_plot") if args.debug else None
    map_output = output_path(config, "map_estimate")
    map_info = output_path(config, "map_estimate_info")

    print(f"[mission] video: {video}")
    print(f"[mission] imu: {imu}")
    print(f"[mission] run label: {run_label}")
    print(
        "[mission] video info: "
        f"{video_info.width}x{video_info.height}, "
        f"{video_info.frame_count} frames at {video_info.fps:.3f} fps"
    )

    mission_progress = ProgressBar("mission stages", 4)
    print("[mission] stage 1/4: drone path")
    path_rows, aruco_samples, _path_summary = build_drone_path(
        video=video,
        imu=imu,
        intrinsics=intrinsics,
        world=run_label,
        config=config,
        output_csv=path_csv,
        plot_path=path_plot,
    )
    mission_progress.update(1, force=True)

    print("\n[mission] stage 2/4: shared frame extraction")
    victim_frame_cfg = dict(config.get("victims", {}))
    victim_frame_cfg["_video_fps"] = video_info.fps
    victim_frames = victim_sample_frames(video_info.frame_count, victim_frame_cfg)
    wall_frames = wall_sample_frames(
        video_info.fps,
        video_info.frame_count,
        config.get("walls", {}),
        source_frames=victim_frames,
    )
    shared_frames = sorted(
        set(victim_frames)
        | set(wall_frames)
    )
    print(
        "[mission] shared frames: "
        f"{len(shared_frames)} cached "
        f"({len(victim_frames)} victim, {len(wall_frames)} wall, "
        f"{len(set(victim_frames) & set(wall_frames))} overlap)"
    )
    frame_cache = extract_frame_memory_cache(video, shared_frames)
    mission_progress.update(2, force=True)

    annotated_dir = None
    if (
        debug_dir is not None
        and bool(config.get("victims", {}).get("save_annotated_frames", False))
    ):
        annotated_dir = debug_dir / "victim_frames"

    print("\n[mission] stage 3/4: victim detection")
    victims = detect_victims(
        video=video,
        model_path=drone_model,
        path_rows=path_rows,
        aruco_samples=aruco_samples,
        intrinsics=intrinsics,
        config=config,
        detailed_csv=detailed_victims_csv,
        detailed_json=detailed_victims_json,
        plot_path=victim_plot,
        annotated_dir=annotated_dir,
        frame_cache=frame_cache,
    )
    write_official_victim_estimates(victims, victims_csv)
    mission_progress.update(3, force=True)

    print("\n[mission] stage 4/4: wall map extraction")
    map_result = build_wall_map(
        video=video,
        imu=imu,
        path_csv=path_csv,
        model_path=wall_model,
        world=run_label,
        config=config,
        map_output=map_output,
        info_output=map_info,
        debug_dir=debug_dir,
        frame_cache=frame_cache,
        sample_frames_override=wall_frames,
    )
    mission_progress.update(4, force=True)
    mission_progress.finish()

    print(f"[mission] wrote victims: {victims_csv}")
    print(f"[mission] wrote map: {map_output}")
    print(f"[mission] wrote map info: {map_result['info']}")
    #Gabriel twigg-ho Krish
    _print_reload_banner()


if __name__ == "__main__":
    main()
