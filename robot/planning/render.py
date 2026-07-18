"""Lazy planner rendering and viewer helpers."""

from __future__ import annotations

import atexit
import datetime
import math
import os
import subprocess
import sys

from config import LIVE_MAP_VIEWER_PATH, SLAM_LITE_DIR
from parameters import RobotGeometry
from robot.mapping.drone_map import DroneExtractionMap
from shared_types import PlannedPath, Pose2D


def render_plan_overlay(planner, planned_path: PlannedPath):
    if not planner.ready or not planner.render_overlay_enabled:
        return

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        planner.error_message = f"PIL unavailable: {exc}"
        print(f"[{planner.robot_id}] Path planner render disabled: {planner.error_message}")
        return

    try:
        map_image = Image.open(planner.drone_map.map_path).convert("RGB")
    except OSError as exc:
        planner.error_message = str(exc)
        print(f"[{planner.robot_id}] Path planner render failed: {planner.error_message}")
        return

    panel_width = 370
    image = Image.new(
        "RGB",
        (map_image.width + panel_width, max(map_image.height, 320)),
        (245, 245, 245),
    )
    image.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((map_image.width, 0, map_image.width, image.height), fill=(70, 70, 70), width=2)
    _draw_planner_legend(planner, draw, planned_path, map_image.width, panel_width)
    _draw_robot_and_victims(planner, draw, planned_path)

    if planned_path is not None and planned_path.success:
        if planned_path.global_comparison_pixel_path:
            _draw_pixel_path(draw, planned_path.global_comparison_pixel_path, colour=(40, 170, 80), width=2)
        _draw_pixel_path(draw, planned_path.pixel_path)
        _draw_planner_waypoints(planner, draw, planned_path.waypoints)

    tmp_path = planner.output_path.with_suffix(".tmp.png")
    image.save(tmp_path)
    os.replace(tmp_path, planner.output_path)


def render_live_replanner(replanner, current_pose: Pose2D, refresh_base_map: bool = True):
    if not replanner.enabled or not replanner.render_enabled:
        return
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print(f"[{replanner.robot_id}] Live replan render disabled: {exc}")
        return

    if refresh_base_map:
        replanner.live_map.render()
    try:
        map_image = Image.open(replanner.live_map.output_path).convert("RGB")
    except OSError as exc:
        print(f"[{replanner.robot_id}] Live replan render failed: {exc}")
        return

    panel_width = replanner.INFO_PANEL_WIDTH_PX
    image = Image.new("RGB", (map_image.width + panel_width, max(map_image.height, 320)), (245, 245, 245))
    image.paste(map_image, (0, 0))
    draw = ImageDraw.Draw(image)
    draw.line((map_image.width, 0, map_image.width, image.height), fill=(70, 70, 70), width=2)
    remaining_path = replanner.planner.remaining_pixel_path(replanner.active_path, current_pose)
    if replanner.active_path is not None and replanner.active_path.global_comparison_pixel_path:
        _draw_path(draw, replanner.active_path.global_comparison_pixel_path, (40, 170, 80), width=2)
    _draw_path(draw, remaining_path, (255, 0, 255), width=3)
    if replanner.candidate_path is not None:
        if replanner.candidate_path.global_comparison_pixel_path:
            _draw_path(draw, replanner.candidate_path.global_comparison_pixel_path, (40, 170, 80), width=2)
        _draw_path(draw, replanner.candidate_path.pixel_path, (0, 220, 255), width=3)
        _draw_live_waypoints(replanner, draw, replanner.candidate_path.waypoints)
    elif getattr(replanner, "failed_candidate_path", None) is not None:
        failed_path = replanner.failed_candidate_path
        _draw_path(draw, failed_path.pixel_path, (0, 120, 255), width=2)
        _draw_live_waypoints(replanner, draw, failed_path.waypoints)

    for px, py in replanner.blocking_cells:
        draw.rectangle((px - 2, py - 2, px + 2, py + 2), fill=(255, 30, 30))

    victim_status = replanner._victim_status()
    _draw_victim_overlay(replanner, draw, victim_status)
    _draw_target(replanner, draw)
    _draw_robot_footprint(replanner, draw, current_pose)
    _draw_live_legend(replanner, draw, map_image.width, panel_width, victim_status)

    tmp_path = replanner.output_path.with_suffix(".tmp.png")
    image.save(tmp_path)
    os.replace(tmp_path, replanner.output_path)


def render_escape_debug(planner, current_pose: Pose2D, context, result_reason: str):
    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print(f"[{planner.robot_id}] Safety escape debug render disabled: {exc}")
        return

    fine_width = int(context["fine_width"])
    fine_height = int(context["fine_height"])
    cell_px = max(1, min(6, 900 // max(1, max(fine_width, fine_height))))
    patch_width_px = fine_width * cell_px
    patch_height_px = fine_height * cell_px
    panel_width_px = 410
    image = Image.new(
        "RGB",
        (patch_width_px + panel_width_px, max(patch_height_px, 380)),
        (245, 245, 245),
    )
    draw = ImageDraw.Draw(image)

    occupied = context["occupied"]
    clearance_m = context["clearance_m"]
    fine_free = context["fine_free"]
    goal_clearance_m = max(1e-6, float(planner.escape_goal_clearance_m))
    for fy in range(fine_height):
        y0 = fy * cell_px
        for fx in range(fine_width):
            x0 = fx * cell_px
            if bool(occupied[fy, fx]):
                colour = (20, 24, 28)
            else:
                clearance_ratio = min(1.0, float(clearance_m[fy, fx]) / goal_clearance_m)
                if bool(fine_free[fy, fx]):
                    base = int(220 + 25 * clearance_ratio)
                    colour = (base, base, 255)
                else:
                    base = int(235 - 55 * clearance_ratio)
                    colour = (255, base, 170)
            draw.rectangle((x0, y0, x0 + cell_px - 1, y0 + cell_px - 1), fill=colour)

    for fx, fy in context.get("cleared_start_cells", ()):
        draw.rectangle(
            (
                int(fx) * cell_px,
                int(fy) * cell_px,
                (int(fx) + 1) * cell_px - 1,
                (int(fy) + 1) * cell_px - 1,
            ),
            fill=(145, 255, 185),
        )

    start_clear_radius_m = float(context.get("start_clear_radius_m", 0.0))
    robot_cell = _escape_world_to_fine_cell(
        planner,
        current_pose.x,
        current_pose.y,
        context,
    )
    if robot_cell is not None and start_clear_radius_m > 0.0:
        centre_x, centre_y = _escape_cell_centre(robot_cell, cell_px)
        radius_px = start_clear_radius_m / planner.fine_resolution_m * cell_px
        draw.ellipse(
            (
                centre_x - radius_px,
                centre_y - radius_px,
                centre_x + radius_px,
                centre_y + radius_px,
            ),
            outline=(145, 255, 185),
            width=max(1, cell_px // 2),
        )

    start = context.get("start")
    if start is not None:
        _draw_debug_cell(draw, int(start[0]), int(start[1]), cell_px, (255, 150, 0), radius=4)

    _draw_escape_debug_states(draw, planner, context, cell_px)
    _draw_escape_debug_robot(draw, planner, current_pose, context, cell_px)
    draw.line(
        (patch_width_px, 0, patch_width_px, image.height),
        fill=(70, 70, 70),
        width=2,
    )
    _draw_escape_debug_panel(
        draw,
        patch_width_px + 16,
        planner,
        current_pose,
        context,
        result_reason,
        cell_px,
    )

    output_path = SLAM_LITE_DIR / f"{planner.robot_id}_safety_escape_debug.png"
    tmp_path = output_path.with_suffix(".tmp.png")
    image.save(tmp_path)
    os.replace(tmp_path, output_path)


def _draw_escape_debug_states(draw, planner, context, cell_px: int):
    states = context.get("states") or ()
    if len(states) >= 2:
        points = [_escape_cell_centre((state[0], state[1]), cell_px) for state in states]
        draw.line(points, fill=(255, 0, 255), width=3)
        for state in states:
            _draw_debug_cell(draw, state[0], state[1], cell_px, (255, 0, 255), radius=2)

    for waypoint in context.get("waypoints") or ():
        cell = _escape_world_to_fine_cell(planner, waypoint.x, waypoint.y, context)
        if cell is not None:
            _draw_debug_cell(draw, cell[0], cell[1], cell_px, (0, 220, 255), radius=3)


def _draw_escape_debug_robot(draw, planner, pose: Pose2D, context, cell_px: int):
    margin_m = float(context.get("footprint_margin_m", 0.0))
    half_length = 0.5 * RobotGeometry.LENGTH_M + margin_m
    half_width = 0.5 * RobotGeometry.WIDTH_M + margin_m
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    corners = []
    for forward_m, lateral_m in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        wx = pose.x + cos_yaw * forward_m - sin_yaw * lateral_m
        wy = pose.y + sin_yaw * forward_m + cos_yaw * lateral_m
        cell = _escape_world_to_fine_cell(planner, wx, wy, context)
        if cell is not None:
            corners.append(_escape_cell_centre(cell, cell_px))
    if len(corners) == 4:
        draw.line(corners + [corners[0]], fill=(0, 210, 70), width=3)

    centre = _escape_world_to_fine_cell(planner, pose.x, pose.y, context)
    front = _escape_world_to_fine_cell(
        planner,
        pose.x + half_length * math.cos(pose.yaw),
        pose.y + half_length * math.sin(pose.yaw),
        context,
    )
    if centre is not None:
        _draw_debug_cell(draw, centre[0], centre[1], cell_px, (0, 255, 80), radius=5)
        if front is not None:
            draw.line(
                (*_escape_cell_centre(centre, cell_px), *_escape_cell_centre(front, cell_px)),
                fill=(0, 140, 60),
                width=3,
            )

    for forward_m, lateral_m in context.get("footprint_points", ()):
        wx = pose.x + cos_yaw * forward_m - sin_yaw * lateral_m
        wy = pose.y + sin_yaw * forward_m + cos_yaw * lateral_m
        cell = _escape_world_to_fine_cell(planner, wx, wy, context)
        if cell is not None:
            cx, cy = _escape_cell_centre(cell, cell_px)
            draw.point((cx, cy), fill=(20, 100, 255))


def _draw_escape_debug_panel(draw, x: int, planner, pose: Pose2D, context, result_reason: str, cell_px: int):
    y = 16
    cleared_count = len(context.get("cleared_start_cells", ()))
    state_count = len(context.get("states") or ())
    waypoint_count = len(context.get("waypoints") or ())
    text_lines = (
        "Safety escape debug",
        f"robot={planner.robot_id}",
        f"generated={datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"result={result_reason}",
        f"patch={context.get('patch_size_m', 0.0):.2f} m  zoom={cell_px}px/cell",
        f"pose x={pose.x:.3f} y={pose.y:.3f} yaw={math.degrees(pose.yaw):+.1f} deg",
        f"yaw_bins={context.get('yaw_bins')}  max_expansions={context.get('max_expansions')}",
        f"step={context.get('primitive_step_m', 0.0):.3f} m  arc_r={context.get('primitive_arc_radius_m', 0.0):.3f} m",
        f"rotate={math.degrees(context.get('rotate_step_rad', 0.0)):.1f} deg",
        f"robot={RobotGeometry.LENGTH_M:.3f}x{RobotGeometry.WIDTH_M:.3f} m",
        f"footprint_margin={context.get('footprint_margin_m', 0.0):.3f} m",
        f"start_clear_radius={context.get('start_clear_radius_m', 0.0):.3f} m",
        f"cleared_start_cells={cleared_count}",
        f"escape_states={state_count}",
        f"escape_waypoints={waypoint_count}",
        f"goal_clearance={planner.escape_goal_clearance_m:.3f} m",
        f"patch_reason={context.get('failure_reason', '-')}",
        "",
        "black = occupied",
        "blue = fine-grid free cells",
        "orange = below fine clearance",
        "pale green = cleared cells/start circle",
        "green = current footprint/heading",
        "magenta = escape state path",
        "cyan = escape waypoints",
        "orange dot = snapped start cell",
    )
    for line in text_lines:
        draw.text((x, y), line, fill=(0, 0, 0))
        y += 20


def _escape_world_to_fine_cell(planner, x_m: float, y_m: float, context):
    px, py = planner.geometry.world_to_pixel_float(x_m, y_m)
    min_px = int(context["min_px"])
    min_py = int(context["min_py"])
    scale = int(context["scale"])
    fx = int(round((px - min_px + 0.5) * scale - 0.5))
    fy = int(round((py - min_py + 0.5) * scale - 0.5))
    if not (0 <= fx < int(context["fine_width"]) and 0 <= fy < int(context["fine_height"])):
        return None
    return fx, fy


def _escape_cell_centre(cell, cell_px: int):
    return (
        int(cell[0]) * cell_px + cell_px // 2,
        int(cell[1]) * cell_px + cell_px // 2,
    )


def _draw_debug_cell(draw, fx: int, fy: int, cell_px: int, colour, radius: int):
    cx, cy = _escape_cell_centre((fx, fy), cell_px)
    radius = max(radius, cell_px)
    draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline=(0, 0, 0))


def start_viewer(image_path, title: str, error_prefix: str):
    script = LIVE_MAP_VIEWER_PATH
    if not script.exists():
        return None
    try:
        process = subprocess.Popen(
            [sys.executable, str(script), str(image_path), title],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        atexit.register(stop_viewer, process)
        return process
    except OSError as exc:
        print(f"{error_prefix}: {exc}")
        return None


def stop_viewer(process):
    if process and process.poll() is None:
        process.terminate()


def _draw_planner_legend(planner, draw, planned_path: PlannedPath, panel_x, panel_width):
    x = panel_x + 16
    y = 16
    draw.text((x, y), "A* path planner", fill=(0, 0, 0))
    y += 22
    draw.text((x, y), f"global inflation={planner.inflation_radius_px} px ({planner.inflation_radius_m:.3f} m)", fill=(0, 0, 0))
    y += 24
    if planned_path is not None and planned_path.success:
        for text in (
            f"mode={planned_path.planning_mode}  victim=v{planned_path.victim_index}",
            f"path_px={len(planned_path.pixel_path)} wp={len(planned_path.waypoints)}",
            f"physical={planned_path.physical_path_cost_m:.2f} m  weighted={planned_path.weighted_path_cost_m:.2f} m",
            f"planning={planned_path.planning_time_s:.3f} s",
        ):
            draw.text((x, y), text, fill=(0, 0, 0))
            y += 20
        y += 4
        for text in (
            "green = ordinary wide route",
            "magenta = selected route",
            "cyan = waypoints",
        ):
            draw.text((x, y), text, fill=(0, 0, 0))
            y += 18
    else:
        reason = planned_path.error_reason if planned_path is not None else "not planned"
        reason_width = max(24, (panel_width - 32) // 7)
        reason_lines = [reason[index:index + reason_width] for index in range(0, len(reason), reason_width)] or ["not planned"]
        draw.multiline_text(
            (x, y),
            "failed:\n" + "\n".join(reason_lines[:5]),
            fill=(180, 0, 0),
            spacing=3,
        )


def _draw_robot_and_victims(planner, draw, planned_path: PlannedPath):
    start_pose = DroneExtractionMap.ROBOT_STARTS["robot1"]
    start_px = planner.geometry.world_to_pixel(start_pose.x, start_pose.y)
    _draw_circle(draw, start_px[0], start_px[1], (0, 220, 0), radius=6)
    draw.text((start_px[0] + 8, start_px[1] - 6), "robot1", fill=(0, 180, 0))

    for i, victim in enumerate(planner.drone_map.victim_estimates, start=1):
        px, py = planner.geometry.world_to_pixel(victim[0], victim[1])
        colour = (255, 210, 0)
        radius = 5
        if planned_path is not None and planned_path.victim_index == i:
            colour = (255, 80, 255)
            radius = 7
        _draw_cross(draw, px, py, colour, radius=radius)
        draw.text((px + 8, py - 6), f"v{i}", fill=colour)


def _draw_pixel_path(draw, pixel_path, colour=(255, 0, 255), width=2):
    if len(pixel_path) < 2:
        return
    stride = max(1, len(pixel_path) // 1500)
    points = pixel_path[::stride]
    if points[-1] != pixel_path[-1]:
        points.append(pixel_path[-1])
    draw.line(points, fill=colour, width=width)


def _draw_planner_waypoints(planner, draw, waypoints):
    for i, waypoint in enumerate(waypoints, start=1):
        px, py = planner.geometry.world_to_pixel(waypoint.x, waypoint.y)
        _draw_circle(draw, px, py, (0, 255, 255), radius=4)
        draw.text((px + 5, py + 4), str(i), fill=(0, 120, 140))


def _draw_path(draw, pixel_path, colour, width):
    if pixel_path is None or len(pixel_path) < 2:
        return
    stride = max(1, len(pixel_path) // 1500)
    points = list(pixel_path[::stride])
    if points[-1] != pixel_path[-1]:
        points.append(pixel_path[-1])
    draw.line(points, fill=colour, width=width)


def _draw_live_waypoints(replanner, draw, waypoints):
    for waypoint in waypoints:
        px, py = replanner.planner.geometry.world_to_pixel(waypoint.x, waypoint.y)
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=(0, 255, 255), outline=(0, 80, 100))


def _draw_target(replanner, draw):
    path = replanner.candidate_path or replanner.active_path
    if path is None or path.victim_world is None:
        return
    px, py = replanner.planner.geometry.world_to_pixel(path.victim_world[0], path.victim_world[1])
    draw.line((px - 7, py, px + 7, py), fill=(255, 215, 0), width=3)
    draw.line((px, py - 7, px, py + 7), fill=(255, 215, 0), width=3)


def _draw_robot_footprint(replanner, draw, pose: Pose2D):
    half_length = 0.5 * replanner.ROBOT_LENGTH_M
    half_width = 0.5 * replanner.ROBOT_WIDTH_M
    c = math.cos(pose.yaw)
    s = math.sin(pose.yaw)
    corners = []
    for local_x, local_y in ((half_length, half_width), (half_length, -half_width), (-half_length, -half_width), (-half_length, half_width)):
        world_x = pose.x + c * local_x - s * local_y
        world_y = pose.y + s * local_x + c * local_y
        corners.append(replanner.planner.geometry.world_to_pixel(world_x, world_y))
    draw.line(corners + [corners[0]], fill=(0, 255, 0), width=2)
    centre = replanner.planner.geometry.world_to_pixel(pose.x, pose.y)
    front = replanner.planner.geometry.world_to_pixel(pose.x + 0.25 * math.cos(pose.yaw), pose.y + 0.25 * math.sin(pose.yaw))
    draw.line((centre[0], centre[1], front[0], front[1]), fill=(0, 255, 0), width=3)


def _draw_live_legend(replanner, draw, panel_x: int, panel_width: int, victim_status=None):
    x = panel_x + 16
    y = 16
    for text, colour in (
        (f"Live replan: {replanner.state}", (0, 0, 0)),
        ("magenta = current path", (0, 0, 0)),
        ("green = coordinator route comparison", (0, 0, 0)),
        ("cyan = candidate path / waypoints", (0, 0, 0)),
        ("blue = failed candidate path", (0, 0, 0)),
        ("red = blocked path cells", (0, 0, 0)),
        ("green = robot footprint / heading", (0, 0, 0)),
        ("yellow = retained route target", (0, 0, 0)),
        ("victim: pink=prior magenta=current cyan=approach green=found", (120, 0, 140)),
    ):
        draw.text((x, y), text, fill=colour)
        y += 18 if not text.startswith("Live replan") else 24
    y += 10
    path = replanner.candidate_path or replanner.active_path
    if path is None or path.victim_world is None:
        target_text = "Target: none"
    else:
        target_name = f"victim {path.victim_index}" if path.victim_index > 0 else "8 m test"
        target_text = (
            f"Target: {target_name}\n"
            f"x={path.victim_world[0]:.3f}  y={path.victim_world[1]:.3f}"
        )
    draw.multiline_text((x, y), target_text, fill=(0, 0, 0), spacing=3)
    y += 42
    if path is not None and path.success:
        draw.text((x, y), f"mode={path.planning_mode}", fill=(0, 0, 0)); y += 20
        draw.text((x, y), f"physical={path.physical_path_cost_m:.2f} m  weighted={path.weighted_path_cost_m:.2f} m", fill=(0, 0, 0)); y += 20
        draw.text((x, y), f"plan_time={path.planning_time_s:.3f} s", fill=(0, 0, 0)); y += 24
    draw.text((x, y), f"revision={replanner.last_snapshot_revision}  interval={replanner.interval_s:.1f}s", fill=(0, 0, 0)); y += 22
    if replanner.recovery_controller is None:
        recovery_text = "recovery attempts: unavailable"
    else:
        recovery = replanner.recovery_controller
        recovery_text = f"recovery={recovery.state}  used={recovery.attempt_count}/{recovery.max_attempts}  left={recovery.attempts_remaining}"
    recovery_colour = (180, 0, 0) if (replanner.recovery_controller is not None and replanner.recovery_controller.attempts_remaining == 0) else (0, 0, 0)
    draw.text((x, y), recovery_text, fill=recovery_colour); y += 24
    reason_width = max(24, (panel_width - 32) // 7)
    reason_lines = [replanner.reason[index:index + reason_width] for index in range(0, len(replanner.reason), reason_width)] or ["-"]
    draw.multiline_text(
        (x, y),
        "Reason:\n" + "\n".join(reason_lines[:4]),
        fill=(160, 0, 0) if replanner.state == replanner.FAILED else (0, 0, 0),
        spacing=3,
    )
    y += 84
    if victim_status:
        draw.text((x, y), f"Victim mission: {victim_status.get('state', '-')}", fill=(120, 0, 140)); y += 20
        draw.text((x, y), f"track={victim_status.get('selected_track_id') or '-'}", fill=(0, 0, 0)); y += 20
        draw.text((x, y), "prior selection=" f"{str(victim_status.get('prior_selection_mode', 'nearest')).upper()}", fill=(0, 0, 0)); y += 20
        report_status = victim_status.get("report_status") or "-"
        draw.text((x, y), f"report={report_status}  confidence={victim_status.get('report_confidence', 0.0):.2f}", fill=(0, 0, 0)); y += 20
        victim_reason = str(victim_status.get("reason") or "-")
        draw.text((x, y), f"victim reason: {victim_reason[:44]}", fill=(0, 0, 0))


def _draw_victim_overlay(replanner, draw, status):
    if not status:
        return
    approach_target = status.get("approach_target")
    if approach_target is not None:
        px, py = replanner.planner.geometry.world_to_pixel(
            approach_target[0],
            approach_target[1],
        )
        draw.rectangle((px - 5, py - 5, px + 5, py + 5), fill=(0, 200, 255), outline=(0, 60, 100))
    selected_id = status.get("selected_track_id")
    for track in status.get("tracks", ()):
        position = track.get("position")
        if position is None:
            continue
        px, py = replanner.planner.geometry.world_to_pixel(position[0], position[1])
        if track.get("status") == "FOUND":
            colour = (0, 190, 70)
        elif track.get("id") == selected_id:
            colour = (255, 30, 180)
        else:
            colour = (255, 150, 220)
        radius = 7 if track.get("locked") else 4
        draw.line((px - radius, py, px + radius, py), fill=colour, width=3)
        draw.line((px, py - radius, px, py + radius), fill=colour, width=3)


def _draw_cross(draw, px: int, py: int, colour, radius: int):
    draw.line((px - radius, py, px + radius, py), fill=colour, width=3)
    draw.line((px, py - radius, px, py + radius), fill=colour, width=3)


def _draw_circle(draw, px: int, py: int, colour, radius: int):
    draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=colour, outline=(0, 0, 0))
