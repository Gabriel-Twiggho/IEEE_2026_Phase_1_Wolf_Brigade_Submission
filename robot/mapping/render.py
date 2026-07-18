"""Lazy debug rendering helpers for mapping outputs and image viewers."""

from __future__ import annotations

import atexit
import math
import os
import subprocess
import sys
import uuid

from config import LIVE_MAP_VIEWER_PATH, SLAM_LITE_DIR
from shared_types import Pose2D


ROUTE_DEBUG_CROP_SIZE_M = 4.0
ROUTE_DEBUG_CELL_PX = 8
ROUTE_DEBUG_PANEL_WIDTH_PX = 390


def render_drone_map_overlay(drone_map, image):
    from PIL import ImageDraw

    draw = ImageDraw.Draw(image)
    _draw_drone_origin(drone_map, draw)
    _draw_drone_axes(drone_map, draw)
    _draw_robot_starts(drone_map, draw)
    _draw_victims(drone_map, draw)
    _draw_drone_legend(draw)

    tmp_path = drone_map.output_path.with_name(
        f"{drone_map.output_path.stem}.{drone_map.robot_id}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
    )
    image.save(tmp_path)
    os.replace(tmp_path, drone_map.output_path)


def render_live_grid(mapper):
    if not mapper.enabled or not mapper.render_enabled or not mapper._render_available:
        return

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        print(f"[{mapper.robot_id}] Live map rendering disabled: {exc}")
        mapper._render_available = False
        return

    if not mapper.confidence_grid:
        return
    min_gx, min_gy, max_gx, max_gy = _live_render_bounds(mapper)
    width = max_gx - min_gx + 1
    height = max_gy - min_gy + 1
    image = Image.new("RGB", (width, height), _confidence_color(mapper, 0.0))
    pixels = image.load()
    for (gx, gy), confidence in mapper.confidence_grid.items():
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = _confidence_color(mapper, confidence)

    draw = ImageDraw.Draw(image)
    overlap = mapper.local_obstacles.depth_cells & mapper.local_obstacles.ir_cells
    for gx, gy in mapper.local_obstacles.depth_cells - overlap:
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = (230, 40, 40)
    for gx, gy in mapper.local_obstacles.ir_cells - overlap:
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = (255, 145, 0)
    for gx, gy in overlap:
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = (175, 40, 205)

    for gx, gy in mapper.latest_hit_cells[-1200:]:
        px = gx - min_gx
        py = gy - min_gy
        if 0 <= px < width and 0 <= py < height:
            draw.point((px, py), fill=(255, 210, 0))

    _draw_live_robot(mapper, draw, mapper.last_pose, min_gx, min_gy, width, height)

    tmp_path = mapper.output_path.with_suffix(".tmp.png")
    image.save(tmp_path)
    os.replace(tmp_path, mapper.output_path)


def render_route_start_debug(mapper, planner, planned_path, current_pose: Pose2D):
    """Save a local, source-coloured snapshot when a route is adopted."""
    if (
        not mapper.enabled
        or mapper.geometry is None
        or planned_path is None
        or not planned_path.success
    ):
        return False

    snapshot = mapper.planning_snapshot()
    if snapshot is None:
        return False

    try:
        from PIL import Image, ImageDraw
    except ImportError as exc:
        raise RuntimeError(f"PIL is unavailable: {exc}") from exc

    resolution_m = max(
        1e-9,
        float(getattr(mapper.geometry, "resolution", mapper.RESOLUTION_M_PER_PX)),
    )
    centre_gx, centre_gy = mapper.world_to_grid(current_pose.x, current_pose.y)
    half_crop_cells = max(
        1,
        int(math.ceil(0.5 * ROUTE_DEBUG_CROP_SIZE_M / resolution_m)),
    )
    min_gx = max(0, centre_gx - half_crop_cells)
    min_gy = max(0, centre_gy - half_crop_cells)
    max_gx = min(snapshot.width - 1, centre_gx + half_crop_cells)
    max_gy = min(snapshot.height - 1, centre_gy + half_crop_cells)
    if max_gx < min_gx or max_gy < min_gy:
        return False

    width_cells = max_gx - min_gx + 1
    height_cells = max_gy - min_gy + 1
    base = Image.new(
        "RGBA",
        (width_cells, height_cells),
        (*_confidence_color(mapper, 0.0), 255),
    )
    pixels = base.load()
    for (gx, gy), confidence in mapper.confidence_grid.items():
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = (
                *_confidence_color(mapper, confidence),
                255,
            )

    inflation_radius_m = max(0.0, float(planner.inflation_radius_m))
    inflation_radius_px = int(math.ceil(inflation_radius_m / resolution_m))
    inflated_cells = _route_debug_inflated_cells(
        snapshot.occupied_cells,
        inflation_radius_px,
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )
    inflation_overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    inflation_pixels = inflation_overlay.load()
    for gx, gy in inflated_cells:
        inflation_pixels[gx - min_gx, gy - min_gy] = (145, 90, 205, 90)
    base = Image.alpha_composite(base, inflation_overlay)
    pixels = base.load()

    for gx, gy in mapper.occupied_cells:
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = (25, 25, 25, 255)

    depth_cells = mapper.local_obstacles.depth_cells
    ir_cells = mapper.local_obstacles.ir_cells
    overlap_cells = depth_cells & ir_cells
    _route_debug_fill_cells(
        pixels,
        depth_cells - overlap_cells,
        (230, 40, 40, 255),
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )
    _route_debug_fill_cells(
        pixels,
        ir_cells - overlap_cells,
        (255, 145, 0, 255),
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )
    _route_debug_fill_cells(
        pixels,
        overlap_cells,
        (175, 40, 205, 255),
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )
    _route_debug_fill_cells(
        pixels,
        mapper.latest_hit_cells,
        (255, 210, 0, 255),
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )

    resampling = getattr(Image, "Resampling", Image)
    map_image = base.convert("RGB").resize(
        (width_cells * ROUTE_DEBUG_CELL_PX, height_cells * ROUTE_DEBUG_CELL_PX),
        resampling.NEAREST,
    )
    map_draw = ImageDraw.Draw(map_image)
    path_cells = _route_debug_path_cells(mapper, planned_path, current_pose)
    _draw_route_debug_path(
        map_draw,
        path_cells,
        min_gx,
        min_gy,
        max_gx,
        max_gy,
    )
    blocked_path_cells = tuple(
        cell
        for cell in path_cells
        if cell in inflated_cells
        and min_gx <= cell[0] <= max_gx
        and min_gy <= cell[1] <= max_gy
    )
    for cell in blocked_path_cells:
        cx, cy = _route_debug_cell_centre(cell, min_gx, min_gy)
        map_draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill=(255, 0, 0))

    _draw_route_debug_robot(
        mapper,
        map_draw,
        current_pose,
        min_gx,
        min_gy,
    )
    target = _route_debug_target(planned_path)
    if target is not None:
        target_gx, target_gy = mapper.world_to_grid(target[0], target[1])
        if min_gx <= target_gx <= max_gx and min_gy <= target_gy <= max_gy:
            tx, ty = _route_debug_cell_centre(
                (target_gx, target_gy),
                min_gx,
                min_gy,
            )
            _draw_cross(map_draw, tx, ty, (255, 215, 0), radius=7)

    output_height = max(map_image.height, 540)
    image = Image.new(
        "RGB",
        (map_image.width + ROUTE_DEBUG_PANEL_WIDTH_PX, output_height),
        (245, 245, 245),
    )
    image.paste(map_image, (0, 0))
    panel_draw = ImageDraw.Draw(image)
    _draw_route_debug_panel(
        panel_draw,
        map_image.width + 14,
        mapper,
        planner,
        planned_path,
        current_pose,
        snapshot,
        target,
        resolution_m,
        inflation_radius_px,
        path_cells,
        blocked_path_cells,
        depth_cells,
        ir_cells,
        overlap_cells,
    )

    output_path = SLAM_LITE_DIR / f"{mapper.robot_id}_route_start_debug.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(
        f"{output_path.stem}.{os.getpid()}.{uuid.uuid4().hex}.tmp.png"
    )
    image.save(tmp_path)
    os.replace(tmp_path, output_path)
    return True


def _route_debug_inflated_cells(
    occupied_cells,
    radius_px: int,
    min_gx: int,
    min_gy: int,
    max_gx: int,
    max_gy: int,
):
    from robot.planning.grid_search import inflation_offsets

    offsets = inflation_offsets(radius_px)
    inflated = set()
    for gx, gy in occupied_cells:
        if not (
            min_gx - radius_px <= gx <= max_gx + radius_px
            and min_gy - radius_px <= gy <= max_gy + radius_px
        ):
            continue
        for dx, dy in offsets:
            cell = gx + dx, gy + dy
            if min_gx <= cell[0] <= max_gx and min_gy <= cell[1] <= max_gy:
                inflated.add(cell)
    return inflated


def _route_debug_fill_cells(
    pixels,
    cells,
    colour,
    min_gx: int,
    min_gy: int,
    max_gx: int,
    max_gy: int,
):
    for gx, gy in cells:
        if min_gx <= gx <= max_gx and min_gy <= gy <= max_gy:
            pixels[gx - min_gx, gy - min_gy] = colour


def _route_debug_path_cells(mapper, planned_path, current_pose: Pose2D):
    if planned_path.pixel_path:
        return tuple((int(cell[0]), int(cell[1])) for cell in planned_path.pixel_path)

    from robot.planning.grid_search import bresenham_cells

    route_points = [mapper.world_to_grid(current_pose.x, current_pose.y)]
    for waypoint in planned_path.waypoints:
        if hasattr(waypoint, "x"):
            x_m, y_m = waypoint.x, waypoint.y
        else:
            x_m, y_m = waypoint[0], waypoint[1]
        route_points.append(mapper.world_to_grid(float(x_m), float(y_m)))

    cells = []
    for start, end in zip(route_points, route_points[1:]):
        segment = bresenham_cells(start[0], start[1], end[0], end[1])
        if cells and segment and cells[-1] == segment[0]:
            segment = segment[1:]
        cells.extend(segment)
    return tuple(cells)


def _draw_route_debug_path(
    draw,
    path_cells,
    min_gx: int,
    min_gy: int,
    max_gx: int,
    max_gy: int,
):
    run = []
    for cell in path_cells:
        if min_gx <= cell[0] <= max_gx and min_gy <= cell[1] <= max_gy:
            run.append(_route_debug_cell_centre(cell, min_gx, min_gy))
            continue
        _draw_route_debug_path_run(draw, run)
        run = []
    _draw_route_debug_path_run(draw, run)


def _draw_route_debug_path_run(draw, points):
    if len(points) >= 2:
        draw.line(points, fill=(220, 0, 220), width=2)
    elif points:
        x, y = points[0]
        draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=(220, 0, 220))


def _draw_route_debug_robot(
    mapper,
    draw,
    pose: Pose2D,
    min_gx: int,
    min_gy: int,
):
    from parameters import RobotGeometry

    half_length = 0.5 * RobotGeometry.LENGTH_M
    half_width = 0.5 * RobotGeometry.WIDTH_M
    cos_yaw = math.cos(pose.yaw)
    sin_yaw = math.sin(pose.yaw)
    corners = []
    for forward_m, lateral_m in (
        (half_length, half_width),
        (half_length, -half_width),
        (-half_length, -half_width),
        (-half_length, half_width),
    ):
        world_x = pose.x + cos_yaw * forward_m - sin_yaw * lateral_m
        world_y = pose.y + sin_yaw * forward_m + cos_yaw * lateral_m
        corners.append(
            _route_debug_world_point(mapper, world_x, world_y, min_gx, min_gy)
        )
    draw.line(corners + [corners[0]], fill=(0, 220, 70), width=2)

    centre = _route_debug_world_point(mapper, pose.x, pose.y, min_gx, min_gy)
    front = _route_debug_world_point(
        mapper,
        pose.x + half_length * cos_yaw,
        pose.y + half_length * sin_yaw,
        min_gx,
        min_gy,
    )
    draw.line((*centre, *front), fill=(0, 150, 50), width=2)


def _route_debug_world_point(mapper, x_m: float, y_m: float, min_gx: int, min_gy: int):
    if hasattr(mapper.geometry, "world_to_pixel_float"):
        gx, gy = mapper.geometry.world_to_pixel_float(x_m, y_m)
    else:
        gx, gy = mapper.world_to_grid(x_m, y_m)
    return (
        int(round((gx - min_gx + 0.5) * ROUTE_DEBUG_CELL_PX)),
        int(round((gy - min_gy + 0.5) * ROUTE_DEBUG_CELL_PX)),
    )


def _route_debug_cell_centre(cell, min_gx: int, min_gy: int):
    return (
        (int(cell[0]) - min_gx) * ROUTE_DEBUG_CELL_PX + ROUTE_DEBUG_CELL_PX // 2,
        (int(cell[1]) - min_gy) * ROUTE_DEBUG_CELL_PX + ROUTE_DEBUG_CELL_PX // 2,
    )


def _route_debug_target(planned_path):
    if planned_path.waypoints:
        target = planned_path.waypoints[-1]
        if hasattr(target, "x"):
            return float(target.x), float(target.y)
        return float(target[0]), float(target[1])
    if planned_path.victim_world is not None:
        return float(planned_path.victim_world[0]), float(planned_path.victim_world[1])
    return None


def _draw_route_debug_panel(
    draw,
    x: int,
    mapper,
    planner,
    planned_path,
    pose: Pose2D,
    snapshot,
    target,
    resolution_m: float,
    inflation_radius_px: int,
    path_cells,
    blocked_path_cells,
    depth_cells,
    ir_cells,
    overlap_cells,
):
    target_text = "-"
    if target is not None:
        target_text = f"({target[0]:.2f}, {target[1]:.2f})"
    lines = (
        "Route-start map debug",
        f"robot={mapper.robot_id}",
        f"mode={planned_path.planning_mode}",
        f"pose=({pose.x:.2f}, {pose.y:.2f}, {math.degrees(pose.yaw):+.1f} deg)",
        f"target={target_text}",
        f"map_revision={snapshot.revision}",
        f"resolution={resolution_m:.3f} m/cell",
        f"inflation={planner.inflation_radius_m:.3f} m ({inflation_radius_px} cells)",
        f"snapshot_occupied={len(snapshot.occupied_cells)}",
        f"black_occupied={len(mapper.occupied_cells)}",
        f"depth_cells={len(depth_cells)}",
        f"ir_cells={len(ir_cells)}",
        f"depth_ir_overlap={len(overlap_cells)}",
        f"latest_lidar_hits={len(mapper.latest_hit_cells)}",
        f"route_cells={len(path_cells)}",
        f"inflated_route_hits={len(blocked_path_cells)}",
        "",
        "pale purple = planner inflation",
        "black = static/confirmed lidar",
        "red = depth obstacle",
        "orange = IR obstacle",
        "purple = depth + IR",
        "yellow = latest lidar hit / target",
        "magenta = adopted route",
        "bright red = route in inflation",
        "green = robot footprint / heading",
    )
    y = 14
    for line in lines:
        draw.text((x, y), line, fill=(0, 0, 0))
        y += 20


def start_viewer(image_path, title: str, error_prefix: str):
    script = LIVE_MAP_VIEWER_PATH
    if not script.exists():
        return None
    try:
        process = subprocess.Popen(
            [
                sys.executable,
                str(script),
                str(image_path),
                title,
            ],
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


def _draw_drone_origin(drone_map, draw):
    px, py = drone_map.geometry.origin_px
    drone_map.origin_px_from_formula = drone_map.geometry.world_to_pixel(0.0, 0.0)
    _draw_cross(draw, px, py, (255, 0, 0), radius=7)
    draw.text((px + 8, py + 6), "origin", fill=(255, 0, 0))


def _draw_drone_axes(drone_map, draw):
    origin = drone_map.geometry.world_to_pixel(0.0, 0.0)
    x_end = drone_map.geometry.world_to_pixel(2.0, 0.0)
    y_end = drone_map.geometry.world_to_pixel(0.0, 2.0)
    _draw_arrow(draw, origin, x_end, (255, 70, 70), "x+")
    _draw_arrow(draw, origin, y_end, (70, 170, 255), "y+")


def _draw_robot_starts(drone_map, draw):
    colours = {
        "robot1": (0, 220, 0),
        "robot2": (0, 140, 255),
    }
    for robot_id, pose in drone_map.ROBOT_STARTS.items():
        px, py = drone_map.geometry.world_to_pixel(pose.x, pose.y)
        colour = colours.get(robot_id, (0, 220, 0))
        _draw_circle(draw, px, py, colour, radius=6)
        draw.text((px + 8, py - 6), robot_id, fill=colour)


def _draw_victims(drone_map, draw):
    for i, (x_m, y_m) in enumerate(drone_map.victim_estimates, start=1):
        px, py = drone_map.geometry.world_to_pixel(x_m, y_m)
        _draw_cross(draw, px, py, (255, 210, 0), radius=6)
        draw.text((px + 8, py - 6), f"v{i}", fill=(255, 210, 0))


def _draw_drone_legend(draw):
    draw.rectangle((8, 8, 230, 82), fill=(245, 245, 245), outline=(80, 80, 80))
    draw.text((14, 14), "Drone extraction overlay", fill=(0, 0, 0))
    draw.text((14, 30), "red: origin/x axis", fill=(180, 0, 0))
    draw.text((14, 46), "blue: y axis/robot2", fill=(0, 80, 180))
    draw.text((14, 62), "green: robot1  yellow: victims", fill=(0, 110, 0))


def _live_render_bounds(mapper):
    if mapper.seed_from_drone and mapper.seeded_map_size is not None:
        width, height = mapper.seeded_map_size
        return 0, 0, width - 1, height - 1

    robot_cell = mapper.world_to_grid(mapper.last_pose.x, mapper.last_pose.y)
    cells = list(mapper.confidence_grid.keys())
    cells.extend(mapper.latest_hit_cells)
    cells.extend(mapper.local_obstacles.occupied_cells)
    cells.append(robot_cell)

    min_gx = min(gx for gx, _ in cells) - mapper.render_padding_px
    max_gx = max(gx for gx, _ in cells) + mapper.render_padding_px
    min_gy = min(gy for _, gy in cells) - mapper.render_padding_px
    max_gy = max(gy for _, gy in cells) + mapper.render_padding_px
    return min_gx, min_gy, max_gx, max_gy


def _confidence_color(mapper, confidence: float):
    if confidence >= mapper.occupied_threshold:
        return (25, 25, 25)
    if confidence <= mapper.free_threshold:
        return (230, 245, 255)
    return (125, 125, 125)


def _draw_live_robot(mapper, draw, pose: Pose2D, min_gx: int, min_gy: int, width: int, height: int):
    gx, gy = mapper.world_to_grid(pose.x, pose.y)
    px = gx - min_gx
    py = gy - min_gy
    if not (0 <= px < width and 0 <= py < height):
        return

    radius = 5
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=(0, 180, 0),
        outline=(0, 70, 0),
    )
    arrow_len_m = 0.6
    end_gx, end_gy = mapper.world_to_grid(
        pose.x + arrow_len_m * math.cos(pose.yaw),
        pose.y + arrow_len_m * math.sin(pose.yaw),
    )
    end_x = end_gx - min_gx
    end_y = end_gy - min_gy
    draw.line((px, py, end_x, end_y), fill=(0, 255, 0), width=3)


def _draw_cross(draw, px: int, py: int, colour, radius: int):
    draw.line((px - radius, py, px + radius, py), fill=colour, width=3)
    draw.line((px, py - radius, px, py + radius), fill=colour, width=3)


def _draw_circle(draw, px: int, py: int, colour, radius: int):
    draw.ellipse(
        (px - radius, py - radius, px + radius, py + radius),
        fill=colour,
        outline=(0, 0, 0),
    )


def _draw_arrow(draw, start, end, colour, label: str):
    draw.line((start[0], start[1], end[0], end[1]), fill=colour, width=3)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head_len = 10
    for offset in (math.radians(150), math.radians(-150)):
        hx = end[0] + head_len * math.cos(angle + offset)
        hy = end[1] + head_len * math.sin(angle + offset)
        draw.line((end[0], end[1], hx, hy), fill=colour, width=3)
    draw.text((end[0] + 6, end[1] + 6), label, fill=colour)
