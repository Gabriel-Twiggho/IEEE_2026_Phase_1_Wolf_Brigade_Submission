"""Yaw-aware local escape planner used after safety latches."""

from __future__ import annotations

import heapq
import math
from itertools import count

from config import _env_float, _env_int, _env_flag
from parameters import RobotGeometry
from parameters import SafetyRecovery as SafetyRecoveryParams
from shared_types import PlannedPath, Pose2D


def plan_local_escape(planner, current_pose, snapshot):
    """
    Plan a short local escape route from the current safety-latched pose.

    Unlike the normal global planner, this search includes heading.  The search
    state is (x, y, yaw_bin) in the 1 cm/px fine patch, and each primitive is
    accepted only if the rotated robot footprint remains clear.
    """
    self = planner
    if not self.enabled or not self.use_live_snapshot(snapshot):
        return PlannedPath(
            robot_id=self.robot_id,
            success=False,
            error_reason="planner not ready or no live snapshot for escape",
        )

    cx_px, cy_px = self.geometry.world_to_pixel(current_pose.x, current_pose.y)
    scale = max(
        1,
        int(round(self.geometry.resolution / self.fine_resolution_m)),
    )
    yaw_bins = max(
        4,
        _env_int("SAFETY_ESCAPE_YAW_BINS", SafetyRecoveryParams.YAW_BINS),
    )
    primitive_step_m = max(
        self.fine_resolution_m,
        _env_float(
            "SAFETY_ESCAPE_PRIMITIVE_STEP_M",
            SafetyRecoveryParams.PRIMITIVE_STEP_M,
        ),
    )
    primitive_arc_radius_m = max(
        primitive_step_m,
        _env_float(
            "SAFETY_ESCAPE_PRIMITIVE_ARC_RADIUS_M",
            SafetyRecoveryParams.PRIMITIVE_ARC_RADIUS_M,
        ),
    )
    rotate_step_rad = max(
        math.radians(1.0),
        _env_float(
            "SAFETY_ESCAPE_ROTATE_STEP_RAD",
            SafetyRecoveryParams.ROTATE_STEP_RAD,
        ),
    )
    footprint_margin_m = max(
        0.0,
        _env_float(
            "SAFETY_ESCAPE_FOOTPRINT_MARGIN_M",
            SafetyRecoveryParams.FOOTPRINT_MARGIN_M,
        ),
    )
    max_expansions = max(
        1,
        _env_int(
            "SAFETY_ESCAPE_MAX_EXPANSIONS",
            SafetyRecoveryParams.MAX_EXPANSIONS,
        ),
    )
    escape_speed_cap = _env_float(
        "SAFETY_ESCAPE_SPEED_CAP_RAD_S",
        self.ESCAPE_SPEED_CAP_RAD_S,
    )
    start_clear_radius_m = max(
        0.0,
        _env_float(
            "SAFETY_ESCAPE_START_CLEAR_RADIUS_M",
            SafetyRecoveryParams.START_CLEAR_RADIUS_M,
        ),
    )
    footprint_points = _footprint_sample_points(
        self.fine_resolution_m,
        footprint_margin_m,
    )
    debug_enabled = _env_flag("SAFETY_ESCAPE_DEBUG_RENDER", False)
    debug_context = None

    for patch_size_m in self.ESCAPE_PATCH_SIZES_M:
        patch = _build_fine_patch(self, cx_px, cy_px, patch_size_m, scale)
        if patch is None:
            continue
        (
            min_px,
            min_py,
            fine_width,
            fine_height,
            occupied,
            clearance_m,
            fine_free,
        ) = patch
        cleared_start_cells = _clear_start_circle_from_occupied(
            self,
            current_pose,
            min_px,
            min_py,
            scale,
            fine_width,
            fine_height,
            occupied,
            start_clear_radius_m,
        )
        if cleared_start_cells:
            import numpy as np

            clearance_px = self._euclidean_clearance_pixels(occupied)
            clearance_m = np.maximum(
                0.0,
                clearance_px - 0.5,
            ) * self.fine_resolution_m
            fine_free = clearance_m >= self.fine_clearance_m
        if debug_enabled:
            debug_context = {
                "patch_size_m": patch_size_m,
                "min_px": min_px,
                "min_py": min_py,
                "scale": scale,
                "fine_width": fine_width,
                "fine_height": fine_height,
                "occupied": occupied,
                "clearance_m": clearance_m,
                "fine_free": fine_free,
                "yaw_bins": yaw_bins,
                "primitive_step_m": primitive_step_m,
                "primitive_arc_radius_m": primitive_arc_radius_m,
                "rotate_step_rad": rotate_step_rad,
                "footprint_margin_m": footprint_margin_m,
                "max_expansions": max_expansions,
                "footprint_points": footprint_points,
                "start_clear_radius_m": start_clear_radius_m,
                "cleared_start_cells": cleared_start_cells,
                "start": None,
                "failure_reason": "escape search did not find a valid route",
            }

        raw_start = self._coarse_to_fine_cell(
            (cx_px, cy_px), min_px, min_py, scale
        )
        start = self._nearest_free_cell_on_grid(
            raw_start,
            fine_free,
            fine_width,
            fine_height,
            min(0.20, self.snap_radius_m),
            1.0 / self.fine_resolution_m,
        )
        if start is None:
            if debug_context is not None:
                debug_context["failure_reason"] = "no free start cell in local escape patch"
            continue
        if debug_context is not None:
            debug_context["start"] = start

        result = _search_yaw_escape(
            self,
            current_pose,
            start,
            min_px,
            min_py,
            scale,
            fine_width,
            fine_height,
            occupied,
            clearance_m,
            yaw_bins,
            primitive_step_m,
            primitive_arc_radius_m,
            rotate_step_rad,
            footprint_points,
            max_expansions,
        )
        if result is None:
            if debug_context is not None:
                debug_context["failure_reason"] = (
                    f"search exhausted before reaching {self.escape_goal_clearance_m:.3f} m clearance"
                )
            continue

        states, expansions = result
        waypoints = _waypoints_from_states(
            self,
            states,
            min_px,
            min_py,
            scale,
            yaw_bins,
        )
        if debug_context is not None:
            debug_context["states"] = states
            debug_context["waypoints"] = waypoints
            debug_context["expansions"] = expansions
            debug_context["failure_reason"] = (
                "already clear for normal replanning"
                if not waypoints
                else f"yaw-aware local escape in {expansions} expansions"
            )
        if not waypoints:
            if debug_context is not None:
                _render_escape_debug(
                    self,
                    current_pose,
                    debug_context,
                    "SUCCESS: already clear for normal replanning",
                )
            return PlannedPath(
                robot_id=self.robot_id,
                waypoints=(),
                waypoint_speed_caps=(),
                planning_mode="ESCAPE",
                success=True,
                error_reason="already clear for normal replanning",
            )

        if debug_context is not None:
            _render_escape_debug(
                self,
                current_pose,
                debug_context,
                f"SUCCESS: yaw-aware local escape in {expansions} expansions",
            )
        return PlannedPath(
            robot_id=self.robot_id,
            world_path=tuple((pose.x, pose.y) for pose in waypoints),
            waypoints=tuple(waypoints),
            waypoint_speed_caps=tuple(escape_speed_cap for _ in waypoints),
            planning_mode="ESCAPE",
            physical_path_cost_m=_waypoint_length(waypoints),
            weighted_path_cost_m=_waypoint_length(waypoints),
            success=True,
            error_reason=f"yaw-aware local escape in {expansions} expansions",
        )

    reason = "no yaw-aware local escape route found in any patch size"
    if debug_enabled and debug_context is not None:
        _render_escape_debug(self, current_pose, debug_context, f"FAILED: {reason}")
    return PlannedPath(
        robot_id=self.robot_id,
        planning_mode="ESCAPE",
        success=False,
        error_reason=reason,
    )


def _build_fine_patch(planner, cx_px, cy_px, patch_size_m, scale):
    side_px = max(
        1,
        int(math.ceil(patch_size_m / planner.geometry.resolution)),
    )
    min_px = max(0, cx_px - side_px // 2)
    min_py = max(0, cy_px - side_px // 2)
    max_px = min(planner.width - 1, min_px + side_px - 1)
    max_py = min(planner.height - 1, min_py + side_px - 1)
    min_px = max(0, max_px - side_px + 1)
    min_py = max(0, max_py - side_px + 1)

    fine_width = (max_px - min_px + 1) * scale
    fine_height = (max_py - min_py + 1) * scale

    import numpy as np

    occupied = np.zeros((fine_height, fine_width), dtype=bool)
    for px, py in planner.source_occupied_cells:
        if min_px <= px <= max_px and min_py <= py <= max_py:
            x0 = (px - min_px) * scale
            y0 = (py - min_py) * scale
            occupied[y0:y0 + scale, x0:x0 + scale] = True

    clearance_px = planner._euclidean_clearance_pixels(occupied)
    clearance_m = np.maximum(
        0.0,
        clearance_px - 0.5,
    ) * planner.fine_resolution_m
    fine_free = clearance_m >= planner.fine_clearance_m
    return min_px, min_py, fine_width, fine_height, occupied, clearance_m, fine_free


def _render_escape_debug(planner, current_pose, debug_context, reason):
    try:
        from robot.planning.render import render_escape_debug

        render_escape_debug(planner, current_pose, debug_context, reason)
    except Exception as exc:
        print(f"[{planner.robot_id}] Safety escape debug render failed: {exc}")


def _clear_start_circle_from_occupied(
    planner,
    current_pose,
    min_px,
    min_py,
    scale,
    fine_width,
    fine_height,
    occupied,
    radius_m,
):
    if radius_m <= 0.0:
        return ()
    centre = _world_to_fine_cell(
        planner,
        current_pose.x,
        current_pose.y,
        min_px,
        min_py,
        scale,
        fine_width,
        fine_height,
    )
    if centre is None:
        return ()

    radius_cells = max(
        1,
        int(math.ceil(radius_m / planner.fine_resolution_m)),
    )
    radius_squared = radius_cells * radius_cells
    cleared = set()
    min_fy = max(0, centre[1] - radius_cells)
    max_fy = min(fine_height, centre[1] + radius_cells + 1)
    min_fx = max(0, centre[0] - radius_cells)
    max_fx = min(fine_width, centre[0] + radius_cells + 1)
    for fy in range(min_fy, max_fy):
        for fx in range(min_fx, max_fx):
            if (fx - centre[0]) ** 2 + (fy - centre[1]) ** 2 > radius_squared:
                continue
            if bool(occupied[fy, fx]):
                occupied[fy, fx] = False
                cleared.add((fx, fy))
    return tuple(sorted(cleared))


def _search_yaw_escape(
    planner,
    current_pose,
    start,
    min_px,
    min_py,
    scale,
    fine_width,
    fine_height,
    occupied,
    clearance_m,
    yaw_bins,
    primitive_step_m,
    primitive_arc_radius_m,
    rotate_step_rad,
    footprint_points,
    max_expansions,
):
    start_state = (
        int(start[0]),
        int(start[1]),
        _yaw_to_bin(current_pose.yaw, yaw_bins),
    )
    goal_threshold = planner.escape_goal_clearance_m
    primitives = _escape_primitives(
        primitive_step_m,
        primitive_arc_radius_m,
        rotate_step_rad,
    )

    open_heap = []
    sequence = count()
    heapq.heappush(open_heap, (0.0, next(sequence), start_state))
    came_from = {}
    cost_so_far = {start_state: 0.0}
    expansions = 0

    if _state_is_escape_goal(
        planner,
        start_state,
        min_px,
        min_py,
        scale,
        occupied,
        clearance_m,
        goal_threshold,
        yaw_bins,
        footprint_points,
    ):
        return (start_state,), 0

    while open_heap and expansions < max_expansions:
        _priority, _sequence_id, state = heapq.heappop(open_heap)
        expansions += 1

        for distance_m, yaw_delta_rad, primitive_cost in primitives:
            outcome = _apply_primitive(
                planner,
                state,
                min_px,
                min_py,
                scale,
                fine_width,
                fine_height,
                occupied,
                yaw_bins,
                distance_m,
                yaw_delta_rad,
                footprint_points,
            )
            if outcome is None:
                continue
            next_state = outcome
            if next_state == state:
                continue

            new_cost = cost_so_far[state] + primitive_cost
            if new_cost >= cost_so_far.get(next_state, float("inf")):
                continue
            came_from[next_state] = state
            cost_so_far[next_state] = new_cost

            if _state_is_escape_goal(
                planner,
                next_state,
                min_px,
                min_py,
                scale,
                occupied,
                clearance_m,
                goal_threshold,
                yaw_bins,
                footprint_points,
            ):
                return _reconstruct_state_path(came_from, next_state), expansions

            heuristic = _goal_heuristic(clearance_m, next_state, goal_threshold)
            heapq.heappush(
                open_heap,
                (new_cost + heuristic, next(sequence), next_state),
            )

    return None


def _escape_primitives(step_m, arc_radius_m, rotate_step_rad):
    arc_delta = step_m / arc_radius_m
    return (
        (0.0, rotate_step_rad, 0.5 * rotate_step_rad),
        (0.0, -rotate_step_rad, 0.5 * rotate_step_rad),
        (step_m, 0.0, step_m),
        (step_m, arc_delta, 1.2 * step_m),
        (step_m, -arc_delta, 1.2 * step_m),
    )


def _apply_primitive(
    planner,
    state,
    min_px,
    min_py,
    scale,
    fine_width,
    fine_height,
    occupied,
    yaw_bins,
    distance_m,
    yaw_delta_rad,
    footprint_points,
):
    x_m, y_m, yaw = _state_pose(planner, state, min_px, min_py, scale, yaw_bins)
    for ratio in (0.33, 0.66, 1.0):
        sample = _integrate_pose(
            x_m,
            y_m,
            yaw,
            distance_m * ratio,
            yaw_delta_rad * ratio,
        )
        if not _footprint_clear(
            planner,
            sample[0],
            sample[1],
            sample[2],
            min_px,
            min_py,
            scale,
            fine_width,
            fine_height,
            occupied,
            footprint_points,
        ):
            return None

    end_x, end_y, end_yaw = _integrate_pose(
        x_m,
        y_m,
        yaw,
        distance_m,
        yaw_delta_rad,
    )
    end_cell = _world_to_fine_cell(
        planner,
        end_x,
        end_y,
        min_px,
        min_py,
        scale,
        fine_width,
        fine_height,
    )
    if end_cell is None:
        return None
    return (end_cell[0], end_cell[1], _yaw_to_bin(end_yaw, yaw_bins))


def _state_is_escape_goal(
    planner,
    state,
    min_px,
    min_py,
    scale,
    occupied,
    clearance_m,
    goal_threshold,
    yaw_bins,
    footprint_points,
):
    fx, fy, _yaw_bin = state
    if float(clearance_m[fy, fx]) < goal_threshold:
        return False
    x_m, y_m, yaw = _state_pose(planner, state, min_px, min_py, scale, yaw_bins)
    return _footprint_clear(
        planner,
        x_m,
        y_m,
        yaw,
        min_px,
        min_py,
        scale,
        clearance_m.shape[1],
        clearance_m.shape[0],
        occupied,
        footprint_points,
    )


def _footprint_clear(
    planner,
    x_m,
    y_m,
    yaw,
    min_px,
    min_py,
    scale,
    fine_width,
    fine_height,
    occupied,
    footprint_points,
):
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    for forward_m, lateral_m in footprint_points:
        wx = x_m + cos_yaw * forward_m - sin_yaw * lateral_m
        wy = y_m + sin_yaw * forward_m + cos_yaw * lateral_m
        cell = _world_to_fine_cell(
            planner,
            wx,
            wy,
            min_px,
            min_py,
            scale,
            fine_width,
            fine_height,
        )
        if cell is None:
            return False
        if bool(occupied[cell[1], cell[0]]):
            return False
    return True


def _footprint_sample_points(fine_resolution_m, margin_m):
    half_length = 0.5 * RobotGeometry.LENGTH_M + margin_m
    half_width = 0.5 * RobotGeometry.WIDTH_M + margin_m
    sample_step = max(2.0 * fine_resolution_m, 0.02)
    forward_count = max(2, int(math.ceil((2.0 * half_length) / sample_step)))
    lateral_count = max(2, int(math.ceil((2.0 * half_width) / sample_step)))
    points = []
    for i in range(forward_count + 1):
        forward = -half_length + (2.0 * half_length) * i / forward_count
        for j in range(lateral_count + 1):
            lateral = -half_width + (2.0 * half_width) * j / lateral_count
            points.append((forward, lateral))
    return tuple(points)


def _state_pose(planner, state, min_px, min_py, scale, yaw_bins):
    fx, fy, yaw_bin = state
    x_m, y_m = planner._fine_cell_to_world((fx, fy), min_px, min_py, scale)
    return x_m, y_m, _bin_to_yaw(yaw_bin, yaw_bins)


def _world_to_fine_cell(
    planner,
    x_m,
    y_m,
    min_px,
    min_py,
    scale,
    fine_width,
    fine_height,
):
    px, py = planner.geometry.world_to_pixel_float(x_m, y_m)
    fx = int(round((px - min_px + 0.5) * scale - 0.5))
    fy = int(round((py - min_py + 0.5) * scale - 0.5))
    if not (0 <= fx < fine_width and 0 <= fy < fine_height):
        return None
    return fx, fy


def _integrate_pose(x_m, y_m, yaw, distance_m, yaw_delta_rad):
    if abs(yaw_delta_rad) <= 1e-9:
        return (
            x_m + distance_m * math.cos(yaw),
            y_m + distance_m * math.sin(yaw),
            _wrap_angle(yaw),
        )
    radius_m = distance_m / yaw_delta_rad
    dx_local = radius_m * math.sin(yaw_delta_rad)
    dy_local = radius_m * (1.0 - math.cos(yaw_delta_rad))
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    return (
        x_m + cos_yaw * dx_local - sin_yaw * dy_local,
        y_m + sin_yaw * dx_local + cos_yaw * dy_local,
        _wrap_angle(yaw + yaw_delta_rad),
    )


def _waypoints_from_states(planner, states, min_px, min_py, scale, yaw_bins):
    poses = []
    for state in states[1:]:
        x_m, y_m, yaw = _state_pose(planner, state, min_px, min_py, scale, yaw_bins)
        if poses and math.hypot(poses[-1].x - x_m, poses[-1].y - y_m) < 0.015:
            poses[-1] = Pose2D(poses[-1].x, poses[-1].y, yaw)
            continue
        poses.append(Pose2D(x_m, y_m, yaw))
    if not poses:
        return ()
    final = poses[-1]
    if len(poses) >= 2:
        previous = poses[-2]
        poses[-1] = Pose2D(
            final.x,
            final.y,
            math.atan2(final.y - previous.y, final.x - previous.x),
        )
    return tuple(poses)


def _reconstruct_state_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def _goal_heuristic(clearance_m, state, goal_threshold):
    fx, fy, _yaw_bin = state
    return max(0.0, goal_threshold - float(clearance_m[fy, fx]))


def _yaw_to_bin(yaw, yaw_bins):
    wrapped = _wrap_angle(yaw)
    normalized = (wrapped + math.pi) / (2.0 * math.pi)
    return int(round(normalized * yaw_bins)) % yaw_bins


def _bin_to_yaw(yaw_bin, yaw_bins):
    return _wrap_angle((float(yaw_bin) / float(yaw_bins)) * 2.0 * math.pi - math.pi)


def _wrap_angle(angle_rad):
    return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


def _waypoint_length(waypoints):
    return sum(
        math.hypot(end.x - start.x, end.y - start.y)
        for start, end in zip(waypoints, waypoints[1:])
    )
