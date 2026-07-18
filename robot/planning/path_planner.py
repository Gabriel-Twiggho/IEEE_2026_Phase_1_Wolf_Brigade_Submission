"""PathPlanner facade for global and safety-escape route planning."""

from __future__ import annotations

import heapq
import math
import os
import time
from pathlib import Path

from config import SLAM_LITE_DIR, _env_float, _env_flag
from parameters import EscapePlanning as EscapePlanningParams
from parameters import Planning as PlanningParams
from robot.mapping.drone_map import DroneExtractionMap
from shared_types import (
    LiveMapSnapshot,
    PlannedPath,
    Pose2D,
)


class PathPlanner:
    """
    A* path planner for static drone maps and live occupancy snapshots.

    Inputs:
    - MapGeometry from the drone extraction data.
    - map_estimate.png for startup planning, or a LiveMapSnapshot for replanning.
    - robot start poses and victim estimates in world metres.

    Logic:
    - Inflate blocked cells so the robot centre avoids walls by a safe margin.
    - Plan with 8-neighbour A* from robot1 to the lowest-cost reachable victim.
    - Convert the raw pixel path into world coordinates.
    - Simplify the raw path into a small number of line-of-sight waypoint
      Pose2D targets so the rover does not chase every A* pixel stair-step.
    - Drop tiny near-duplicate waypoints so the route is easier for the smooth
      waypoint follower to track.

    Outputs:
    - plan_paths(...) returns a dict keyed by robot id.
    - robot*_astar_path.png visualises the selected path and waypoint dots.
    - PlannedPath.waypoints can be fed directly into WaypointNavigator.

    Important:
    - The planner API keeps a robot-id keyed shape for coordinator assignments.
    - Startup planning uses the drone map; LiveReplanner supplies lidar-updated
      snapshots for current-pose replanning.
    - Driving happens elsewhere: the mission runtime starts these waypoints in
      WaypointNavigator, and motion.py turns them into wheel commands.
    """

    ROBOT_WIDTH_M = PlanningParams.ROBOT_WIDTH_M
    ROBOT_LENGTH_M = PlanningParams.ROBOT_LENGTH_M
    OCCUPIED_PIXEL_THRESHOLD = PlanningParams.OCCUPIED_PIXEL_THRESHOLD
    WALL_CLEARANCE_M = PlanningParams.WALL_CLEARANCE_M
    WAYPOINT_SPACING_M = PlanningParams.WAYPOINT_SPACING_M
    MIN_WAYPOINT_SEPARATION_M = PlanningParams.MIN_WAYPOINT_SEPARATION_M
    SNAP_RADIUS_M = PlanningParams.SNAP_RADIUS_M
    FINE_RESOLUTION_M = EscapePlanningParams.FINE_RESOLUTION_M
    FINE_CLEARANCE_M = EscapePlanningParams.FINE_CLEARANCE_M
    ESCAPE_PATCH_SIZES_M = EscapePlanningParams.PATCH_SIZES_M
    ESCAPE_SPEED_CAP_RAD_S = EscapePlanningParams.SPEED_CAP_RAD_S
    # Clearance a post-latch escape goal must have.  Must exceed the distance at
    # which the safety layer latches (front IR mounted 0.10 m forward stops at a
    # 0.10 m reading -> obstacle ~0.20 m from robot centre), otherwise a wedged
    # pose looks "already clear" to the escape planner and it never moves.
    ESCAPE_GOAL_CLEARANCE_M = EscapePlanningParams.GOAL_CLEARANCE_M

    def __init__(
        self,
        robot_id: str,
        drone_map: DroneExtractionMap,
        enabled_by_default: bool,
        auto_plan: bool = True,
        render_overlay: bool = True,
    ):
        self.robot_id = robot_id
        self.drone_map = drone_map
        self.enabled = _env_flag("PATH_PLANNER_ENABLED", enabled_by_default)
        self.auto_plan = bool(auto_plan)
        self.render_overlay_enabled = bool(render_overlay) and _env_flag(
            "PATH_PLANNER_RENDER",
            bool(render_overlay),
        )
        self.print_plan_summary = _env_flag(
            "PATH_PRINT_PLAN_SUMMARY",
            PlanningParams.PRINT_PLAN_SUMMARY_ENABLED,
        )
        self.robot_width_m = _env_float("PATH_ROBOT_WIDTH_M", self.ROBOT_WIDTH_M)
        self.wall_clearance_m = _env_float("PATH_WALL_CLEARANCE_M", self.WALL_CLEARANCE_M)
        self.inflation_radius_m = max(self.wall_clearance_m, 0.5 * self.robot_width_m)
        self.waypoint_spacing_m = _env_float(
            "PATH_WAYPOINT_SPACING_M",
            self.WAYPOINT_SPACING_M,
        )
        self.min_waypoint_separation_m = _env_float(
            "PATH_MIN_WAYPOINT_SEPARATION_M",
            self.MIN_WAYPOINT_SEPARATION_M,
        )
        self.snap_radius_m = _env_float("PATH_SNAP_RADIUS_M", self.SNAP_RADIUS_M)
        # Legacy variable/env name from the old victim-lookout stage. It now
        # means "strict target endpoint must have enough clearance to turn."
        self.viewpoint_clearance_m = max(
            0.5 * math.hypot(self.ROBOT_LENGTH_M, self.robot_width_m),
            _env_float(
                "VICTIM_VIEWPOINT_CLEARANCE_M",
                PlanningParams.VICTIM_VIEWPOINT_CLEARANCE_M,
            ),
        )
        self.occupied_threshold = int(
            _env_float("PATH_OCCUPIED_PIXEL_THRESHOLD", self.OCCUPIED_PIXEL_THRESHOLD)
        )
        self.fine_resolution_m = max(
            0.001,
            _env_float(
                "SAFETY_ESCAPE_FINE_RESOLUTION_M",
                self.FINE_RESOLUTION_M,
            ),
        )
        self.fine_clearance_m = max(
            0.0,
            _env_float(
                "SAFETY_ESCAPE_FINE_CLEARANCE_M",
                self.FINE_CLEARANCE_M,
            ),
        )
        # Post-latch escape goal clearance.  Kept above inflation_radius_m so a
        # safety-latched pose is recognised as "needs escaping" rather than
        # "already clear" — see plan_local_escape().
        self.escape_goal_clearance_m = max(
            self.inflation_radius_m,
            _env_float("SAFETY_ESCAPE_GOAL_CLEARANCE_M", self.ESCAPE_GOAL_CLEARANCE_M),
        )
        output_dir = Path(
            os.environ.get(
                "PATH_PLANNER_DIR",
                str(SLAM_LITE_DIR),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{robot_id}_astar_path.png"

        self.geometry = drone_map.geometry if drone_map is not None else None
        self.free_grid = None
        self.width = 0
        self.height = 0
        self.inflation_radius_px = 0
        self.source_occupied_cells = frozenset()
        self.paths = {}
        self.ready = False
        self.error_message = ""
        self._viewer_process = None

        if self.enabled:
            self.ready = self._initialise()
            if self.ready and self.auto_plan:
                self.paths = self.plan_paths(
                    DroneExtractionMap.ROBOT_STARTS,
                    self.drone_map.victim_estimates,
                )
                if self.render_overlay_enabled:
                    self.render_plan_overlay(self.paths.get("robot1"))

            viewer_default = (
                self.ready
                and self.render_overlay_enabled
                and self.robot_id in ("robot1", "coordinator")
            )
            if _env_flag("PATH_PLANNER_VIEWER", viewer_default):
                self._start_viewer()

    def plan_paths(self, robot_starts, victims):
        result = {}
        if "robot1" not in robot_starts:
            result["robot1"] = PlannedPath(
                robot_id="robot1",
                success=False,
                error_reason="robot1 start pose missing",
            )
            return result

        result["robot1"] = self._plan_for_robot(
            "robot1",
            robot_starts["robot1"],
            victims,
        )
        return result

    def plan_nearest_path(
        self,
        robot_id: str,
        start_pose: Pose2D,
        victims,
        snapshot: LiveMapSnapshot = None,
    ):
        if snapshot is not None and not self.use_live_snapshot(snapshot):
            return PlannedPath(
                robot_id=robot_id,
                success=False,
                error_reason=self.error_message or "invalid live-map snapshot",
            )
        return self._plan_for_robot(robot_id, start_pose, victims)

    def plan_to_fixed_victim(
        self,
        robot_id: str,
        start_pose: Pose2D,
        victim_index: int,
        victim_world,
        snapshot: LiveMapSnapshot = None,
        log_result: bool = True,
        forbidden_cells=(),
    ):
        if snapshot is not None and not self.use_live_snapshot(snapshot):
            return PlannedPath(
                robot_id=robot_id,
                victim_index=victim_index,
                victim_world=victim_world,
                success=False,
                error_reason=self.error_message or "invalid live-map snapshot",
            )
        temporarily_blocked = self._temporarily_block_cells(forbidden_cells)
        try:
            planned_path = self._plan_target(
                robot_id,
                victim_index,
                victim_world,
                start_pose,
            )
        finally:
            self._restore_temporarily_blocked_cells(temporarily_blocked)
        if planned_path.success and log_result:
            self._log_path_plan(planned_path)
        return planned_path

    def plan_to_fixed_pose(
        self,
        robot_id: str,
        start_pose: Pose2D,
        target_pose: Pose2D,
        target_index: int = 0,
        snapshot: LiveMapSnapshot = None,
        log_result: bool = True,
        strict_target: bool = False,
        forbidden_cells=(),
    ):
        """Plan to a coordinate while preserving a requested final camera yaw."""
        if strict_target:
            if snapshot is not None and not self.use_live_snapshot(snapshot):
                return PlannedPath(
                    robot_id=robot_id,
                    victim_index=target_index,
                    victim_world=(target_pose.x, target_pose.y),
                    success=False,
                    error_reason=self.error_message or "invalid live-map snapshot",
                )
            raw_goal = self.geometry.world_to_pixel(target_pose.x, target_pose.y)
            if not self._strict_viewpoint_cell_is_valid(raw_goal):
                return PlannedPath(
                    robot_id=robot_id,
                    victim_index=target_index,
                    victim_world=(target_pose.x, target_pose.y),
                    success=False,
                    error_reason=(
                        "strict target is blocked or lacks turning clearance"
                    ),
                )
        planned_path = self.plan_to_fixed_victim(
            robot_id,
            start_pose,
            target_index,
            (target_pose.x, target_pose.y),
            snapshot,
            log_result=log_result,
            forbidden_cells=forbidden_cells,
        )
        if not planned_path.success:
            return planned_path
        waypoints = list(planned_path.waypoints)
        speed_caps = list(planned_path.waypoint_speed_caps)
        if waypoints:
            final = waypoints[-1]
            waypoints[-1] = Pose2D(final.x, final.y, target_pose.yaw)
        else:
            waypoints.append(
                Pose2D(target_pose.x, target_pose.y, target_pose.yaw)
            )
            speed_caps.append(None)
        planned_path.waypoints = tuple(waypoints)
        planned_path.waypoint_speed_caps = tuple(speed_caps)
        planned_path.final_target_yaw = float(target_pose.yaw)
        return planned_path

    def estimate_route_cost(
        self,
        start_pose: Pose2D,
        target_pose: Pose2D,
        snapshot: LiveMapSnapshot = None,
    ):
        """Fast coarse-grid reachability/cost check used for candidate ranking."""
        if snapshot is not None and not self.use_live_snapshot(snapshot):
            return False, float("inf")
        raw_start = self.geometry.world_to_pixel(start_pose.x, start_pose.y)
        raw_goal = self.geometry.world_to_pixel(target_pose.x, target_pose.y)
        start = self._nearest_free_cell(raw_start)
        goal = self._nearest_free_cell(raw_goal)
        if start is None or goal is None:
            return False, float("inf")
        pixel_path, cost_px = self._astar(start, goal)
        if not pixel_path:
            return False, float("inf")
        return True, cost_px * self.geometry.resolution

    def _strict_viewpoint_cell_is_valid(self, cell):
        return self._is_free(cell) and self._cell_has_source_clearance(
            cell,
            self.viewpoint_clearance_m,
        )

    def use_live_snapshot(self, snapshot: LiveMapSnapshot):
        if snapshot is None or snapshot.geometry is None:
            self.error_message = "live-map snapshot has no map geometry"
            return False
        if snapshot.width <= 0 or snapshot.height <= 0:
            self.error_message = "live-map snapshot has invalid dimensions"
            return False

        self.geometry = snapshot.geometry
        self.width = snapshot.width
        self.height = snapshot.height
        self.inflation_radius_px = int(
            math.ceil(self.inflation_radius_m * self.geometry.pixels_per_metre)
        )
        self.source_occupied_cells = frozenset(snapshot.occupied_cells)
        self.free_grid = self._build_inflated_free_grid_from_cells(
            self.source_occupied_cells
        )
        return True

    def remaining_path_blocking_cells(
        self,
        planned_path: PlannedPath,
        current_pose: Pose2D,
        snapshot: LiveMapSnapshot,
        ignored_occupied_cells=(),
    ):
        if (
            planned_path is None
            or not planned_path.success
            or not planned_path.pixel_path
            or snapshot is None
            or snapshot.geometry is None
        ):
            return ()

        self.geometry = snapshot.geometry
        current_cell = self.geometry.world_to_pixel(current_pose.x, current_pose.y)
        nearest_index = min(
            range(len(planned_path.pixel_path)),
            key=lambda index: (
                planned_path.pixel_path[index][0] - current_cell[0]
            ) ** 2
            + (
                planned_path.pixel_path[index][1] - current_cell[1]
            ) ** 2,
        )
        remaining = planned_path.pixel_path[nearest_index:]
        inflation_radius_px = int(
            math.ceil(self.inflation_radius_m * self.geometry.pixels_per_metre)
        )
        offsets = self._inflation_offsets(inflation_radius_px)
        occupied = snapshot.occupied_cells
        if ignored_occupied_cells:
            occupied = occupied.difference(ignored_occupied_cells)
        blocked = []
        for px, py in remaining:
            if any((px + dx, py + dy) in occupied for dx, dy in offsets):
                blocked.append((px, py))
        return tuple(blocked)

    def _temporarily_block_cells(self, cells):
        blocked = []
        if not cells or self.free_grid is None:
            return blocked
        for px, py in cells:
            if (
                0 <= px < self.width
                and 0 <= py < self.height
                and self.free_grid[py][px]
            ):
                self.free_grid[py][px] = 0
                blocked.append((px, py))
        return blocked

    def _restore_temporarily_blocked_cells(self, cells):
        if self.free_grid is None:
            return
        for px, py in cells:
            self.free_grid[py][px] = 1

    def remaining_pixel_path(self, planned_path: PlannedPath, current_pose: Pose2D):
        if planned_path is None or not planned_path.pixel_path:
            return ()
        current_cell = self.geometry.world_to_pixel(current_pose.x, current_pose.y)
        nearest_index = min(
            range(len(planned_path.pixel_path)),
            key=lambda index: (
                planned_path.pixel_path[index][0] - current_cell[0]
            ) ** 2
            + (
                planned_path.pixel_path[index][1] - current_cell[1]
            ) ** 2,
        )
        return tuple(planned_path.pixel_path[nearest_index:])

    def render_plan_overlay(self, planned_path: PlannedPath):
        from robot.planning.render import render_plan_overlay

        render_plan_overlay(self, planned_path)

    def _start_viewer(self):
        from robot.planning.render import start_viewer

        self._viewer_process = start_viewer(
            self.output_path,
            f"{self.robot_id} A* path",
            f"[{self.robot_id}] Could not start path planner viewer",
        )

    def _stop_viewer(self):
        from robot.planning.render import stop_viewer

        stop_viewer(self._viewer_process)

    def _initialise(self):
        if self.drone_map is None or not self.drone_map.ready:
            self.error_message = "drone map is not ready"
            print(f"[{self.robot_id}] Path planner disabled: {self.error_message}")
            return False

        try:
            from PIL import Image
        except ImportError as exc:
            self.error_message = f"PIL unavailable: {exc}"
            print(f"[{self.robot_id}] Path planner disabled: {self.error_message}")
            return False

        try:
            image = Image.open(self.drone_map.map_path).convert("L")
        except OSError as exc:
            self.error_message = str(exc)
            print(f"[{self.robot_id}] Path planner disabled: {self.error_message}")
            return False

        self.width, self.height = image.size
        self.inflation_radius_px = int(
            math.ceil(self.inflation_radius_m * self.geometry.pixels_per_metre)
        )
        self.source_occupied_cells = frozenset(self._occupied_cells_from_image(image))
        self.free_grid = self._build_inflated_free_grid_from_cells(
            self.source_occupied_cells
        )
        return True

    def _occupied_cells_from_image(self, image):
        occupied_pixels = []
        pixels = image.load()

        for py in range(self.height):
            for px in range(self.width):
                if pixels[px, py] < self.occupied_threshold:
                    occupied_pixels.append((px, py))
        return occupied_pixels

    def _build_inflated_free_grid_from_cells(
        self,
        occupied_cells,
        radius_px=None,
        width=None,
        height=None,
    ):
        width = self.width if width is None else int(width)
        height = self.height if height is None else int(height)
        radius_px = (
            self.inflation_radius_px
            if radius_px is None
            else max(0, int(radius_px))
        )
        free_grid = [bytearray([1]) * width for _ in range(height)]
        offsets = self._inflation_offsets(radius_px)
        for px, py in occupied_cells:
            for dx, dy in offsets:
                xx = px + dx
                yy = py + dy
                if 0 <= xx < width and 0 <= yy < height:
                    free_grid[yy][xx] = 0

        return free_grid

    def _plan_for_robot(self, robot_id: str, start_pose: Pose2D, victims):
        planning_started = time.perf_counter()
        if not victims:
            return PlannedPath(
                robot_id=robot_id,
                success=False,
                error_reason="no victim estimates available",
            )

        best = None
        for victim_index, victim_world in enumerate(victims, start=1):
            candidate = self._plan_target(
                robot_id,
                victim_index,
                victim_world,
                start_pose,
            )
            if not candidate.success:
                continue

            if (
                best is None
                or candidate.weighted_path_cost_m
                < best.weighted_path_cost_m
            ):
                best = candidate

        if best is None:
            return PlannedPath(
                robot_id=robot_id,
                success=False,
                error_reason="no reachable victim found",
            )
        best.planning_time_s = time.perf_counter() - planning_started
        self._log_path_plan(best)
        return best

    def _plan_target(
        self,
        robot_id,
        victim_index,
        victim_world,
        start_pose,
    ):
        planning_started = time.perf_counter()
        raw_start = self.geometry.world_to_pixel(start_pose.x, start_pose.y)
        raw_goal = self.geometry.world_to_pixel(victim_world[0], victim_world[1])

        selected = None
        safe_start = self._nearest_free_cell(raw_start)
        safe_goal = self._nearest_free_cell(raw_goal)
        if safe_start is not None and safe_goal is not None:
            safe_pixels, _ = self._astar(safe_start, safe_goal)
            if safe_pixels:
                selected = self._planned_path_from_pixels(
                    robot_id,
                    victim_index,
                    victim_world,
                    safe_pixels,
                )

        if selected is None:
            selected = PlannedPath(
                robot_id=robot_id,
                victim_index=victim_index,
                victim_world=victim_world,
                success=False,
                error_reason=f"no route from {raw_start} to {raw_goal}",
            )

        selected.planning_time_s = time.perf_counter() - planning_started
        return selected

    def _log_path_plan(self, planned_path):
        if not self.print_plan_summary:
            return
        print(
            f"[{self.robot_id}] Path plan {planned_path.planning_mode}: "
            f"physical={planned_path.physical_path_cost_m:.2f} m "
            f"weighted={planned_path.weighted_path_cost_m:.2f} m "
            f"time={planned_path.planning_time_s:.3f} s"
        )

    def _planned_path_from_pixels(
        self,
        robot_id,
        victim_index,
        victim_world,
        pixel_path,
    ):
        world_path = tuple(self.geometry.pixel_to_world(px, py) for px, py in pixel_path)
        waypoints = self._waypoints_from_pixel_path(pixel_path)
        physical_cost = self._polyline_length(world_path)
        return PlannedPath(
            robot_id=robot_id,
            victim_index=victim_index,
            victim_world=victim_world,
            pixel_path=tuple(pixel_path),
            world_path=world_path,
            waypoints=tuple(waypoints),
            waypoint_speed_caps=tuple(None for _ in waypoints),
            planning_mode="GLOBAL",
            physical_path_cost_m=physical_cost,
            weighted_path_cost_m=physical_cost,
            success=True,
            error_reason="",
        )

    def plan_local_escape(
        self,
        current_pose: "Pose2D",
        snapshot: "LiveMapSnapshot",
    ) -> "PlannedPath":
        from robot.planning.escape import plan_local_escape

        return plan_local_escape(self, current_pose, snapshot)

    @staticmethod
    def _euclidean_clearance_pixels(occupied):
        from robot.planning.multires import euclidean_clearance_pixels

        return euclidean_clearance_pixels(occupied)

    @staticmethod
    def _coarse_to_fine_cell(cell, min_px, min_py, scale):
        return (
            (cell[0] - min_px) * scale + scale // 2,
            (cell[1] - min_py) * scale + scale // 2,
        )

    def _fine_cell_to_world(self, cell, min_px, min_py, scale):
        global_px = min_px + (cell[0] + 0.5) / scale - 0.5
        global_py = min_py + (cell[1] + 0.5) / scale - 0.5
        return self.geometry.pixel_to_world(global_px, global_py)

    @staticmethod
    def _polyline_length(points):
        from robot.planning.waypoints import polyline_length

        return polyline_length(points)

    def _astar(self, start, goal):
        return self._astar_on_grid(
            self.free_grid,
            self.width,
            self.height,
            start,
            goal,
        )

    def _astar_on_grid(
        self,
        free_grid,
        width,
        height,
        start,
        goal,
        prevent_corner_cutting=False,
    ):
        open_heap = []
        counter = 0
        heapq.heappush(open_heap, (self._octile_distance(start, goal), 0.0, counter, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        neighbours = (
            (-1, -1, math.sqrt(2.0)),
            (0, -1, 1.0),
            (1, -1, math.sqrt(2.0)),
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (-1, 1, math.sqrt(2.0)),
            (0, 1, 1.0),
            (1, 1, math.sqrt(2.0)),
        )

        while open_heap:
            _, current_g, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self._reconstruct_path(came_from, current), current_g

            closed.add(current)
            cx, cy = current
            for dx, dy, move_cost in neighbours:
                neighbour = (cx + dx, cy + dy)
                nx, ny = neighbour
                if (
                    nx < 0
                    or nx >= width
                    or ny < 0
                    or ny >= height
                    or not free_grid[ny][nx]
                    or neighbour in closed
                ):
                    continue
                if (
                    prevent_corner_cutting
                    and dx != 0
                    and dy != 0
                    and (
                        not free_grid[cy][nx]
                        or not free_grid[ny][cx]
                    )
                ):
                    continue

                tentative_g = current_g + move_cost
                if tentative_g >= g_score.get(neighbour, float("inf")):
                    continue

                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                counter += 1
                f_score = tentative_g + self._octile_distance(neighbour, goal)
                heapq.heappush(open_heap, (f_score, tentative_g, counter, neighbour))

        return (), float("inf")

    def _nearest_free_cell(self, cell):
        return self._nearest_free_cell_on_grid(
            cell,
            self.free_grid,
            self.width,
            self.height,
            self.snap_radius_m,
            self.geometry.pixels_per_metre,
        )

    def _nearest_free_cell_on_grid(
        self,
        cell,
        free_grid,
        width,
        height,
        snap_radius_m,
        pixels_per_metre,
    ):
        if self._is_free_on_grid(cell, free_grid, width, height):
            return cell

        max_radius_px = int(
            math.ceil(snap_radius_m * pixels_per_metre)
        )
        cx, cy = cell
        best = None
        best_dist2 = None

        for radius in range(1, max_radius_px + 1):
            for yy in range(cy - radius, cy + radius + 1):
                for xx in range(cx - radius, cx + radius + 1):
                    if (
                        abs(xx - cx) != radius
                        and abs(yy - cy) != radius
                    ):
                        continue
                    candidate = (xx, yy)
                    if not self._is_free_on_grid(
                        candidate,
                        free_grid,
                        width,
                        height,
                    ):
                        continue
                    dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
                    if best is None or dist2 < best_dist2:
                        best = candidate
                        best_dist2 = dist2

            if best is not None:
                return best

        return None

    def _is_free(self, cell):
        return self._is_free_on_grid(
            cell,
            self.free_grid,
            self.width,
            self.height,
        )

    def _cell_has_source_clearance(self, cell, clearance_m):
        """Require a full circular footprint around a strict target cell."""
        radius_px = max(
            0.0,
            float(clearance_m) * self.geometry.pixels_per_metre,
        )
        search_px = int(math.ceil(radius_px))
        cx, cy = cell
        for dy in range(-search_px, search_px + 1):
            for dx in range(-search_px, search_px + 1):
                if dx * dx + dy * dy > radius_px * radius_px:
                    continue
                px = cx + dx
                py = cy + dy
                if (
                    px < 0
                    or px >= self.width
                    or py < 0
                    or py >= self.height
                    or (px, py) in self.source_occupied_cells
                ):
                    return False
        return True

    @staticmethod
    def _is_free_on_grid(cell, free_grid, width, height):
        px, py = cell
        return (
            0 <= px < width
            and 0 <= py < height
            and bool(free_grid[py][px])
        )

    def _waypoints_from_pixel_path(self, pixel_path):
        if len(pixel_path) < 2:
            return ()

        simplified_pixels = self._line_of_sight_pixels(pixel_path)
        simplified_world = [
            self.geometry.pixel_to_world(px, py)
            for px, py in simplified_pixels
        ]
        waypoint_points = self._densify_world_points(simplified_world)
        waypoint_points = self._drop_close_intermediate_points(waypoint_points)
        waypoints = []

        for i in range(1, len(waypoint_points)):
            current = waypoint_points[i]
            if i + 1 < len(waypoint_points):
                next_point = waypoint_points[i + 1]
                yaw = math.atan2(next_point[1] - current[1], next_point[0] - current[0])
            else:
                previous = waypoint_points[i - 1]
                yaw = math.atan2(current[1] - previous[1], current[0] - previous[0])
            waypoints.append(Pose2D(x=current[0], y=current[1], yaw=yaw))

        return waypoints

    def _line_of_sight_pixels(self, pixel_path):
        if len(pixel_path) <= 2:
            return list(pixel_path)

        simplified = [pixel_path[0]]
        anchor_index = 0
        last_index = len(pixel_path) - 1

        while anchor_index < last_index:
            next_index = last_index
            while (
                next_index > anchor_index + 1
                and not self._line_is_free(pixel_path[anchor_index], pixel_path[next_index])
            ):
                next_index -= 1

            simplified.append(pixel_path[next_index])
            anchor_index = next_index

        return simplified

    def _line_is_free(self, start, end):
        for px, py in self._bresenham_cells(start[0], start[1], end[0], end[1]):
            if not self._is_free((px, py)):
                return False
        return True

    def _densify_world_points(self, points):
        if not points:
            return []

        result = [points[0]]
        if self.waypoint_spacing_m <= 0.0:
            return result + list(points[1:])

        max_spacing = max(self.waypoint_spacing_m, self.geometry.resolution)
        for start, end in zip(points, points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            distance = math.hypot(dx, dy)
            steps = max(1, int(math.ceil(distance / max_spacing)))
            for step in range(1, steps + 1):
                ratio = step / steps
                result.append((start[0] + ratio * dx, start[1] + ratio * dy))
        return result

    def _drop_close_intermediate_points(self, points):
        if len(points) <= 2 or self.min_waypoint_separation_m <= 0.0:
            return points

        filtered = [points[0]]
        for point in points[1:-1]:
            if self._world_distance(filtered[-1], point) >= self.min_waypoint_separation_m:
                filtered.append(point)

        final = points[-1]
        if self._world_distance(filtered[-1], final) < self.min_waypoint_separation_m:
            filtered[-1] = final
        else:
            filtered.append(final)

        return filtered

    @staticmethod
    def _inflation_offsets(radius_px):
        from robot.planning.grid_search import inflation_offsets

        return inflation_offsets(radius_px)

    @staticmethod
    def _reconstruct_path(came_from, current):
        from robot.planning.grid_search import reconstruct_path

        return reconstruct_path(came_from, current)

    @staticmethod
    def _bresenham_cells(x0: int, y0: int, x1: int, y1: int):
        from robot.planning.grid_search import bresenham_cells

        return bresenham_cells(x0, y0, x1, y1)

    @staticmethod
    def _octile_distance(a, b):
        from robot.planning.grid_search import octile_distance

        return octile_distance(a, b)

    @staticmethod
    def _world_distance(a, b):
        from robot.planning.grid_search import world_distance

        return world_distance(a, b)
