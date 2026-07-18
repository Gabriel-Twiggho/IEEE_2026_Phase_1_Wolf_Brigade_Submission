"""Live route replan state machine for the robot runtime."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from config import SLAM_LITE_DIR, _env_float, _env_flag, _env_int
from parameters import LiveReplanning as LiveReplanningParams
from shared_types import LiveMapSnapshot, LiveReplanEvent, PlannedPath, Pose2D

if TYPE_CHECKING:
    from robot.mapping.live_grid import LiveOccupancyGridMapper
    from robot.planning.path_planner import PathPlanner


class LiveReplanner:
    """
    Per-robot live-occupancy dynamic replanning coordinator.

    The live mapper owns lidar, depth, and IR occupancy. PathPlanner owns
    inflation/A*, and this class decides when an active route must be replaced.
    Normal map changes are checked at a conservative interval.

    Normal live-map replanning workflow:
    1. A caller supplies a victim, test, prior, approach, or close-in target.
       The C-key victim mission and B-key straight test both use this interface.
    2. The live map revision changes when lidar/depth/IR occupancy changes.
    3. At LIVE_REPLAN_INTERVAL_S, check whether newly occupied, inflated cells
       intersect the remaining portion of the active pixel path.
    4. If the remaining path is still clear, keep following it.
    5. If it is blocked, run A* from the current pose to the same retained target.
    6. Adopt the replacement automatically by default. Debug mode can require N.

    Safety recovery workflow:
    - A safety latch stops the motors and records the depth/IR obstacle.
    - SafetyRecoveryController asks PathPlanner once for a yaw-aware 1 cm/px
      local escape route from the current wedged pose.
    - Once an escape route finishes, replan_after_recovery() runs A* from the
      new pose back to the retained target.
    - If no local escape exists, the route enters FAILED and the robot remains
      stopped for the coordinator to handle.

    Live-replanning image colour key:
    - all LiveOccupancyGridMapper colours remain visible underneath
    - magenta line: remaining portion of the current/old route
    - green line: coordinator route comparison, when available
    - cyan line and dots: proposed replacement route and its waypoints
    - bright-red squares: current-route cells classified as blocked
    - green outline/line: robot footprint and heading
    - yellow cross: retained victim or temporary test target
    - the legend reports recovery attempts used and remaining for this route
    - all labels and status text are drawn in a panel to the right of the map,
      so occupancy cells and paths are never hidden behind text

    Display refresh:
    - The route overlay is refreshed whenever LiveOccupancyGridMapper renders
      its normal image (0.50 s by default).
    - This keeps the robot footprint moving between waypoints without adding a
      second lidar/map update or extra map-rendering pass.

    Note that red has two related but distinct uses:
    - ordinary red map cells are depth-camera obstacles
    - larger bright-red squares are blocked cells on the current route

    Replacements are adopted automatically by default for competition autonomy.
    Set LIVE_REPLAN_REQUIRE_APPROVAL=1 during debugging to pause on a valid
    replacement and inspect the old/proposed paths before pressing N.
    """

    IDLE = "IDLE"
    FOLLOWING = "FOLLOWING"
    PENDING_REPLAN = "PENDING_REPLAN"
    FAILED = "FAILED"

    ROBOT_LENGTH_M = LiveReplanningParams.ROBOT_LENGTH_M
    ROBOT_WIDTH_M = LiveReplanningParams.ROBOT_WIDTH_M
    REPLAN_INTERVAL_S = LiveReplanningParams.REPLAN_INTERVAL_S
    COORDINATOR_LOCAL_RETRY_COUNT = (
        LiveReplanningParams.COORDINATOR_LOCAL_RETRY_COUNT
    )
    COORDINATOR_START_CLEAR_RADIUS_M = (
        LiveReplanningParams.COORDINATOR_START_CLEAR_RADIUS_M
    )
    INFO_PANEL_WIDTH_PX = LiveReplanningParams.INFO_PANEL_WIDTH_PX

    def __init__(
        self,
        robot_id: str,
        planner: PathPlanner,
        live_map: LiveOccupancyGridMapper,
        victims,
        enabled_by_default: bool,
        recovery_controller=None,
    ):
        self.robot_id = robot_id
        self.planner = planner
        self.live_map = live_map
        self.victims = tuple(victims)
        self.enabled = (
            planner.ready
            and live_map.enabled
            and _env_flag("LIVE_REPLAN_ENABLED", enabled_by_default)
        )
        self.interval_s = _env_float(
            "LIVE_REPLAN_INTERVAL_S",
            self.REPLAN_INTERVAL_S,
        )
        self.coordinator_local_retry_count = min(
            3,
            max(
                0,
                _env_int(
                    "COORDINATOR_LOCAL_RETRY_COUNT",
                    self.COORDINATOR_LOCAL_RETRY_COUNT,
                ),
            ),
        )
        self.coordinator_start_clear_radius_m = max(
            0.0,
            _env_float(
                "COORDINATOR_START_CLEAR_RADIUS_M",
                self.COORDINATOR_START_CLEAR_RADIUS_M,
            ),
        )
        self.require_approval = _env_flag(
            "LIVE_REPLAN_REQUIRE_APPROVAL",
            LiveReplanningParams.REQUIRE_APPROVAL,
        )
        self.render_enabled = (
            _env_flag("LIVE_REPLAN_RENDER", True)
            and getattr(live_map, "render_enabled", True)
        )

        output_dir = Path(
            os.environ.get(
                "LIVE_REPLAN_DIR",
                str(SLAM_LITE_DIR),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{robot_id}_live_replan.png"

        self.state = self.IDLE
        self.active_path = None
        self.candidate_path = None
        self.failed_candidate_path = None
        self.fixed_target_strict = False
        self.blocking_cells = ()
        self.last_check_time = -self.interval_s
        self.last_snapshot_revision = -1
        self.reason = ""
        self.recovery_controller = recovery_controller
        self.victim_status_provider = None
        self._start_clear_origin = None
        self._start_clear_radius_m = 0.0
        self._start_clear_ignored_cells = frozenset()
        self._viewer_process = None

        viewer_default = self.enabled and self.render_enabled and robot_id == "robot1"
        if _env_flag("LIVE_REPLAN_VIEWER", viewer_default):
            self._start_viewer()

    def refresh_visual(
        self,
        current_pose: Pose2D,
        base_map_rendered: bool,
    ):
        """
        Refresh the route overlay after the base live-map image was regenerated.

        Reusing that freshly written image avoids another occupancy-grid render.
        """
        if not self.enabled or not self.render_enabled or not base_map_rendered:
            return
        self.render(current_pose, refresh_base_map=False)

    def follow_preplanned_path(
        self,
        planned_path: PlannedPath,
        current_pose: Pose2D,
        sim_time: float,
        reason: str = "coordinator preplanned route",
    ):
        """Adopt a coordinator-provided route for local blockage checks."""
        self._reset_start_clearance_exemption()
        if not self.enabled:
            return LiveReplanEvent("START", planned_path, "live replanning is unavailable")
        if planned_path is None or not planned_path.success:
            self.state = self.FAILED
            self.active_path = None
            self.reason = (
                planned_path.error_reason
                if planned_path is not None
                else "missing preplanned path"
            )
            return LiveReplanEvent("FAILED", planned_path, self.reason)

        snapshot = self.live_map.planning_snapshot()
        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision if snapshot is not None else -1
        self.blocking_cells = ()
        self.candidate_path = None
        self.failed_candidate_path = None
        self.state = self.FOLLOWING
        self.active_path = planned_path
        self.fixed_target_strict = False

        if snapshot is not None:
            self.blocking_cells = self._remaining_path_blocking_cells(
                planned_path,
                current_pose,
                snapshot,
            )
            if self.blocking_cells:
                blocked_count = len(self.blocking_cells)
                candidate = self._plan_active_target(current_pose, snapshot)
                if not candidate.success:
                    initial_error = candidate.error_reason
                    (
                        candidate,
                        retry_number,
                        retry_policy,
                        cleared_cells,
                    ) = self._retry_blocked_coordinator_route(
                        current_pose,
                        snapshot,
                        candidate,
                    )
                    if not candidate.success:
                        self.state = self.FAILED
                        self.candidate_path = None
                        self.failed_candidate_path = candidate
                        self.reason = (
                            f"coordinator route blocked locally ({blocked_count} cells): "
                            f"normal local plan failed ({initial_error}); "
                            f"{retry_number} start-clear retries failed "
                            f"({candidate.error_reason})"
                        )
                        self._reset_start_clearance_exemption()
                        self.render(current_pose)
                        return LiveReplanEvent("FAILED", candidate, self.reason)

                    self._set_start_clearance_exemption(
                        current_pose,
                        cleared_cells,
                    )
                    self.active_path = candidate
                    self.blocking_cells = ()
                    self.reason = (
                        f"{blocked_count} coordinator path cells blocked; "
                        f"local retry {retry_number}/{self.coordinator_local_retry_count} "
                        f"adopted ({retry_policy}, cleared {len(cleared_cells)} cells)"
                    )
                    print(f"[{self.robot_id}] LIVE {self.reason}")
                    self.render(current_pose)
                    return LiveReplanEvent("START", candidate, self.reason)

                self.active_path = candidate
                self.blocking_cells = ()
                self.reason = (
                    f"{blocked_count} coordinator path cells blocked; "
                    "local replacement adopted"
                )
                self.render(current_pose)
                return LiveReplanEvent("START", candidate, self.reason)

        self.reason = reason
        self.render(current_pose)
        return LiveReplanEvent("START", planned_path, self.reason)

    def start(self, current_pose: Pose2D, sim_time: float):
        self._reset_start_clearance_exemption()
        if not self.enabled:
            self.state = self.FAILED
            self.reason = "live replanning is unavailable"
            return LiveReplanEvent("FAILED", reason=self.reason)

        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            self.state = self.FAILED
            self.reason = "live map has no planning snapshot"
            return LiveReplanEvent("FAILED", reason=self.reason)

        planned_path = self.planner.plan_nearest_path(
            self.robot_id,
            current_pose,
            self.victims,
            snapshot,
        )
        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision
        self.blocking_cells = ()
        self.candidate_path = None
        self.failed_candidate_path = None

        if not planned_path.success:
            self.state = self.FAILED
            self.active_path = None
            self.failed_candidate_path = planned_path
            self.reason = planned_path.error_reason
            self.render(current_pose)
            return LiveReplanEvent("FAILED", planned_path, self.reason)

        self.state = self.FOLLOWING
        self.active_path = planned_path
        self.failed_candidate_path = None
        self.fixed_target_strict = False
        self.reason = "initial live-map route"
        self.render(current_pose)
        return LiveReplanEvent("START", planned_path, self.reason)

    def start_fixed_target(
        self,
        current_pose: Pose2D,
        sim_time: float,
        target_world,
        target_index: int = 0,
    ):
        """
        Start a live-map route to an explicit world coordinate.

        This uses the same A*, blockage checks, safety recovery, and subsequent
        replanning as a victim route. target_index=0 marks the destination as a
        temporary test target in the live-map visualisation.
        """
        self._reset_start_clearance_exemption()
        if not self.enabled:
            self.state = self.FAILED
            self.reason = "live replanning is unavailable"
            return LiveReplanEvent("FAILED", reason=self.reason)

        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            self.state = self.FAILED
            self.reason = "live map has no planning snapshot"
            return LiveReplanEvent("FAILED", reason=self.reason)

        planned_path = self.planner.plan_to_fixed_victim(
            self.robot_id,
            current_pose,
            target_index,
            tuple(target_world),
            snapshot,
        )
        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision
        self.blocking_cells = ()
        self.candidate_path = None
        self.failed_candidate_path = None

        if not planned_path.success:
            self.state = self.FAILED
            self.active_path = None
            self.failed_candidate_path = planned_path
            self.reason = planned_path.error_reason
            self.render(current_pose)
            return LiveReplanEvent("FAILED", planned_path, self.reason)

        self.state = self.FOLLOWING
        self.active_path = planned_path
        self.failed_candidate_path = None
        self.fixed_target_strict = False
        self.reason = "fixed live-map test target"
        self.render(current_pose)
        return LiveReplanEvent("START", planned_path, self.reason)

    def start_fixed_pose(
        self,
        current_pose: Pose2D,
        sim_time: float,
        target_pose: Pose2D,
        target_index: int = 0,
        strict_target: bool = False,
    ):
        """Start a live route that must finish at a requested final yaw."""
        self._reset_start_clearance_exemption()
        if not self.enabled:
            self.state = self.FAILED
            self.reason = "live replanning is unavailable"
            return LiveReplanEvent("FAILED", reason=self.reason)
        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            self.state = self.FAILED
            self.reason = "live map has no planning snapshot"
            return LiveReplanEvent("FAILED", reason=self.reason)
        planned_path = self.planner.plan_to_fixed_pose(
            self.robot_id,
            current_pose,
            target_pose,
            target_index,
            snapshot,
            strict_target=strict_target,
        )
        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision
        self.blocking_cells = ()
        self.candidate_path = None
        self.failed_candidate_path = None
        if not planned_path.success:
            self.state = self.FAILED
            self.active_path = None
            self.failed_candidate_path = planned_path
            self.reason = planned_path.error_reason
            self.render(current_pose)
            return LiveReplanEvent("FAILED", planned_path, self.reason)
        self.state = self.FOLLOWING
        self.active_path = planned_path
        self.failed_candidate_path = None
        self.fixed_target_strict = bool(strict_target)
        self.reason = "fixed-pose victim mission target"
        self.render(current_pose)
        return LiveReplanEvent("START", planned_path, self.reason)

    def start_best_fixed_pose(
        self,
        current_pose: Pose2D,
        sim_time: float,
        target_poses,
        target_index: int = 0,
    ):
        """Plan fixed-pose candidates once and start the lowest-cost route."""
        self._reset_start_clearance_exemption()
        if not self.enabled:
            self.state = self.FAILED
            self.reason = "live replanning is unavailable"
            return LiveReplanEvent("FAILED", reason=self.reason)
        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            self.state = self.FAILED
            self.reason = "live map has no planning snapshot"
            return LiveReplanEvent("FAILED", reason=self.reason)

        targets = tuple(target_poses or ())
        candidates = []
        failures = []
        for candidate_index, target_pose in enumerate(targets):
            planned_path = self.planner.plan_to_fixed_pose(
                self.robot_id,
                current_pose,
                target_pose,
                target_index,
                snapshot,
                log_result=False,
                strict_target=False,
            )
            if planned_path.success:
                candidates.append((candidate_index, planned_path))
            else:
                failures.append((candidate_index, planned_path))

        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision
        self.blocking_cells = ()
        self.candidate_path = None
        self.failed_candidate_path = None
        if not candidates:
            failed_path = (
                failures[-1][1]
                if failures
                else PlannedPath(
                    robot_id=self.robot_id,
                    victim_index=target_index,
                    success=False,
                )
            )
            details = "; ".join(
                f"candidate {index + 1}: {path.error_reason or 'no route'}"
                for index, path in failures
            )
            self.reason = (
                f"no reachable side-view approach ({details})"
                if details
                else "no side-view approach candidates"
            )
            failed_path.error_reason = self.reason
            self.state = self.FAILED
            self.active_path = None
            self.failed_candidate_path = failed_path
            self.render(current_pose)
            return LiveReplanEvent("FAILED", failed_path, self.reason)

        selected_index, selected_path = min(
            candidates,
            key=lambda item: (
                float(item[1].weighted_path_cost_m),
                float(item[1].physical_path_cost_m),
                item[0],
            ),
        )
        self.state = self.FOLLOWING
        self.active_path = selected_path
        self.fixed_target_strict = False
        self.reason = (
            f"side-view approach candidate {selected_index + 1}/{len(targets)} "
            "selected by route cost"
        )
        self.render(current_pose)
        return LiveReplanEvent("START", selected_path, self.reason)

    def update(self, current_pose: Pose2D, sim_time: float):
        self._active_start_clear_ignored_cells(current_pose)
        if self.state != self.FOLLOWING or self.active_path is None:
            return LiveReplanEvent()
        if sim_time - self.last_check_time < self.interval_s:
            return LiveReplanEvent()

        self.last_check_time = sim_time
        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            return LiveReplanEvent()
        if snapshot.revision == self.last_snapshot_revision:
            return LiveReplanEvent()

        self.last_snapshot_revision = snapshot.revision
        self.blocking_cells = self._remaining_path_blocking_cells(
            self.active_path,
            current_pose,
            snapshot,
        )
        if not self.blocking_cells:
            return LiveReplanEvent()

        candidate = self._plan_active_target(current_pose, snapshot)
        if not candidate.success:
            self.state = self.FAILED
            self.candidate_path = None
            self.failed_candidate_path = candidate
            self.reason = candidate.error_reason
            self._reset_start_clearance_exemption()
            self.render(current_pose)
            return LiveReplanEvent("FAILED", candidate, self.reason)

        self.candidate_path = candidate
        self.failed_candidate_path = None
        self.reason = (
            f"{len(self.blocking_cells)} remaining path cells are blocked"
        )
        if self.require_approval:
            self.state = self.PENDING_REPLAN
            self.render(current_pose)
            return LiveReplanEvent("PENDING", candidate, self.reason)

        self.active_path = candidate
        self._reset_start_clearance_exemption()
        self.candidate_path = None
        self.failed_candidate_path = None
        self.blocking_cells = ()
        self.state = self.FOLLOWING
        self.render(current_pose)
        return LiveReplanEvent("ADOPT", candidate, self.reason)

    def replan_after_recovery(
        self,
        current_pose: Pose2D,
        sim_time: float,
        recovery_reason: str,
    ):
        """
        Replan immediately to the retained victim after a safety escape.

        Recovery replacements are adopted automatically because competition
        operation cannot wait for the normal N-key approval workflow.
        """
        if self.active_path is None:
            return LiveReplanEvent(
                "FAILED",
                reason="no active victim is available for recovery replanning",
            )

        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            return LiveReplanEvent(
                "FAILED",
                reason="live map has no planning snapshot",
            )

        self.last_check_time = sim_time
        self.last_snapshot_revision = snapshot.revision
        self._reset_start_clearance_exemption()
        self.blocking_cells = self._remaining_path_blocking_cells(
            self.active_path,
            current_pose,
            snapshot,
        )
        candidate = self._plan_active_target(current_pose, snapshot)
        if not candidate.success:
            self.candidate_path = None
            self.failed_candidate_path = candidate
            self.state = self.FOLLOWING
            self.reason = f"{recovery_reason}: {candidate.error_reason}"
            self.render(current_pose)
            return LiveReplanEvent("FAILED", candidate, self.reason)

        self.active_path = candidate
        self.candidate_path = None
        self.failed_candidate_path = None
        self.blocking_cells = ()
        self.state = self.FOLLOWING
        self.reason = recovery_reason
        self.render(current_pose)
        return LiveReplanEvent("ADOPT", candidate, self.reason)

    def _retry_blocked_coordinator_route(
        self,
        current_pose,
        snapshot,
        initial_failure,
    ):
        retry_limit = self.coordinator_local_retry_count
        radius_m = self.coordinator_start_clear_radius_m
        if retry_limit <= 0 or radius_m <= 0.0 or snapshot.geometry is None:
            return initial_failure, 0, "retries disabled", frozenset()

        centre = snapshot.geometry.world_to_pixel(current_pose.x, current_pose.y)
        radius_px = max(
            1,
            int(math.ceil(radius_m * snapshot.geometry.pixels_per_metre)),
        )
        cleared_cells = frozenset(
            cell
            for cell in snapshot.occupied_cells
            if self._cell_distance_squared(cell, centre) <= radius_px * radius_px
        )
        retry_snapshot = LiveMapSnapshot(
            geometry=snapshot.geometry,
            width=snapshot.width,
            height=snapshot.height,
            occupied_cells=snapshot.occupied_cells.difference(cleared_cells),
            revision=snapshot.revision,
        )
        policies = self._coordinator_retry_policies(
            centre,
            radius_px,
            snapshot.width,
            snapshot.height,
        )[:retry_limit]
        last_failure = initial_failure
        attempts = 0
        last_policy = "start-clear fallback"
        try:
            for attempts, (policy, forbidden_cells) in enumerate(policies, start=1):
                last_policy = policy
                candidate = self._plan_active_target(
                    current_pose,
                    retry_snapshot,
                    forbidden_cells=forbidden_cells,
                )
                if candidate.success:
                    return candidate, attempts, policy, cleared_cells
                last_failure = candidate
                print(
                    f"[{self.robot_id}] LIVE coordinator local retry "
                    f"{attempts}/{retry_limit} failed ({policy}): "
                    f"{candidate.error_reason}"
                )
        finally:
            restore_snapshot = getattr(self.planner, "use_live_snapshot", None)
            if callable(restore_snapshot):
                restore_snapshot(snapshot)

        return last_failure, attempts, last_policy, cleared_cells

    def _coordinator_retry_policies(
        self,
        start_cell,
        endpoint_radius_px,
        width,
        height,
    ):
        route_cells = {
            (int(cell[0]), int(cell[1]))
            for cell in (self.active_path.pixel_path or ())
        }
        if not route_cells:
            return (("allow previous route fallback", frozenset()),)

        goal_cell = tuple(self.active_path.pixel_path[-1])
        endpoint_radius2 = endpoint_radius_px * endpoint_radius_px

        def outside_endpoints(cell):
            return (
                self._cell_distance_squared(cell, start_cell) > endpoint_radius2
                and self._cell_distance_squared(cell, goal_cell) > endpoint_radius2
            )

        exact_route = frozenset(cell for cell in route_cells if outside_endpoints(cell))
        if not exact_route:
            return (("allow previous route fallback", frozenset()),)

        buffered_route = set()
        for px, py in route_cells:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    cell = (px + dx, py + dy)
                    if (
                        0 <= cell[0] < width
                        and 0 <= cell[1] < height
                        and outside_endpoints(cell)
                    ):
                        buffered_route.add(cell)

        return (
            ("avoid previous route plus one-cell buffer", frozenset(buffered_route)),
            ("avoid exact previous route", exact_route),
            ("allow previous route fallback", frozenset()),
        )

    def _remaining_path_blocking_cells(
        self,
        planned_path,
        current_pose,
        snapshot,
    ):
        ignored_cells = self._active_start_clear_ignored_cells(current_pose)
        if not ignored_cells:
            return self.planner.remaining_path_blocking_cells(
                planned_path,
                current_pose,
                snapshot,
            )
        return self.planner.remaining_path_blocking_cells(
            planned_path,
            current_pose,
            snapshot,
            ignored_occupied_cells=ignored_cells,
        )

    def _set_start_clearance_exemption(self, current_pose, cleared_cells):
        if not cleared_cells:
            self._reset_start_clearance_exemption()
            return
        self._start_clear_origin = (current_pose.x, current_pose.y)
        self._start_clear_radius_m = self.coordinator_start_clear_radius_m
        self._start_clear_ignored_cells = frozenset(cleared_cells)

    def _active_start_clear_ignored_cells(self, current_pose):
        if self._start_clear_origin is None:
            return frozenset()
        distance = math.hypot(
            current_pose.x - self._start_clear_origin[0],
            current_pose.y - self._start_clear_origin[1],
        )
        if distance > self._start_clear_radius_m:
            self._reset_start_clearance_exemption()
            return frozenset()
        return self._start_clear_ignored_cells

    def _reset_start_clearance_exemption(self):
        self._start_clear_origin = None
        self._start_clear_radius_m = 0.0
        self._start_clear_ignored_cells = frozenset()

    @staticmethod
    def _cell_distance_squared(a, b):
        return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2

    def _plan_active_target(self, current_pose, snapshot, forbidden_cells=()):
        extra_args = {}
        if forbidden_cells:
            extra_args["forbidden_cells"] = forbidden_cells
        if self.active_path.final_target_yaw is not None:
            return self.planner.plan_to_fixed_pose(
                self.robot_id,
                current_pose,
                Pose2D(
                    self.active_path.victim_world[0],
                    self.active_path.victim_world[1],
                    self.active_path.final_target_yaw,
                ),
                self.active_path.victim_index,
                snapshot,
                strict_target=self.fixed_target_strict,
                **extra_args,
            )
        return self.planner.plan_to_fixed_victim(
            self.robot_id,
            current_pose,
            self.active_path.victim_index,
            self.active_path.victim_world,
            snapshot,
            **extra_args,
        )

    def fail(self, reason: str, current_pose: Pose2D):
        self.state = self.FAILED
        self.candidate_path = None
        self.failed_candidate_path = None
        self.blocking_cells = ()
        self.reason = reason
        self._reset_start_clearance_exemption()
        self.render(current_pose)

    def approve(self, current_pose: Pose2D):
        if self.state != self.PENDING_REPLAN or self.candidate_path is None:
            return LiveReplanEvent(
                "NONE",
                reason="no replacement route is waiting for approval",
            )

        self.active_path = self.candidate_path
        self._reset_start_clearance_exemption()
        self.candidate_path = None
        self.failed_candidate_path = None
        self.blocking_cells = ()
        self.state = self.FOLLOWING
        self.reason = "replacement route approved"
        self.render(current_pose)
        return LiveReplanEvent("ADOPT", self.active_path, self.reason)

    def finish(self):
        self.state = self.IDLE
        self.active_path = None
        self.candidate_path = None
        self.failed_candidate_path = None
        self.fixed_target_strict = False
        self.blocking_cells = ()
        self.reason = "route complete"
        self._reset_start_clearance_exemption()

    def cancel(self):
        self.state = self.IDLE
        self.active_path = None
        self.candidate_path = None
        self.failed_candidate_path = None
        self.fixed_target_strict = False
        self.blocking_cells = ()
        self.reason = "cancelled"
        self._reset_start_clearance_exemption()

    def render(
        self,
        current_pose: Pose2D,
        refresh_base_map: bool = True,
    ):
        from robot.planning.render import render_live_replanner

        render_live_replanner(self, current_pose, refresh_base_map)

    def _victim_status(self):
        if self.victim_status_provider is None:
            return None
        try:
            return self.victim_status_provider()
        except Exception as exc:
            return {"state": "ERROR", "reason": str(exc), "tracks": ()}

    def _start_viewer(self):
        from robot.planning.render import start_viewer

        self._viewer_process = start_viewer(
            self.output_path,
            f"{self.robot_id} live replanning",
            f"[{self.robot_id}] Could not start live replan viewer",
        )

    def _stop_viewer(self):
        from robot.planning.render import stop_viewer

        stop_viewer(self._viewer_process)
