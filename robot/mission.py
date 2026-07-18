"""Robot1 local mission runtime for the flat architecture."""

from __future__ import annotations

import atexit
import csv
import json
import math
import os
from pathlib import Path
import time
import traceback

from config import (
    GROUND_VICTIM_MODEL_PATH,
    REPO_ROOT,
    VICTIM_DIAGNOSTICS_DIR,
    _env_flag,
    _env_float,
)
from parameters import DepthCamera as DepthCameraParams
from parameters import Mission as MissionParams
from parameters import Safety as SafetyParams
from parameters import Victim as VictimParams
from coordinator.coordinator import (
    CoordinatorClient,
    CoordinatorDirective,
    CoordinatorLeader,
    CoordinatorStatus,
    decode_coordinator_message,
    encode_coordinator_message,
)
from robot.mapping.drone_map import DroneExtractionMap
from robot.mapping.live_grid import LiveOccupancyGridMapper
from robot.motion import (
    SlewLimiter,
    WaypointNavigator,
    cap_wheel_pair,
    scan_spin_command,
)
from robot.planning.live_replanner import LiveReplanner
from robot.planning.path_planner import PathPlanner
from robot.robot_io import RobotIO
from robot.robot_state import RobotState
from robot.safety import SafetyCollisionLayer, SafetyRecoveryController
from robot.sensing import (
    AnnotatedDepthViewer,
    CompassEncoderOdometry,
    CompetitionLocaliser,
    LocalObstacleDetector,
    TerrainAwarenessLayer,
)
from shared_types import LiveReplanEvent, PlannedPath, Pose2D, SafetyDecision
from robot.victim_id import (
    MissionAction,
    VictimDebugViewer,
    VictimDetector,
    VictimReporter,
    VictimSearchController,
    VictimTracker,
    target_record_unavailable_for_robot,
)


class ControllerTickProfiler:
    """Tiny opt-in profiler for measuring Python controller work per tick."""

    def __init__(self, robot_id: str):
        self.enabled = _env_flag("SAR_PROFILE", MissionParams.PROFILE_ENABLED)
        self.robot_id = robot_id
        self.tick = 0
        self.seconds = max(
            0.0,
            _env_float("SAR_PROFILE_SECONDS", MissionParams.PROFILE_SECONDS),
        )
        self.path = None
        self.trace_path = None
        self.file = None
        self.trace_file = None
        self.writer = None

        if not self.enabled:
            return

        default_path = (
            REPO_ROOT
            / "benchmark_results"
            / "controller_profile"
            / f"{robot_id}_profile.csv"
        )
        self.path = Path(os.environ.get("SAR_PROFILE_PATH", default_path))
        self.trace_path = self.path.with_suffix(".trace.txt")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open("w", newline="", encoding="utf-8", buffering=1)
        self.trace_file = self.trace_path.open("w", encoding="utf-8", buffering=1)
        self.writer = csv.DictWriter(
            self.file,
            fieldnames=("tick", "sim_time", "total_ms"),
        )
        self.writer.writeheader()
        atexit.register(self.close)
        print(
            f"[{self.robot_id}] SAR_PROFILE enabled: writing {self.path} "
            f"for {self.seconds:.1f}s"
        )
        self.trace("profiler_initialised")

    def trace(self, message: str):
        if not self.enabled or self.trace_file is None:
            return
        self.trace_file.write(f"{time.perf_counter():.6f} {message}\n")
        self.trace_file.flush()

    def start_tick(self):
        if not self.enabled:
            return 0.0
        return time.perf_counter()

    def record(self, sim_time: float, started_at: float):
        if not self.enabled or self.writer is None:
            return
        total_ms = (time.perf_counter() - started_at) * 1000.0
        self.tick += 1
        self.writer.writerow(
            {
                "tick": self.tick,
                "sim_time": f"{sim_time:.6f}",
                "total_ms": f"{total_ms:.6f}",
            }
        )
        if self.file is not None:
            self.file.flush()

    def should_stop(self, sim_time: float):
        return self.enabled and self.seconds > 0.0 and sim_time >= self.seconds

    def close(self):
        if self.trace_file is not None:
            self.trace_file.close()
            self.trace_file = None
        if self.file is None:
            return
        self.file.close()
        self.file = None


class Robot1FlatRuntime:
    """
    Flat runtime for the extracted controller architecture.

    This class wires sensing, mapping, planning, motion, safety, coordinator
    directives, and victim handling through the new module boundaries.

    Main loop shape:
    RobotIO snapshot -> sensing -> mapping -> victim/state/coordinator updates
    -> mission decision -> planning when needed -> motion -> safety
    -> RobotIO motor output.
    """

    WHEEL_RADIUS_M = MissionParams.WHEEL_RADIUS_M
    PRINT_STATUS_ENABLED = MissionParams.PRINT_STATUS_ENABLED
    PRINT_INTERVAL_S = MissionParams.PRINT_INTERVAL_S
    ROBOT1_STRAIGHT_TEST_DISTANCE_M = MissionParams.ROBOT1_STRAIGHT_TEST_DISTANCE_M
    DRIVE_SPEED_RAD_S = MissionParams.DRIVE_SPEED_RAD_S
    TURN_SPEED_RAD_S = MissionParams.TURN_SPEED_RAD_S
    MAX_WHEEL_SPEED_RAD_S = MissionParams.MAX_WHEEL_SPEED_RAD_S
    MAX_WHEEL_ACCEL_RAD_S2 = MissionParams.MAX_WHEEL_ACCEL_RAD_S2
    COORDINATOR_STATUS_INTERVAL_S = MissionParams.COORDINATOR_STATUS_INTERVAL_S
    DEPTH_CAMERA_HZ = DepthCameraParams.HZ

    KEYS = {
        "forward": ord("W"),
        "back": ord("S"),
        "left": ord("A"),
        "right": ord("D"),
        "goto": ord("C"),
        "straight_test": ord("B"),
        "approve_replan": ord("N"),
        "cancel": ord("X"),
        "safety_reset": ord("R"),
        "victim_target_mode": ord("T"),
    }
    ROBOT2_KEYS = {
        "forward": ord("I"),
        "back": ord("K"),
        "left": ord("J"),
        "right": ord("L"),
        "goto": ord("U"),
        "straight_test": ord("B"),
        "approve_replan": ord("N"),
        "cancel": ord("P"),
        "safety_reset": ord("O"),
        "victim_target_mode": ord("T"),
    }

    def __init__(
        self,
        robot=None,
        coordinator_enabled: bool = True,
        manual_only: bool = False,
        victim_test: bool = False,
    ):
        if robot is None:
            from controller import Robot

            robot = Robot()
        self.robot = robot
        self.io = RobotIO(robot)
        self.robot_id = self.io.robot_id
        self.timestep = self.io.timestep
        self.keyboard = self.io.enable_keyboard()
        self.previous_pressed = set()
        self.benchmark_mode = _env_flag(
            "SAR_BENCHMARK",
            MissionParams.BENCHMARK_ENABLED,
        )
        self.manual_only = bool(manual_only)
        self.victim_test = bool(victim_test)
        self.keys = self.ROBOT2_KEYS if self.robot_id == "robot2" else self.KEYS
        coordinator_active = coordinator_enabled and not self.manual_only

        self.drive_speed_rad_s = _env_float("DRIVE_SPEED_RAD_S", self.DRIVE_SPEED_RAD_S)
        self.turn_speed_rad_s = _env_float("TURN_SPEED_RAD_S", self.TURN_SPEED_RAD_S)
        self.slew = SlewLimiter(
            _env_float("MAX_WHEEL_SPEED_RAD_S", self.MAX_WHEEL_SPEED_RAD_S),
            _env_float("MAX_WHEEL_ACCEL_RAD_S2", self.MAX_WHEEL_ACCEL_RAD_S2),
        )

        self.odometry = CompassEncoderOdometry(self.WHEEL_RADIUS_M)
        self.localiser = CompetitionLocaliser(self.robot_id)
        self.terrain_awareness = TerrainAwarenessLayer(enabled=not self.benchmark_mode)
        self.terrain = None
        self.world_pose = self.localiser.localise(self.odometry.pose)
        self.state = RobotState(self.robot_id)

        self.obstacle_detector = LocalObstacleDetector(
            depth_width=self.io.depth_width,
            depth_height=self.io.depth_height,
            depth_min_range=self.io.depth_min_range,
            depth_max_range=self.io.depth_max_range,
            depth_hfov=self.io.depth_fov,
        )
        self.depth_debug_viewer = AnnotatedDepthViewer(
            self.robot_id,
            self.obstacle_detector,
            enabled_by_default=(
                not self.benchmark_mode
                and self.robot_id == "robot1"
                and AnnotatedDepthViewer.ENABLED_FOR_ROBOT1_BY_DEFAULT
            ),
        )
        self.safety = SafetyCollisionLayer(
            depth_width=self.obstacle_detector.depth_width,
            depth_height=self.obstacle_detector.depth_height,
            depth_min_range=self.obstacle_detector.depth_min_range,
            depth_max_range=self.obstacle_detector.depth_max_range,
            depth_hfov=self.obstacle_detector.depth_hfov,
            enabled=_env_flag(
                "SAFETY_ENABLED",
                SafetyParams.ENABLED and not self.benchmark_mode,
            ),
            obstacle_detector=self.obstacle_detector,
        )
        self.safety_recovery = SafetyRecoveryController(
            self.robot_id,
            rear_stop_m=self.safety.ir_stop_m,
        )

        self.drone_map = DroneExtractionMap(
            self.robot_id,
            enabled_by_default=not self.benchmark_mode,
            render_overlay=False,
        )
        self.path_planner = PathPlanner(
            self.robot_id,
            self.drone_map,
            enabled_by_default=not self.benchmark_mode,
            auto_plan=False,
            render_overlay=False,
        )
        self.live_map = LiveOccupancyGridMapper(
            self.robot_id,
            enabled_by_default=not self.benchmark_mode,
            drone_map=self.drone_map,
        )
        self.live_replanner = LiveReplanner(
            self.robot_id,
            self.path_planner,
            self.live_map,
            self.drone_map.victim_estimates,
            enabled_by_default=not self.benchmark_mode,
            recovery_controller=self.safety_recovery,
        )
        self.route_start_debug_render = _env_flag(
            "ROUTE_START_DEBUG_RENDER",
            False,
        )

        self.waypoint_navigator = WaypointNavigator()

        self.supervisor_emitter = self._optional_supervisor_emitter()
        victim_enabled = not self.benchmark_mode and _env_flag(
            "VICTIM_ID_ENABLED",
            VictimParams.DETECTOR_ENABLED,
        )
        model_path = Path(
            os.environ.get(
                "VICTIM_MODEL_PATH",
                str(GROUND_VICTIM_MODEL_PATH),
            )
        )
        self.victim_tracker = VictimTracker(
            self.drone_map.victim_estimates if self.drone_map.ready else ()
        )
        self.victim_detector = VictimDetector(
            model_path=model_path,
            image_width=self.io.rgb_width,
            image_height=self.io.rgb_height,
            depth_min_m=self.obstacle_detector.depth_min_range,
            depth_max_m=self.obstacle_detector.depth_max_range,
            horizontal_fov=self.obstacle_detector.depth_hfov,
            camera_forward_offset_m=self.obstacle_detector.camera_forward_offset_m,
            camera_height_m=self.obstacle_detector.camera_height_m,
            enabled=victim_enabled,
        )
        self.victim_mission = VictimSearchController(
            self.robot_id,
            self.victim_tracker,
            enabled=(
                victim_enabled
                and not self.victim_test
                and _env_flag("VICTIM_AUTONOMY_ENABLED", VictimParams.AUTONOMY_ENABLED)
            ),
        )
        self.live_replanner.victim_status_provider = self.victim_mission.snapshot
        self.victim_reporter = VictimReporter(self.robot_id, self.supervisor_emitter)
        self.victim_viewer = VictimDebugViewer(
            self.robot_id,
            VICTIM_DIAGNOSTICS_DIR,
            enabled=victim_enabled,
        )

        self.last_control_time = 0.0
        self.print_status_enabled = _env_flag(
            "MISSION_PRINT_STATUS",
            self.PRINT_STATUS_ENABLED,
        )
        self.last_print_time = -float("inf")
        self.command_left = 0.0
        self.command_right = 0.0
        self.target_left = 0.0
        self.target_right = 0.0
        self.route_paused_for_safety = False
        self.route_paused_for_coordinator = False
        self.robot_operational = True
        self.collision_hold = False
        self.encounter_hold = False
        self.escape_route_active = False
        self.escape_route_started_time = None
        self.victim_route_active = False
        self.victim_route_failed_pending = False
        self.victim_scan_spinning = False
        self.victim_debug_print = _env_flag(
            "VICTIM_DEBUG_PRINT",
            VictimParams.DEBUG_PRINT_ENABLED,
        )
        self.last_obstacle_observation = None
        self.last_victim_result = None
        self._victim_detector_status = None
        self.depth_camera_hz = _env_float("DEPTH_CAMERA_HZ", self.DEPTH_CAMERA_HZ)
        self.depth_interval_s = self._interval_from_hz(self.depth_camera_hz)
        self.last_depth_read_time = -float("inf")
        if self.robot_id == "robot1":
            self.coordinator = CoordinatorLeader(
                enabled=coordinator_active,
            )
        else:
            self.coordinator = CoordinatorClient(
                self.robot_id,
                enabled=coordinator_active,
            )
        self.coordinator_is_leader = isinstance(self.coordinator, CoordinatorLeader)
        self.active_directive_id = ""
        self.active_directive_kind = ""
        self.active_target_prior_id = ""
        self.directive_state = "IDLE"
        self.directive_reason = ""
        self.completed_directive_ids = set()
        self.pending_report_action = None
        self.last_coordinator_status_time = -float("inf")
        self.profiler = ControllerTickProfiler(self.robot_id)

        mode = "victim_test" if self.victim_test else ("manual" if self.manual_only else "normal")
        print(f"[{self.robot_id}] SAR_CONTROLLER_MODE={mode} flat runtime active")
        if self.benchmark_mode:
            self._start_benchmark_waypoints()

    def _victim_debug(self, message: str):
        if not self.victim_debug_print:
            return
        print(
            f"[{self.robot_id}] VICTIM DEBUG mission "
            f"directive={self.directive_state}:{self.directive_reason or '-'} "
            f"route={self.waypoint_navigator.state} "
            f"victim={self.victim_mission.state}:{self.victim_mission.reason or '-'} "
            f"- {message}"
        )

    @staticmethod
    def _interval_from_hz(hz: float) -> float:
        return 0.0 if hz <= 0.0 else 1.0 / hz

    def _start_benchmark_waypoints(self):
        waypoints = self._benchmark_waypoints_from_env()
        self.waypoint_navigator.start(waypoints)
        print(f"[{self.robot_id}] benchmark route started with {len(waypoints)} waypoint(s)")

    def _benchmark_waypoints_from_env(self):
        waypoints_text = os.environ.get("TARGET_WAYPOINTS")
        if waypoints_text:
            try:
                return self._parse_waypoints_json(waypoints_text)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                print(f"[CONFIG] Ignoring invalid TARGET_WAYPOINTS: {exc}")

        return (
            Pose2D(
                x=_env_float("TARGET_X", MissionParams.BENCHMARK_TARGET_X),
                y=_env_float("TARGET_Y", MissionParams.BENCHMARK_TARGET_Y),
                yaw=_env_float("TARGET_YAW", MissionParams.BENCHMARK_TARGET_YAW),
            ),
        )

    @staticmethod
    def _parse_waypoints_json(text: str):
        raw_waypoints = json.loads(text)
        if not isinstance(raw_waypoints, list):
            raise ValueError("expected a JSON list")

        waypoints = []
        for i, item in enumerate(raw_waypoints, start=1):
            if isinstance(item, dict):
                waypoints.append(
                    Pose2D(
                        x=float(item["x"]),
                        y=float(item["y"]),
                        yaw=float(item["yaw"]),
                    )
                )
            elif isinstance(item, list) and len(item) == 3:
                waypoints.append(
                    Pose2D(x=float(item[0]), y=float(item[1]), yaw=float(item[2]))
                )
            else:
                raise ValueError(f"waypoint {i} must be {{x,y,yaw}} or [x,y,yaw]")

        if not waypoints:
            raise ValueError("expected at least one waypoint")
        return tuple(waypoints)

    def run(self):
        self.profiler.trace("enter_run")
        while True:
            self.profiler.trace("before_step")
            step_result = self.robot.step(self.timestep)
            self.profiler.trace(f"after_step result={step_result}")
            if step_result == -1:
                break
            profile_started = self.profiler.start_tick()
            try:
                sim_time = self.robot.getTime()
                dt = sim_time - self.last_control_time
                if self.last_control_time == 0.0:
                    dt = self.timestep / 1000.0
                self.last_control_time = sim_time

                needs_ir = self._needs_ir()
                snapshot = self.io.read_snapshot(
                    sim_time,
                    include_depth=False,
                    include_lidar=False,
                    include_ir=needs_ir,
                )
                self._update_pose(snapshot, dt)

                victim_submit_due = self._prepare_victim_perception(sim_time)
                needs_lidar = self._needs_lidar(sim_time)
                lidar_ranges = ()
                lidar_fov = snapshot.lidar_fov
                if needs_lidar:
                    lidar_ranges, lidar_fov = self.io.read_lidar_ranges()

                needs_depth = self._needs_depth(sim_time, victim_submit_due)
                read_depth = (
                    needs_depth
                    and self._should_read_depth(
                        sim_time,
                        force=victim_submit_due,
                    )
                )
                depth_image = None
                observation = None
                if read_depth:
                    depth_image = self._read_depth_image(sim_time)
                if read_depth or needs_ir:
                    observation = self.obstacle_detector.observe(
                        depth_image,
                        snapshot.ir_ranges,
                        self.terrain,
                    )
                self.last_obstacle_observation = observation

                rendered = False
                if self._needs_live_map_update():
                    rendered = self.live_map.update(
                        self.world_pose,
                        sim_time,
                        lidar_ranges=lidar_ranges,
                        lidar_fov=lidar_fov,
                        observation=observation,
                        terrain=self.terrain,
                    )
                if self._needs_live_replan_visual(rendered):
                    self.live_replanner.refresh_visual(self.world_pose, rendered)
                if self._needs_runtime_state_update():
                    self._update_state(sim_time)
                if self.coordinator.enabled:
                    self._update_coordinator(sim_time)
                self._update_victim_perception(
                    sim_time,
                    depth_image,
                    submit_due=victim_submit_due,
                )

                pressed = set()
                if self._needs_keyboard():
                    pressed = self.io.read_pressed_keys(self.keyboard)
                    self._handle_mode_keys(pressed, sim_time)

                requested_left, requested_right = self._requested_command(pressed, sim_time)
                if self.victim_mission.state == self.victim_mission.CLOSE_IN:
                    requested_left, requested_right = cap_wheel_pair(
                        requested_left,
                        requested_right,
                        self.victim_mission.close_in_speed_cap_rad_s,
                    )
                requested_left, requested_right = self.terrain_awareness.limit_wheel_command(
                    requested_left,
                    requested_right,
                )
                if self.safety.enabled:
                    safety_decision = self.safety.evaluate(
                        observation,
                        snapshot.ir_ranges,
                        requested_left,
                        requested_right,
                    )
                else:
                    safety_decision = SafetyDecision(
                        left_rad_s=requested_left,
                        right_rad_s=requested_right,
                        reason="safety disabled",
                    )
                if safety_decision.new_stop:
                    self.live_map.confirm_safety_obstacle(
                        observation,
                        self.world_pose,
                        sim_time,
                        safety_decision.source,
                    )
                    close_in_reported = False
                    if self.victim_mission.state == self.victim_mission.CLOSE_IN:
                        action = self.victim_mission.accept_close_in_safety_stop(
                            self.world_pose,
                            sim_time,
                        )
                        if action.kind == "REPORT":
                            self._execute_victim_action(action, sim_time)
                            close_in_reported = True
                    if (
                        not close_in_reported
                        and (self.waypoint_navigator.active or self.victim_route_active)
                    ):
                        self._begin_safety_recovery(safety_decision, sim_time)
                    requested_left = requested_right = 0.0
                    self.command_left = self.command_right = 0.0
                    self.slew.set_immediate(0.0, 0.0)
                else:
                    self._maybe_reset_recovery_attempt_after_clear(sim_time)
                    requested_left = safety_decision.left_rad_s
                    requested_right = safety_decision.right_rad_s

                if self.safety.stop_latched:
                    final_left = final_right = 0.0
                    self.slew.set_immediate(0.0, 0.0)
                else:
                    final_left, final_right = self.slew.update(
                        requested_left,
                        requested_right,
                        dt,
                    )

                self.target_left = requested_left
                self.target_right = requested_right
                self.command_left = final_left
                self.command_right = final_right
                if self._needs_depth_viewer_update():
                    self.depth_debug_viewer.update(
                        sim_time,
                        depth_image,
                        observation,
                        self.world_pose,
                        final_left,
                        final_right,
                        len(self.live_map.local_obstacles.depth_cells),
                        snapshot.accelerometer,
                        snapshot.gyro,
                    )
                self.io.write_motors(final_left, final_right)
                self.previous_pressed = pressed

                if (
                    self.print_status_enabled
                    and sim_time - self.last_print_time >= self.PRINT_INTERVAL_S
                ):
                    self._print_status(sim_time)
                    self.last_print_time = sim_time

                self.profiler.record(sim_time, profile_started)
                if self.profiler.should_stop(sim_time):
                    print(f"[{self.robot_id}] SAR_PROFILE complete at t={sim_time:.2f}s")
                    break
            except Exception:
                self.profiler.trace("exception_in_tick")
                self.profiler.trace(traceback.format_exc())
                raise

    def _victim_perception_active(self, sim_time: float = 0.0) -> bool:
        del sim_time
        return self.victim_test or self.victim_mission.active

    def _prepare_victim_perception(self, sim_time: float) -> bool:
        active = self._victim_perception_active(sim_time)
        self.victim_detector.set_active(
            active,
            scan_rate=self.victim_test or self.victim_mission.scan_rate_requested,
        )
        return self.victim_detector.should_submit(sim_time)

    def _needs_depth(self, sim_time: float, victim_submit_due: bool = False) -> bool:
        del sim_time
        return (
            self.safety.enabled
            or self.live_map.local_obstacles.enabled
            or self.depth_debug_viewer.enabled
            or victim_submit_due
        )

    def _should_read_depth(self, sim_time: float, force: bool = False) -> bool:
        if force:
            return True
        return (
            self.depth_interval_s <= 0.0
            or sim_time - self.last_depth_read_time >= self.depth_interval_s
        )

    def _read_depth_image(self, sim_time: float):
        self.last_depth_read_time = sim_time
        return self.io.read_depth_image()

    def _needs_lidar(self, sim_time: float) -> bool:
        return self.live_map.should_read_lidar(sim_time, self.terrain)

    def _needs_ir(self) -> bool:
        return (
            self.safety.enabled
            or self.live_map.local_obstacles.enabled
            or self.safety_recovery.active
        )

    def _needs_runtime_state_update(self) -> bool:
        return not self.benchmark_mode or self.coordinator.enabled

    def _needs_keyboard(self) -> bool:
        return not self.benchmark_mode

    def _needs_live_map_update(self) -> bool:
        return self.live_map.enabled

    def _needs_live_replan_visual(self, rendered: bool) -> bool:
        return bool(rendered and self.live_replanner.enabled)

    def _needs_depth_viewer_update(self) -> bool:
        return self.depth_debug_viewer.enabled

    def _update_pose(self, snapshot, dt: float):
        wheel_angles = snapshot.wheel_angles or {}
        left_angle = (wheel_angles.get("fl", 0.0) + wheel_angles.get("rl", 0.0)) / 2.0
        right_angle = (wheel_angles.get("fr", 0.0) + wheel_angles.get("rr", 0.0)) / 2.0
        compass = snapshot.compass or (0.0, 1.0, 0.0)
        compass_yaw = math.atan2(compass[0], compass[1])
        odom_pose = self.odometry.update(left_angle, right_angle, compass_yaw)
        self.world_pose = self.localiser.localise(odom_pose)
        self.terrain = self.terrain_awareness.update(
            snapshot.accelerometer,
            snapshot.gyro,
            dt,
        )

    def _update_state(self, sim_time: float):
        snapshot = self.live_map.planning_snapshot()
        self.state.update_pose(sim_time, self.odometry.pose, self.world_pose, self.terrain)
        self.state.update_map_summary(
            ready=snapshot is not None,
            revision=snapshot.revision if snapshot is not None else 0,
            occupied_count=len(snapshot.occupied_cells) if snapshot is not None else 0,
            victim_count=self.drone_map.victim_count,
            error="" if self.drone_map.ready else self.drone_map.error_message,
        )
        self.state.update_victim_summary_from_mission(self.victim_mission)
        self.state.update_route_summary(
            active=self.waypoint_navigator.active,
            state=self.waypoint_navigator.state,
            waypoint_index=self.waypoint_navigator.current_waypoint_number,
            waypoint_count=self.waypoint_navigator.total_waypoints,
        )
        self.state.update_safety_summary(self.safety, self.safety_recovery)

    def _handle_mode_keys(self, pressed, sim_time: float):
        just_pressed = pressed - self.previous_pressed
        if self.keys["safety_reset"] in just_pressed:
            self.safety.reset()
            if self.safety_recovery.stopped:
                self.safety_recovery.reset_route()
            if self.route_paused_for_safety and self.waypoint_navigator.paused:
                self.waypoint_navigator.resume()
            self.route_paused_for_safety = False
            print(f"[{self.robot_id}] SAFETY stop reset")

        if self.manual_only:
            if self.keys["cancel"] in just_pressed:
                self._cancel_routes("cancelled")
            return

        if self.keys["victim_target_mode"] in just_pressed:
            mode = self.victim_mission.toggle_prior_selection_mode()
            print(f"[{self.robot_id}] Victim prior selection: {mode.upper()}")

        if self.keys["goto"] in just_pressed:
            if self.safety.stop_latched:
                print(f"[{self.robot_id}] victim mission cannot start: safety latched")
            else:
                self._start_victim_mission(sim_time)

        if self.keys["straight_test"] in just_pressed:
            if self.safety.stop_latched:
                print(f"[{self.robot_id}] 8 m test cannot start: safety latched")
            else:
                self._start_straight_test(sim_time)

        if self.keys["approve_replan"] in just_pressed:
            self._approve_pending_replan()

        if self.keys["cancel"] in just_pressed:
            self._cancel_routes("cancelled")

    def _update_coordinator(self, sim_time: float):
        if not self.coordinator.enabled:
            return
        self._process_coordinator_messages()
        status = self._build_coordinator_status(sim_time)
        if self.coordinator_is_leader:
            self.coordinator.update_status(status)
            self.coordinator.tick(sim_time)
            self._broadcast_coordinator_directives()
            self._sync_coordinator_targets()
            directive = self.coordinator.next_directive(self.robot_id)
        else:
            self._publish_coordinator_status(status, sim_time)
            self._sync_coordinator_targets()
            directive = self.coordinator.next_directive()
        if directive is None:
            return
        if directive.directive_id == self.active_directive_id:
            return
        if directive.directive_id in self.completed_directive_ids:
            return
        self._accept_coordinator_directive(directive, sim_time)

    def _build_coordinator_status(self, sim_time: float):
        report_ready = self.pending_report_action is not None
        report_confidence = (
            self.pending_report_action.confidence
            if self.pending_report_action is not None
            else self.victim_mission.report_confidence
        )
        return CoordinatorStatus(
            robot_id=self.robot_id,
            sim_time_s=sim_time,
            world_pose=self.world_pose,
            route_state=self.waypoint_navigator.state,
            directive_id=self.active_directive_id,
            directive_state=self.directive_state,
            reason=self.directive_reason,
            victim_state=self.victim_mission.state,
            selected_track_id=self.victim_mission.selected_track_id,
            assigned_prior_id=self.active_target_prior_id,
            report_ready=report_ready,
            report_confidence=report_confidence,
            robot_operational=self.robot_operational,
            mission_active=self._coordinator_mission_active(),
            encountered_prior_id=self.victim_mission.encountered_track_id,
        )

    def _coordinator_mission_active(self):
        if not self.robot_operational or self.directive_state != "ACTIVE":
            return False
        return bool(
            self.waypoint_navigator.active
            or self.waypoint_navigator.paused
            or self.victim_scan_spinning
            or self.safety_recovery.active
            or self.victim_mission.active
        )

    def _process_coordinator_messages(self):
        for raw_message in self.io.poll_team_messages():
            message = decode_coordinator_message(raw_message)
            if message is None or message.get("sender") == self.robot_id:
                continue
            if self.coordinator_is_leader:
                if message.get("kind") == "STATUS":
                    self.coordinator.update_status(
                        CoordinatorStatus.from_dict(message.get("payload"))
                    )
            else:
                self.coordinator.receive_message(message)

    def _sync_coordinator_targets(self):
        records = getattr(self.coordinator, "target_records", None)
        if records is None and self.coordinator_is_leader:
            records = getattr(self.coordinator, "targets", None)
        if not records:
            return
        self.victim_tracker.apply_coordinator_targets(records, self.robot_id)
        if (
            self.victim_mission.active
            and self.victim_mission.selected_track_id
            and self._coordinator_target_unavailable(
                self.victim_mission.selected_track_id,
                records,
                allow_assigned=True,
            )
        ):
            self._complete_superseded_target("selected victim already handled by coordinator")
            return
        if (
            self.pending_report_action is not None
            and self._coordinator_target_unavailable(
                self.pending_report_action.track_id,
                records,
                allow_assigned=True,
            )
        ):
            self._clear_pending_report("victim already handled by coordinator")
        if (
            self.active_directive_kind == "TARGET_VICTIM"
            and self.active_target_prior_id
            and self._coordinator_target_unavailable(
                self.active_target_prior_id,
                records,
            )
        ):
            self._complete_superseded_target("target already handled by teammate")

    def _coordinator_target_unavailable(
        self,
        prior_id: str,
        records,
        allow_assigned: bool = False,
    ):
        record = records.get(prior_id) if isinstance(records, dict) else None
        return target_record_unavailable_for_robot(
            record,
            self.robot_id,
            allow_assigned=allow_assigned,
        )

    def _clear_pending_report(self, reason: str):
        self.pending_report_action = None
        self.victim_mission.report_status = ""
        self.victim_mission.reason = reason

    def _complete_superseded_target(self, reason: str):
        self.victim_mission.cancel()
        self.victim_route_active = False
        self.victim_scan_spinning = False
        self._clear_pending_report(reason)
        self.waypoint_navigator.cancel()
        self.live_replanner.cancel()
        self.safety_recovery.reset_route()
        self.route_paused_for_safety = False
        self.route_paused_for_coordinator = False
        self.collision_hold = False
        self.encounter_hold = False
        self._mark_directive_done(reason)

    def _publish_coordinator_status(self, status: CoordinatorStatus, sim_time: float):
        if sim_time - self.last_coordinator_status_time < self.COORDINATOR_STATUS_INTERVAL_S:
            return
        self.last_coordinator_status_time = sim_time
        self.io.send_team_message(
            encode_coordinator_message(self.robot_id, "STATUS", status)
        )

    def _broadcast_coordinator_directives(self):
        target_board = self.coordinator.target_board_message()
        if target_board:
            self.io.send_team_message(
                encode_coordinator_message(self.robot_id, "TARGETS", target_board)
            )
        for directive in self.coordinator.directives_for_broadcast(self.robot_id):
            self.io.send_team_message(
                encode_coordinator_message(self.robot_id, "DIRECTIVE", directive)
            )
        for control in self.coordinator.controls_for_broadcast(self.robot_id):
            self.io.send_team_message(
                encode_coordinator_message(self.robot_id, "CONTROL", control)
            )

    def _accept_coordinator_directive(self, directive, sim_time: float):
        self.encounter_hold = False
        self.collision_hold = False
        self.state.latest_directive = directive
        self.active_directive_id = directive.directive_id
        self.active_directive_kind = directive.kind
        self.active_target_prior_id = directive.target_prior_id
        self.directive_state = "ACTIVE"
        self.directive_reason = directive.reason
        print(
            f"[{self.robot_id}] COORDINATOR directive accepted: "
            f"{directive.directive_id} {directive.kind} - {directive.reason}"
        )

        if directive.kind == "TARGET_VICTIM":
            self._start_coordinator_victim_route(directive, sim_time)
            return
        if directive.kind == "VICTIM_MISSION":
            self._start_victim_mission(sim_time)
            return
        if directive.kind == "REPORT_VICTIM":
            self._send_victim_report_from_directive(directive, sim_time)
            return
        if directive.kind == "REPORT_DENIED":
            self._clear_pending_report(directive.reason or "report denied by coordinator")
            self._mark_directive_done(directive.reason or "report denied by coordinator")
            return
        if directive.kind == "FIXED_POSE" and directive.target_pose is not None:
            self._start_directive_pose_route(directive.target_pose, sim_time)
            return
        if directive.kind == "HOLD":
            self._cancel_routes("held by coordinator")
            return

        self._mark_directive_failed(f"unsupported directive kind {directive.kind!r}")

    def _start_coordinator_victim_route(
        self,
        directive: CoordinatorDirective,
        sim_time: float,
    ):
        if not directive.target_prior_id:
            self._mark_directive_failed("TARGET_VICTIM directive has no prior id")
            return
        if not directive.waypoints:
            self._mark_directive_failed(
                "TARGET_VICTIM directive has no coordinator waypoints"
            )
            return
        if not self.victim_detector.enabled and self.victim_detector.error_message:
            self.victim_mission.state = self.victim_mission.FAILED
            self.victim_mission.reason = self.victim_detector.error_message
            self._mark_directive_failed(self.victim_detector.error_message)
            print(
                f"[{self.robot_id}] VICTIM mission unavailable: "
                f"{self.victim_detector.error_message}"
            )
            return

        self.safety_recovery.reset_route()
        self.safety.reset()
        self.waypoint_navigator.cancel()
        self.live_replanner.cancel()
        self.victim_scan_spinning = False
        self.encounter_hold = False
        self._clear_pending_report("new coordinator target accepted")

        action = self.victim_mission.start(
            self.world_pose,
            sim_time,
            evaluate_route=lambda _target: (True, 0.0),
            target_track_id=directive.target_prior_id,
        )
        if action.kind != "ROUTE":
            self._mark_directive_failed(
                self.victim_mission.reason or "assigned victim prior could not start"
            )
            return

        planned_path = self._planned_path_from_directive(directive)
        event = self.live_replanner.follow_preplanned_path(
            planned_path,
            self.world_pose,
            sim_time,
            reason=f"coordinator route to {directive.target_prior_id}",
        )
        self._handle_live_replan_event(event)
        if event.action not in ("START", "ADOPT") or event.path is None:
            return

        self.victim_route_active = True
        self.victim_route_failed_pending = False
        print(
            f"[{self.robot_id}] COORDINATOR victim route started: "
            f"{directive.target_prior_id} mode={event.path.planning_mode} "
            f"waypoints={len(event.path.waypoints)}"
        )

    def _planned_path_from_directive(self, directive: CoordinatorDirective):
        target_world = None
        if directive.target_pose is not None:
            target_world = (directive.target_pose.x, directive.target_pose.y)
        return PlannedPath(
            robot_id=self.robot_id,
            victim_index=directive.target_index,
            victim_world=target_world,
            pixel_path=tuple(directive.pixel_path or ()),
            waypoints=tuple(directive.waypoints or ()),
            waypoint_speed_caps=tuple(directive.waypoint_speed_caps or ()),
            planning_mode=directive.planning_mode or "COORDINATOR",
            physical_path_cost_m=directive.physical_path_cost_m,
            weighted_path_cost_m=directive.weighted_path_cost_m,
            success=True,
        )

    def _start_directive_pose_route(self, target_pose: Pose2D, sim_time: float):
        self.safety_recovery.reset_route()
        self.waypoint_navigator.cancel()
        event = self.live_replanner.start_fixed_pose(
            self.world_pose,
            sim_time,
            target_pose,
            target_index=0,
            strict_target=False,
        )
        if event.action == "START" and event.path is not None:
            self._start_planned_path(event.path)
            print(
                f"[{self.robot_id}] COORDINATOR route started: "
                f"{self.active_directive_id}"
            )
            return
        self._mark_directive_failed(event.reason)

    def _requested_command(self, pressed, sim_time: float):
        if self.safety_recovery.active:
            if self.escape_route_active and self.waypoint_navigator.active:
                return self.waypoint_navigator.update(self.world_pose, sim_time)
            if self.safety_recovery.state == self.safety_recovery.REPLANNING:
                self._handle_recovery_replan(sim_time)
            return 0.0, 0.0

        if self._coordinator_hold_active():
            if self.waypoint_navigator.active:
                self.waypoint_navigator.pause()
                self.route_paused_for_coordinator = True
            self.slew.set_immediate(0.0, 0.0)
            return 0.0, 0.0

        if self.route_paused_for_coordinator:
            if self.waypoint_navigator.paused:
                self.waypoint_navigator.resume()
            self.route_paused_for_coordinator = False

        self._update_victim_mission(sim_time)

        if self.victim_scan_spinning:
            return scan_spin_command()

        if self.waypoint_navigator.active:
            event = self.live_replanner.update(self.world_pose, sim_time)
            self._handle_live_replan_event(event)
            return self.waypoint_navigator.update(self.world_pose, sim_time)

        if self.waypoint_navigator.state == self.waypoint_navigator.DONE:
            if not self.victim_mission.active and not self.victim_route_active:
                self._mark_directive_done("route complete")
            self.live_replanner.finish()
            self.victim_route_active = False

        return self._manual_command_from_keys(pressed)

    def _manual_command_from_keys(self, pressed):
        left = 0.0
        right = 0.0
        if self.keys["forward"] in pressed:
            left += self.drive_speed_rad_s
            right += self.drive_speed_rad_s
        if self.keys["back"] in pressed:
            left -= self.drive_speed_rad_s
            right -= self.drive_speed_rad_s
        if self.keys["left"] in pressed:
            left -= self.turn_speed_rad_s
            right += self.turn_speed_rad_s
        if self.keys["right"] in pressed:
            left += self.turn_speed_rad_s
            right -= self.turn_speed_rad_s
        return (
            max(-self.slew.max_speed_rad_s, min(self.slew.max_speed_rad_s, left)),
            max(-self.slew.max_speed_rad_s, min(self.slew.max_speed_rad_s, right)),
        )

    def _coordinator_hold_active(self):
        if not self.coordinator.enabled:
            self.collision_hold = False
            return self.encounter_hold
        control = (
            self.coordinator.control_for(self.robot_id)
            if self.coordinator_is_leader
            else self.coordinator.control
        )
        self.collision_hold = bool(control.get("collision_hold", False))
        denied_prior = str(control.get("denied_encounter_prior_id", ""))
        if denied_prior and denied_prior == self.victim_mission.encountered_track_id:
            self.victim_mission.deny_encounter(denied_prior)
            self.encounter_hold = False
        return self.collision_hold or self.encounter_hold

    def _start_planned_path(self, planned_path: PlannedPath):
        self.waypoint_navigator.start(
            planned_path.waypoints,
            planned_path.waypoint_speed_caps or None,
        )
        if not self.route_start_debug_render:
            return
        try:
            from robot.mapping.render import render_route_start_debug

            render_route_start_debug(
                self.live_map,
                self.path_planner,
                planned_path,
                self.world_pose,
            )
        except Exception as exc:
            self.route_start_debug_render = False
            print(f"[{self.robot_id}] Route-start debug rendering disabled: {exc}")

    def _start_straight_test(self, sim_time: float):
        start = self.world_pose
        target = (
            start.x + self.ROBOT1_STRAIGHT_TEST_DISTANCE_M * math.cos(start.yaw),
            start.y + self.ROBOT1_STRAIGHT_TEST_DISTANCE_M * math.sin(start.yaw),
        )
        self.safety_recovery.reset_route()
        event = self.live_replanner.start_fixed_target(
            self.world_pose,
            sim_time,
            target,
            target_index=0,
        )
        if event.action == "START" and event.path is not None:
            self._start_planned_path(event.path)
            print(f"[{self.robot_id}] 8 m LIVE test started")
        else:
            print(f"[{self.robot_id}] 8 m LIVE test failed: {event.reason}")

    def _start_victim_mission(self, sim_time: float, target_prior_id: str = ""):
        if not self.victim_detector.enabled and self.victim_detector.error_message:
            self.victim_mission.state = self.victim_mission.FAILED
            self.victim_mission.reason = self.victim_detector.error_message
            self._mark_directive_failed(self.victim_detector.error_message)
            print(f"[{self.robot_id}] VICTIM mission unavailable: {self.victim_detector.error_message}")
            return
        self.safety_recovery.reset_route()
        self.safety.reset()
        self.waypoint_navigator.cancel()
        self.live_replanner.cancel()
        action = self.victim_mission.start(
            self.world_pose,
            sim_time,
            self._evaluate_victim_pose_route,
            target_track_id=target_prior_id,
        )
        self._execute_victim_action(action, sim_time)
        print(f"[{self.robot_id}] VICTIM mission: {self.victim_mission.state} - {self.victim_mission.reason}")

    def _evaluate_victim_pose_route(self, target):
        snapshot = self.live_map.planning_snapshot()
        if snapshot is None:
            return False, float("inf")
        target_pose = target
        if not isinstance(target_pose, Pose2D):
            target_pose = Pose2D(
                x=float(target[0]),
                y=float(target[1]),
                yaw=float(target[2]) if len(target) > 2 else self.world_pose.yaw,
            )
        return self.path_planner.estimate_route_cost(
            self.world_pose,
            target_pose,
            snapshot,
        )

    def _execute_victim_action(self, action: MissionAction, sim_time: float):
        if action.kind in ("NONE", ""):
            return
        if action.kind == "HOLD":
            return
        if action.kind == "ENCOUNTER":
            self.encounter_hold = True
            if self.waypoint_navigator.active:
                self.waypoint_navigator.pause()
                self.route_paused_for_coordinator = True
            self.slew.set_immediate(0.0, 0.0)
            self._victim_debug(
                f"waiting for coordinator decision on encountered victim "
                f"{action.track_id or '-'}"
            )
            return
        if action.kind == "FAIL":
            self.live_replanner.cancel()
            self.waypoint_navigator.cancel()
            self.victim_route_active = False
            self.victim_route_failed_pending = False
            self.victim_scan_spinning = False
            self.encounter_hold = False
            self._mark_directive_failed(action.reason or "victim mission failed")
            return
        if action.kind == "SPIN":
            self.live_replanner.cancel()
            self.waypoint_navigator.cancel()
            self.victim_route_active = False
            self.victim_scan_spinning = True
            return
        self.victim_scan_spinning = False
        if action.kind == "CLOSE_IN":
            target = Pose2D(*action.target_pose)
            self.live_replanner.cancel()
            self.waypoint_navigator.cancel()
            self.waypoint_navigator.start((target,))
            self.victim_route_active = True
            self.victim_route_failed_pending = False
            self._victim_debug(
                f"starting direct close-in track={action.track_id or '-'} "
                f"target=({target.x:.2f},{target.y:.2f},"
                f"{math.degrees(target.yaw):.1f}deg) "
                f"reason={action.reason or '-'}"
            )
            return
        if action.kind == "ROUTE":
            candidate_targets = tuple(
                Pose2D(*target_pose)
                for target_pose in action.candidate_target_poses
            )
            if candidate_targets:
                self._victim_debug(
                    f"starting victim side-view selection track={action.track_id or '-'} "
                    f"candidates={len(candidate_targets)} reason={action.reason or '-'}"
                )
                event = self.live_replanner.start_best_fixed_pose(
                    self.world_pose,
                    sim_time,
                    candidate_targets,
                    target_index=0,
                )
            else:
                target = Pose2D(*action.target_pose)
                self._victim_debug(
                    f"starting victim route kind=ROUTE track={action.track_id or '-'} "
                    f"target=({target.x:.2f},{target.y:.2f},"
                    f"{math.degrees(target.yaw):.1f}deg) "
                    f"strict={action.strict_target} reason={action.reason or '-'}"
                )
                event = self.live_replanner.start_fixed_pose(
                    self.world_pose,
                    sim_time,
                    target,
                    target_index=0,
                    strict_target=action.strict_target,
                )
            if event.action == "START" and event.path is not None:
                if candidate_targets:
                    final_yaw = event.path.final_target_yaw
                    if final_yaw is None and event.path.waypoints:
                        final_yaw = event.path.waypoints[-1].yaw
                    chosen_target = (
                        float(event.path.victim_world[0]),
                        float(event.path.victim_world[1]),
                        float(final_yaw or 0.0),
                    )
                    self.victim_mission.accept_approach_target(
                        chosen_target,
                        action.reason,
                    )
                self._start_planned_path(event.path)
                self.victim_route_active = True
                self.victim_route_failed_pending = False
                self._victim_debug(
                    f"victim route started mode={event.path.planning_mode} "
                    f"waypoints={len(event.path.waypoints)} "
                    f"reason={event.reason or '-'}"
                )
            else:
                self.victim_route_active = False
                self.victim_route_failed_pending = True
                self.waypoint_navigator.cancel()
                self.victim_mission.reason = event.reason
                self._victim_debug(
                    f"victim route failed to start action={event.action or '-'} "
                    f"reason={event.reason or '-'}"
                )
            return
        if action.kind == "REPORT":
            self.live_replanner.cancel()
            self.waypoint_navigator.cancel()
            self.victim_route_active = False
            self.encounter_hold = False
            if self.coordinator.enabled and self.active_directive_kind != "REPORT_VICTIM":
                self.pending_report_action = action
                self.victim_mission.report_status = "READY"
                self.victim_mission.report_confidence = action.confidence
                self.victim_mission.reason = "victim report waiting for coordinator approval"
                return
            self._send_victim_report(action.track_id, action.confidence, sim_time)

    def _send_victim_report_from_directive(
        self,
        directive: CoordinatorDirective,
        sim_time: float,
    ):
        action = self.pending_report_action
        track_id = directive.target_prior_id
        confidence = directive.confidence
        if action is not None and action.track_id == track_id:
            confidence = action.confidence
        self.pending_report_action = None
        self._send_victim_report(track_id, confidence, sim_time)

    def _send_victim_report(self, track_id: str, confidence: float, sim_time: float):
        sent, dry_run, _payload = self.victim_reporter.report(
            track_id,
            sim_time,
            self.world_pose,
            confidence,
        )
        if sent or dry_run:
            self.victim_mission.mark_reported(
                confidence,
                dry_run,
                self.world_pose,
            )
            self._mark_directive_done("victim reported")
        else:
            self.victim_mission.state = self.victim_mission.FAILED
            self.victim_mission.reason = self.victim_reporter.last_status
            self._mark_directive_failed(self.victim_reporter.last_status)

    def _update_victim_perception(
        self,
        sim_time: float,
        depth_image,
        submit_due: bool = None,
    ):
        if submit_due is None:
            submit_due = self._prepare_victim_perception(sim_time)
        if submit_due:
            if depth_image is None:
                depth_image = self._read_depth_image(sim_time)
            self.victim_detector.submit(
                sim_time,
                self.world_pose,
                self.io.read_rgb_image(),
                depth_image,
            )
        result = self.victim_detector.poll()
        if result is None:
            return
        self.last_victim_result = result
        self.victim_tracker.update(result)
        self.victim_viewer.render(
            result,
            self.victim_mission.snapshot(),
            current_sim_time=sim_time,
        )
        if self.victim_mission.active:
            self.victim_mission.on_result(result, self.world_pose, sim_time)

    def _update_victim_mission(self, sim_time: float):
        if not self.victim_mission.active:
            return
        route_done = (
            self.victim_route_active
            and self.waypoint_navigator.state == self.waypoint_navigator.DONE
        )
        route_failed = self.victim_route_failed_pending or (
            self.victim_route_active
            and self.live_replanner.state == self.live_replanner.FAILED
        )
        if route_failed:
            self._victim_debug(
                "passing route_failed into victim mission "
                f"pending={self.victim_route_failed_pending} "
                f"live_replanner={self.live_replanner.state}:{self.live_replanner.reason or '-'}"
            )
        self.victim_route_failed_pending = False
        action = self.victim_mission.update(
            self.world_pose,
            sim_time,
            route_done=route_done,
            route_failed=route_failed,
            robot_stopped=abs(self.command_left) < 0.2 and abs(self.command_right) < 0.2,
        )
        self._execute_victim_action(action, sim_time)

    def _handle_live_replan_event(self, event: LiveReplanEvent):
        if event.action in ("NONE", ""):
            return
        if event.action in ("START", "ADOPT") and event.path is not None:
            self._start_planned_path(event.path)
            self.route_paused_for_safety = False
            return
        if event.action == "PENDING":
            self.waypoint_navigator.pause()
            return
        if event.action == "FAILED":
            self.waypoint_navigator.cancel()
            self.victim_route_active = False
            if self.victim_mission.active:
                self.victim_route_failed_pending = True
                self._victim_debug(
                    f"live replanner failed active victim route: {event.reason}"
                )
            else:
                self._mark_directive_failed(event.reason)
            print(f"[{self.robot_id}] LIVE replan failed: {event.reason}")

    def _approve_pending_replan(self):
        event = self.live_replanner.approve(self.world_pose)
        if event.action == "ADOPT" and event.path is not None:
            self._start_planned_path(event.path)
            self.route_paused_for_safety = False
            print(f"[{self.robot_id}] LIVE replacement approved: {event.reason}")

    def _begin_safety_recovery(self, safety_decision, sim_time: float):
        self.waypoint_navigator.cancel()
        self.route_paused_for_safety = True
        recovery = self.safety_recovery.begin_no_reverse(
            self.world_pose,
            safety_decision.source,
            safety_decision.trigger_sensor,
        )
        if recovery.action == "STOPPED":
            self._stop_after_recovery_failure(recovery.reason)
            return
        self.safety.reset()
        print(f"[{self.robot_id}] SAFETY RECOVERY: {recovery.reason}")
        self._handle_recovery_replan(sim_time)

    def _maybe_reset_recovery_attempt_after_clear(self, sim_time: float):
        if not self.escape_route_active:
            return
        if self.escape_route_started_time is None:
            return
        if self.safety.stop_latched:
            return
        if self.safety_recovery.attempt_count <= 0:
            return
        if sim_time - self.escape_route_started_time < self.safety_recovery.reset_clear_time_s:
            return
        self.safety_recovery.mark_escape_clear()

    def _handle_recovery_replan(self, sim_time: float):
        self.io.write_motors(0.0, 0.0)
        snapshot = self.live_map.planning_snapshot()
        if not self.escape_route_active:
            if snapshot is None:
                self._stop_after_recovery_failure("no live-map snapshot for yaw-aware escape")
                return
            if not self.path_planner.enabled:
                self._stop_after_recovery_failure("path planner disabled for yaw-aware escape")
                return
            escape = self.path_planner.plan_local_escape(self.world_pose, snapshot)
            if not escape.success:
                self._stop_after_recovery_failure(escape.error_reason or "yaw-aware escape failed")
                return
            if escape.success and escape.waypoints:
                self.escape_route_active = True
                self.escape_route_started_time = sim_time
                self._start_planned_path(escape)
                print(f"[{self.robot_id}] ESCAPE ROUTE started: {escape.error_reason or 'ok'}")
                return

        if self.escape_route_active:
            if self.waypoint_navigator.state != self.waypoint_navigator.DONE:
                return
            self.waypoint_navigator.cancel()
            self.escape_route_active = False
            self.escape_route_started_time = None

        event = self.live_replanner.replan_after_recovery(
            self.world_pose,
            sim_time,
            self.safety_recovery.reason,
        )
        if event.action == "ADOPT" and event.path is not None:
            self._start_planned_path(event.path)
            self.safety_recovery.replan_succeeded()
            self.safety.reset()
            self.route_paused_for_safety = False
            print(f"[{self.robot_id}] SAFETY RECOVERY route adopted: {event.reason}")
            return

        stopped = self.safety_recovery.replan_failed(
            self.world_pose,
            event.reason,
        )
        self._stop_after_recovery_failure(stopped.reason)

    def _stop_after_recovery_failure(self, reason: str):
        self.safety_recovery.stop(reason)
        self.safety.reset()
        self.waypoint_navigator.cancel()
        self.live_replanner.fail(reason, self.world_pose)
        self.victim_route_active = False
        self.victim_scan_spinning = False
        self.escape_route_active = False
        self.escape_route_started_time = None
        self.robot_operational = False
        if self.victim_mission.active:
            selected = self.victim_mission.selected_track
            if selected is not None and selected.status != "FOUND":
                selected.status = "SEARCH_EXHAUSTED"
            self.victim_mission.state = self.victim_mission.FAILED
            self.victim_mission.reason = f"safety recovery failed: {reason}"
            self._victim_debug(f"safety recovery forced FAILED: {reason}")
        self.route_paused_for_safety = False
        self.collision_hold = False
        self.encounter_hold = False
        self.slew.set_immediate(0.0, 0.0)
        self.io.write_motors(0.0, 0.0)
        self._mark_directive_failed(reason)
        print(f"[{self.robot_id}] SAFETY RECOVERY STOPPED: {reason}")

    def _cancel_routes(self, reason: str):
        self.victim_mission.cancel()
        self.victim_route_active = False
        self.victim_scan_spinning = False
        self.pending_report_action = None
        self.waypoint_navigator.cancel()
        self.live_replanner.cancel()
        self.safety_recovery.reset_route()
        self.route_paused_for_safety = False
        self.route_paused_for_coordinator = False
        self.collision_hold = False
        self.encounter_hold = False
        if self.active_directive_id:
            self._mark_directive_failed(reason)
        print(f"[{self.robot_id}] AUTO route {reason}")

    def _mark_directive_done(self, reason: str):
        if not self.active_directive_id or self.directive_state != "ACTIVE":
            return
        self.directive_state = "DONE"
        self.directive_reason = reason
        self.completed_directive_ids.add(self.active_directive_id)
        print(
            f"[{self.robot_id}] COORDINATOR directive done: "
            f"{self.active_directive_id} - {reason}"
        )

    def _mark_directive_failed(self, reason: str):
        if not self.active_directive_id or self.directive_state in ("DONE", "FAILED"):
            return
        self.directive_state = "FAILED"
        self.directive_reason = reason
        self.completed_directive_ids.add(self.active_directive_id)
        print(
            f"[{self.robot_id}] COORDINATOR directive failed: "
            f"{self.active_directive_id} - {reason}"
        )

    def _optional_supervisor_emitter(self):
        try:
            return self.robot.getDevice("supervisor emitter")
        except Exception as exc:
            print(f"[{self.robot_id}] Warning: supervisor emitter unavailable: {exc}")
            return None

    def _print_status(self, sim_time: float):
        print(
            f"[{self.robot_id}] v2 t={sim_time:7.2f}s "
            f"world=({self.world_pose.x:+.3f},{self.world_pose.y:+.3f},"
            f"{math.degrees(self.world_pose.yaw):+.1f}deg) "
            f"route={self.waypoint_navigator.state} "
            f"wp={self.waypoint_navigator.current_waypoint_number}/"
            f"{self.waypoint_navigator.total_waypoints} "
            f"safety={self.safety.state}:{self.safety.reason or '-'} "
            f"recovery={self.safety_recovery.state}:{self.safety_recovery.reason or '-'} "
            f"victim={self.victim_mission.state}:{self.victim_mission.reason or '-'} "
            f"directive={self.directive_state}:{self.directive_reason or '-'}"
        )
