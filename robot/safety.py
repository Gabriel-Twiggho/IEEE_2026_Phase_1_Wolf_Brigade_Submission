"""Local collision safety and recovery for the flat robot runtime."""

from __future__ import annotations

from config import _env_float, _env_int, _env_flag
from parameters import Safety as SafetyParams
from parameters import SafetyRecovery as SafetyRecoveryParams
from robot.sensing import LocalObstacleDetector
from shared_types import (
    LocalObstacleObservation,
    Pose2D,
    RecoveryDecision,
    SafetyDecision,
)


class SafetyCollisionLayer:
    """
    Last-line collision guard for commands produced by teleop or navigation.

    Sensor handoff:
    - LocalObstacleDetector owns the camera geometry, collision-height band,
      safety depth band, and robot-width corridor.
    - Ground-like tilted IR readings are removed before this layer evaluates
      forward, reverse, or turning collision risk.
    - Depth latches a stop when that shared detector reports enough pixels
      inside the robot's swept volume.
    - IR limits the wheels to 5 rad/s at 0.30 m.
    - IR latches a full stop at the configured threshold (currently 0.10 m).

    Depth collision corridor:
    - Height limits come from LocalObstacleDetector.
    - Width comes from the detector's robot width plus its side margin.
    - Require at least 300 pixels across 8 image columns for 3 consecutive
      frames so isolated depth pixels do not trigger an emergency stop.

    Calibration basis:
    - Robot dimensions define the corridor; the captured test scenes validate
      it rather than deriving all dimensions from image pixels.
    - Open space, a 0.50 x 0.50 m passage, and 0.28 m height clearance remained
      clear with this corridor.
    - A 0.19 m opening, solid wall, and 0.131 m-high low box were detected.
    - The 0.22 m-high tunnel was treated as unsafe because it provides no
      clearance above a robot whose stated height is also 0.22 m.

    This class does not choose a route or avoidance manoeuvre. It only modifies
    or stops the requested wheel command.

    A latched stop remains active through invalid depth readings and requires an
    explicit reset. The caller owns cancelling any active autonomous route.
    """

    CLEAR = "CLEAR"
    IR_SLOW = "IR_SLOW"
    STOP_LATCHED = "STOP_LATCHED"

    DEPTH_MIN_PIXELS = SafetyParams.DEPTH_MIN_PIXELS
    DEPTH_MIN_COLUMNS = SafetyParams.DEPTH_MIN_COLUMNS
    DEPTH_CONFIRM_FRAMES = SafetyParams.DEPTH_CONFIRM_FRAMES
    IR_SLOW_M = SafetyParams.IR_SLOW_M
    IR_STOP_M = SafetyParams.IR_STOP_M
    IR_SLOW_SPEED_RAD_S = SafetyParams.IR_SLOW_SPEED_RAD_S

    def __init__(
        self,
        depth_width: int,
        depth_height: int,
        depth_min_range: float,
        depth_max_range: float,
        depth_hfov: float,
        enabled: bool = True,
        obstacle_detector: LocalObstacleDetector = None,
    ):
        self.enabled = enabled
        self.obstacle_detector = obstacle_detector or LocalObstacleDetector(
            depth_width=depth_width,
            depth_height=depth_height,
            depth_min_range=depth_min_range,
            depth_max_range=depth_max_range,
            depth_hfov=depth_hfov,
        )
        self.depth_width = self.obstacle_detector.depth_width
        self.depth_height = self.obstacle_detector.depth_height
        self.depth_min_range = self.obstacle_detector.depth_min_range
        self.depth_max_range = self.obstacle_detector.depth_max_range
        self.depth_hfov = self.obstacle_detector.depth_hfov
        self.depth_min_pixels = _env_int(
            "SAFETY_DEPTH_MIN_PIXELS",
            self.DEPTH_MIN_PIXELS,
        )
        self.depth_min_columns = _env_int(
            "SAFETY_DEPTH_MIN_COLUMNS",
            self.DEPTH_MIN_COLUMNS,
        )
        self.depth_confirm_frames = _env_int(
            "SAFETY_DEPTH_CONFIRM_FRAMES",
            self.DEPTH_CONFIRM_FRAMES,
        )
        self.ir_slow_m = _env_float("SAFETY_IR_SLOW_M", self.IR_SLOW_M)
        self.ir_stop_m = _env_float("SAFETY_IR_STOP_M", self.IR_STOP_M)
        self.ir_slow_speed_rad_s = _env_float(
            "SAFETY_IR_SLOW_SPEED_RAD_S",
            self.IR_SLOW_SPEED_RAD_S,
        )

        self.state = self.CLEAR
        self.source = ""
        self.trigger_sensor = ""
        self.reason = ""
        self.depth_confirm_count = 0
        self.last_depth_pixels = 0
        self.last_depth_columns = 0

    @property
    def stop_latched(self) -> bool:
        return self.state == self.STOP_LATCHED

    def reset(self):
        self.state = self.CLEAR
        self.source = ""
        self.trigger_sensor = ""
        self.reason = ""
        self.depth_confirm_count = 0
        self.last_depth_pixels = 0
        self.last_depth_columns = 0

    def evaluate(
        self,
        depth,
        ir_ranges,
        requested_left: float,
        requested_right: float,
    ) -> SafetyDecision:
        if not self.enabled:
            return SafetyDecision(
                left_rad_s=requested_left,
                right_rad_s=requested_right,
                reason="safety disabled",
            )

        if self.stop_latched:
            return self._decision(0.0, 0.0)

        if isinstance(depth, LocalObstacleObservation):
            observation = depth
        else:
            observation = self.obstacle_detector.observe(depth, ir_ranges or {})
        ir_ranges = (
            observation.collision_ir_ranges
            or observation.ir_ranges
            or {}
        )

        moving_forward = requested_left > 0.0 and requested_right > 0.0
        moving_backward = requested_left < 0.0 and requested_right < 0.0
        turning = requested_left * requested_right < 0.0

        ir_source, ir_distance, ir_sensor = self._relevant_ir(
            ir_ranges,
            moving_forward,
            moving_backward,
            turning,
        )
        if ir_distance is not None and ir_distance <= self.ir_stop_m:
            return self._latch_stop(
                ir_source,
                (
                    f"{ir_source}({ir_sensor})={ir_distance:.3f} m "
                    f"<= {self.ir_stop_m:.3f} m"
                ),
                trigger_sensor=ir_sensor,
            )

        if moving_forward:
            if observation.depth_sampled:
                depth_pixels = observation.safety_depth_pixels
                depth_columns = observation.safety_depth_columns
                self.last_depth_pixels = depth_pixels
                self.last_depth_columns = depth_columns
                depth_blocked = (
                    depth_pixels >= self.depth_min_pixels
                    and depth_columns >= self.depth_min_columns
                )
                if depth_blocked:
                    self.depth_confirm_count += 1
                else:
                    self.depth_confirm_count = 0

                if self.depth_confirm_count >= self.depth_confirm_frames:
                    return self._latch_stop(
                        "DEPTH",
                        (
                            f"{depth_pixels} pixels across {depth_columns} columns "
                            f"for {self.depth_confirm_count} frames"
                        ),
                        trigger_sensor="camera depth",
                    )
        else:
            self.depth_confirm_count = 0
            self.last_depth_pixels = 0
            self.last_depth_columns = 0

        if ir_distance is not None and ir_distance <= self.ir_slow_m:
            self.state = self.IR_SLOW
            self.source = ir_source
            self.trigger_sensor = ir_sensor
            self.reason = (
                f"{ir_source}({ir_sensor})={ir_distance:.3f} m "
                f"<= {self.ir_slow_m:.3f} m"
            )
            return self._decision(
                self.cap_speed(requested_left),
                self.cap_speed(requested_right),
            )

        self.state = self.CLEAR
        self.source = ""
        self.trigger_sensor = ""
        self.reason = "no collision risk"
        return self._decision(requested_left, requested_right)

    def _relevant_ir(
        self,
        ir_ranges,
        moving_forward: bool,
        moving_backward: bool,
        turning: bool,
    ):
        if turning:
            names = ("fl_range", "fr_range", "rl_range", "rr_range")
            label = "TURN_IR"
        elif moving_forward:
            names = ("fl_range", "fr_range")
            label = "FRONT_IR"
        elif moving_backward:
            names = ("rl_range", "rr_range")
            label = "REAR_IR"
        else:
            return "", None, ""

        sensor_name = min(names, key=lambda name: float(ir_ranges[name]))
        distance = float(ir_ranges[sensor_name])
        return label, distance, sensor_name

    def _latch_stop(
        self,
        source: str,
        reason: str,
        trigger_sensor: str = "",
    ):
        self.state = self.STOP_LATCHED
        self.source = source
        self.trigger_sensor = trigger_sensor
        self.reason = reason
        return self._decision(0.0, 0.0, new_stop=True)

    def _decision(self, left: float, right: float, new_stop: bool = False):
        return SafetyDecision(
            left_rad_s=left,
            right_rad_s=right,
            state=self.state,
            source=self.source,
            reason=self.reason,
            new_stop=new_stop,
            depth_pixels=self.last_depth_pixels,
            depth_columns=self.last_depth_columns,
            trigger_sensor=self.trigger_sensor,
        )

    def cap_speed(self, speed: float) -> float:
        return max(
            -self.ir_slow_speed_rad_s,
            min(self.ir_slow_speed_rad_s, speed),
        )


class SafetyRecoveryController:
    """
    One-shot autonomous recovery after a safety latch on an active route.

    The recovery controller only tracks the post-latch state.  The actual escape
    manoeuvre is produced by PathPlanner.plan_local_escape(), which searches a
    local 1 cm/px (x, y, yaw) grid and returns a short safe route.  If that route
    cannot be found, recovery stops instead of trying fixed backing patterns.
    """

    IDLE = "IDLE"
    REPLANNING = "REPLANNING"
    STOPPED = "STOPPED"

    def __init__(self, robot_id: str, rear_stop_m: float):
        del rear_stop_m
        self.robot_id = robot_id
        self.enabled = _env_flag(
            "SAFETY_RECOVERY_ENABLED",
            SafetyRecoveryParams.ENABLED,
        )
        self.max_attempts = 1
        self.reset_clear_time_s = max(
            0.0,
            _env_float(
                "SAFETY_RECOVERY_RESET_CLEAR_TIME_S",
                SafetyRecoveryParams.RESET_CLEAR_TIME_S,
            ),
        )

        self.state = self.IDLE
        self.attempt_count = 0
        self.reason = ""
        self.trigger_source = ""
        self.trigger_sensor = ""
        self.start_pose = Pose2D()

    @property
    def active(self):
        return self.state == self.REPLANNING

    @property
    def stopped(self):
        return self.state == self.STOPPED

    @property
    def attempts_remaining(self):
        return max(0, self.max_attempts - self.attempt_count)

    def reset_route(self):
        self.state = self.IDLE
        self.attempt_count = 0
        self.reason = ""
        self.trigger_source = ""
        self.trigger_sensor = ""

    def begin_no_reverse(
        self,
        pose: Pose2D,
        trigger_source: str,
        trigger_sensor: str,
    ) -> "RecoveryDecision":
        """Start the single yaw-aware local escape attempt."""
        if not self.enabled:
            self.state = self.STOPPED
            self.reason = "autonomous safety recovery is disabled"
            return RecoveryDecision(action="STOPPED", reason=self.reason)

        self.trigger_source = trigger_source
        self.trigger_sensor = trigger_sensor

        if self.attempt_count >= self.max_attempts:
            self.state = self.STOPPED
            self.reason = (
                "yaw-aware escape already attempted"
            )
            return RecoveryDecision(action="STOPPED", reason=self.reason)

        self.attempt_count += 1
        self.start_pose = Pose2D(pose.x, pose.y, pose.yaw)
        self.state = self.REPLANNING
        self.reason = (
            "starting yaw-aware local escape "
            f"{self.attempt_count}/{self.max_attempts}"
        )
        return RecoveryDecision(action="REPLAN", reason=self.reason)

    def update(
        self,
        pose: Pose2D,
        observation: LocalObstacleObservation,
    ) -> RecoveryDecision:
        del pose, observation
        return RecoveryDecision(reason=self.reason)

    def replan_succeeded(self):
        self.state = self.IDLE
        self.attempt_count = 0
        self.reason = (
            "yaw-aware escape produced a replacement route"
        )

    def mark_escape_clear(self):
        if self.attempt_count == 0:
            return
        self.attempt_count = 0
        if self.state == self.REPLANNING:
            self.reason = "escape route stayed clear; recovery attempt reset"

    def replan_failed(self, pose: Pose2D, reason: str) -> RecoveryDecision:
        del pose
        self.state = self.STOPPED
        self.reason = f"yaw-aware escape failed: {reason}"
        return RecoveryDecision(action="STOPPED", reason=self.reason)

    def stop(self, reason: str):
        self.state = self.STOPPED
        self.reason = reason
