"""Motion command generation for the flat robot runtime."""

from __future__ import annotations

import math

from config import _env_float
from parameters import Motion as MotionParams
from parameters import RobotGeometry
from shared_types import Pose2D


class MotionProfile:
    """Tuning bundle for normal or cautious route following."""

    def __init__(
        self,
        lookahead_m: float,
        max_linear_m_s: float,
        max_angular_rad_s: float,
        max_wheel_rad_s: float,
        rotate_in_place_heading_rad: float,
        heading_slowdown_min: float,
    ):
        self.lookahead_m = lookahead_m
        self.max_linear_m_s = max_linear_m_s
        self.max_angular_rad_s = max_angular_rad_s
        self.max_wheel_rad_s = max_wheel_rad_s
        self.rotate_in_place_heading_rad = rotate_in_place_heading_rad
        self.heading_slowdown_min = heading_slowdown_min


class WaypointNavigator:
    """
    Runs a route made from multiple Pose2D waypoints.

    The navigator follows the whole waypoint polyline with a lookahead target.
    It converts desired body linear/angular velocity into wheel speeds using
    the calibrated skid/differential-drive model, while preserving the public
    route API used by mission, live replanning, and safety recovery.
    """

    IDLE = "IDLE"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    DONE = "DONE"

    def __init__(self, _legacy_goto_controller=None):
        self.wheel_radius_m = _env_float(
            "WHEEL_RADIUS_M",
            RobotGeometry.WHEEL_RADIUS_M,
        )
        self.effective_track_m = _env_float(
            "EFFECTIVE_TRACK_M",
            RobotGeometry.EFFECTIVE_TRACK_M,
        )
        self.position_tolerance_m = _env_float(
            "POSITION_TOLERANCE_M",
            MotionParams.POSITION_TOLERANCE_M,
        )
        self.final_yaw_tolerance_rad = _env_float(
            "FINAL_YAW_TOLERANCE_RAD",
            MotionParams.FINAL_YAW_TOLERANCE_RAD,
        )
        self.kp_heading = _env_float("KP_HEADING", MotionParams.KP_HEADING)
        self.kp_distance = _env_float("KP_DISTANCE", MotionParams.KP_DISTANCE)
        self.normal_profile = MotionProfile(
            lookahead_m=_env_float("LOOKAHEAD_M", MotionParams.LOOKAHEAD_M),
            max_linear_m_s=_env_float(
                "MAX_LINEAR_M_S",
                MotionParams.MAX_LINEAR_M_S,
            ),
            max_angular_rad_s=_env_float(
                "MAX_ANGULAR_RAD_S",
                MotionParams.MAX_ANGULAR_RAD_S,
            ),
            max_wheel_rad_s=_env_float(
                "MAX_PROFILE_WHEEL_RAD_S",
                MotionParams.MAX_PROFILE_WHEEL_RAD_S,
            ),
            rotate_in_place_heading_rad=_env_float(
                "ROTATE_IN_PLACE_HEADING_RAD",
                MotionParams.ROTATE_IN_PLACE_HEADING_RAD,
            ),
            heading_slowdown_min=_env_float(
                "HEADING_SLOWDOWN_MIN",
                MotionParams.HEADING_SLOWDOWN_MIN,
            ),
        )
        self.cautious_profile = MotionProfile(
            lookahead_m=_env_float(
                "NARROW_LOOKAHEAD_M",
                MotionParams.NARROW_LOOKAHEAD_M,
            ),
            max_linear_m_s=_env_float(
                "NARROW_MAX_LINEAR_M_S",
                MotionParams.NARROW_MAX_LINEAR_M_S,
            ),
            max_angular_rad_s=_env_float(
                "NARROW_MAX_ANGULAR_RAD_S",
                MotionParams.NARROW_MAX_ANGULAR_RAD_S,
            ),
            max_wheel_rad_s=_env_float(
                "NARROW_MAX_PROFILE_WHEEL_RAD_S",
                MotionParams.NARROW_MAX_PROFILE_WHEEL_RAD_S,
            ),
            rotate_in_place_heading_rad=_env_float(
                "NARROW_ROTATE_IN_PLACE_HEADING_RAD",
                MotionParams.NARROW_ROTATE_IN_PLACE_HEADING_RAD,
            ),
            heading_slowdown_min=_env_float(
                "NARROW_HEADING_SLOWDOWN_MIN",
                MotionParams.NARROW_HEADING_SLOWDOWN_MIN,
            ),
        )

        self.state = self.IDLE
        self.waypoints = []
        self.speed_caps = []
        self.current_index = 0
        self.current_arc_m = 0.0
        self.target_arc_m = 0.0
        self.route_points = []
        self.cumulative_lengths = []
        self.total_length_m = 0.0
        self._start_anchor = None
        self._final_yaw_error = 0.0
        self._heading_error = 0.0
        self._distance_to_goal = 0.0
        self._using_cautious_profile = False

    @property
    def active(self):
        return self.state == self.RUNNING

    @property
    def paused(self):
        return self.state == self.PAUSED

    @property
    def total_waypoints(self):
        return len(self.waypoints)

    @property
    def current_waypoint_number(self):
        if not self.waypoints:
            return 0
        return min(self.current_index + 1, len(self.waypoints))

    @property
    def current_target(self):
        if not self.waypoints:
            return None
        if self.current_index >= len(self.waypoints):
            return self.waypoints[-1]
        return self.waypoints[self.current_index]

    @property
    def using_cautious_profile(self):
        return self._using_cautious_profile

    def start(self, waypoints, speed_caps=None):
        self.waypoints = list(waypoints)
        if speed_caps is None:
            self.speed_caps = [None] * len(self.waypoints)
        else:
            self.speed_caps = list(speed_caps)
            if len(self.speed_caps) != len(self.waypoints):
                raise ValueError("waypoint speed caps must align with waypoints")
        self.current_index = 0
        self.current_arc_m = 0.0
        self.target_arc_m = 0.0
        self.route_points = []
        self.cumulative_lengths = []
        self.total_length_m = 0.0
        self._start_anchor = None
        self._using_cautious_profile = False
        self._final_yaw_error = 0.0
        self._heading_error = 0.0
        self._distance_to_goal = 0.0
        self.state = self.RUNNING if self.waypoints else self.DONE

    def cancel(self):
        self.state = self.IDLE
        self.waypoints = []
        self.speed_caps = []
        self.current_index = 0
        self.current_arc_m = 0.0
        self.target_arc_m = 0.0
        self.route_points = []
        self.cumulative_lengths = []
        self.total_length_m = 0.0
        self._start_anchor = None
        self._using_cautious_profile = False
        self._final_yaw_error = 0.0
        self._heading_error = 0.0
        self._distance_to_goal = 0.0

    def pause(self):
        if self.state == self.RUNNING:
            self.state = self.PAUSED

    def resume(self):
        if self.state != self.PAUSED:
            return
        self.state = self.RUNNING if self.waypoints else self.DONE

    def update(self, current_pose: Pose2D, sim_time: float):
        del sim_time
        if self.state != self.RUNNING:
            return 0.0, 0.0
        if not self.waypoints:
            self.state = self.DONE
            return 0.0, 0.0

        self._ensure_route_points(current_pose)
        goal = self.waypoints[-1]
        self._distance_to_goal = math.hypot(
            goal.x - current_pose.x,
            goal.y - current_pose.y,
        )
        self._final_yaw_error = _angle_diff(goal.yaw, current_pose.yaw)

        if self._distance_to_goal <= self.position_tolerance_m:
            if abs(self._final_yaw_error) <= self.final_yaw_tolerance_rad:
                self.current_index = max(0, len(self.waypoints) - 1)
                self.state = self.DONE
                return 0.0, 0.0
            profile, cap = self._active_profile_and_cap(self.total_length_m)
            left, right = self._turn_in_place(self._final_yaw_error, profile)
            return self._apply_wheel_cap(left, right, cap, profile.max_wheel_rad_s)

        closest_arc = self._project_pose_to_route_arc(current_pose)
        profile, cap = self._active_profile_and_cap(closest_arc)
        lookahead_arc = min(self.total_length_m, closest_arc + profile.lookahead_m)
        lookahead = self._point_at_arc(lookahead_arc)
        target_bearing = math.atan2(
            lookahead.y - current_pose.y,
            lookahead.x - current_pose.x,
        )
        self._heading_error = _angle_diff(target_bearing, current_pose.yaw)
        self.current_arc_m = closest_arc
        self.target_arc_m = lookahead_arc
        self.current_index = self._waypoint_index_for_arc(lookahead_arc)

        linear = min(
            profile.max_linear_m_s,
            max(0.0, self.kp_distance * self._distance_to_goal),
        )
        angular = _clamp(
            self.kp_heading * self._heading_error,
            -profile.max_angular_rad_s,
            profile.max_angular_rad_s,
        )
        if abs(self._heading_error) >= profile.rotate_in_place_heading_rad:
            linear = 0.0
        else:
            ratio = abs(self._heading_error) / max(
                profile.rotate_in_place_heading_rad,
                1e-6,
            )
            slowdown = 1.0 - ratio * (1.0 - profile.heading_slowdown_min)
            linear *= _clamp(slowdown, profile.heading_slowdown_min, 1.0)

        left, right = self._wheel_inverse(linear, angular)
        return self._apply_wheel_cap(left, right, cap, profile.max_wheel_rad_s)

    def _ensure_route_points(self, current_pose: Pose2D):
        if self.route_points:
            return
        self._start_anchor = Pose2D(current_pose.x, current_pose.y, current_pose.yaw)
        self.route_points = [self._start_anchor]
        self.route_points.extend(self.waypoints)
        self.cumulative_lengths = [0.0]
        for previous, current in zip(self.route_points, self.route_points[1:]):
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1]
                + math.hypot(current.x - previous.x, current.y - previous.y)
            )
        self.total_length_m = self.cumulative_lengths[-1]

    def _active_profile_and_cap(self, arc_m: float):
        cap = self._minimum_cap_near_arc(arc_m, self.normal_profile.lookahead_m)
        if cap is None:
            self._using_cautious_profile = False
            return self.normal_profile, None

        cautious_cap = self._minimum_cap_near_arc(
            arc_m,
            self.cautious_profile.lookahead_m,
        )
        if cautious_cap is not None:
            cap = min(cap, cautious_cap)
        self._using_cautious_profile = True
        return self.cautious_profile, cap

    def _minimum_cap_near_arc(self, arc_m: float, lookahead_m: float):
        best_cap = None
        end_arc = min(self.total_length_m, arc_m + lookahead_m)
        for waypoint_index, cap in enumerate(self.speed_caps):
            if cap is None or cap <= 0.0:
                continue
            route_index = waypoint_index + 1
            if route_index >= len(self.cumulative_lengths):
                continue
            waypoint_arc = self.cumulative_lengths[route_index]
            if arc_m - 1e-6 <= waypoint_arc <= end_arc + 1e-6:
                best_cap = cap if best_cap is None else min(best_cap, cap)
        return best_cap

    def _project_pose_to_route_arc(self, pose: Pose2D):
        if len(self.route_points) < 2:
            return 0.0
        best_arc = 0.0
        best_dist_sq = float("inf")
        for index, (start, end) in enumerate(
            zip(self.route_points, self.route_points[1:])
        ):
            dx = end.x - start.x
            dy = end.y - start.y
            seg_len_sq = dx * dx + dy * dy
            if seg_len_sq <= 1e-12:
                continue
            ratio = _clamp(
                ((pose.x - start.x) * dx + (pose.y - start.y) * dy) / seg_len_sq,
                0.0,
                1.0,
            )
            px = start.x + ratio * dx
            py = start.y + ratio * dy
            dist_sq = (pose.x - px) ** 2 + (pose.y - py) ** 2
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_arc = self.cumulative_lengths[index] + math.sqrt(seg_len_sq) * ratio
        return best_arc

    def _point_at_arc(self, arc_m: float):
        if not self.route_points:
            return Pose2D()
        if arc_m <= 0.0 or len(self.route_points) == 1:
            return self.route_points[0]
        if arc_m >= self.total_length_m:
            return self.route_points[-1]
        for index in range(len(self.route_points) - 1):
            start_arc = self.cumulative_lengths[index]
            end_arc = self.cumulative_lengths[index + 1]
            if arc_m > end_arc:
                continue
            segment_len = max(end_arc - start_arc, 1e-9)
            ratio = (arc_m - start_arc) / segment_len
            start = self.route_points[index]
            end = self.route_points[index + 1]
            x = start.x + ratio * (end.x - start.x)
            y = start.y + ratio * (end.y - start.y)
            yaw = math.atan2(end.y - start.y, end.x - start.x)
            return Pose2D(x, y, yaw)
        return self.route_points[-1]

    def _waypoint_index_for_arc(self, arc_m: float):
        if not self.waypoints:
            return 0
        for waypoint_index in range(len(self.waypoints)):
            route_index = waypoint_index + 1
            if route_index >= len(self.cumulative_lengths):
                break
            if arc_m <= self.cumulative_lengths[route_index] + 1e-6:
                return waypoint_index
        return len(self.waypoints) - 1

    def _turn_in_place(self, heading_error: float, profile: MotionProfile):
        angular = _clamp(
            self.kp_heading * heading_error,
            -profile.max_angular_rad_s,
            profile.max_angular_rad_s,
        )
        return self._wheel_inverse(0.0, angular)

    def _wheel_inverse(self, linear_m_s: float, angular_rad_s: float):
        left = (
            linear_m_s - 0.5 * angular_rad_s * self.effective_track_m
        ) / self.wheel_radius_m
        right = (
            linear_m_s + 0.5 * angular_rad_s * self.effective_track_m
        ) / self.wheel_radius_m
        return left, right

    @staticmethod
    def _apply_wheel_cap(left: float, right: float, speed_cap, profile_cap: float):
        cap = profile_cap
        if speed_cap is not None and speed_cap > 0.0:
            cap = min(cap, speed_cap)
        return cap_wheel_pair(left, right, cap)


class SlewLimiter:
    """Wheel speed clamp and acceleration limiter used before motor output."""

    def __init__(
        self,
        max_speed_rad_s: float = MotionParams.MAX_WHEEL_SPEED_RAD_S,
        max_accel_rad_s2: float = MotionParams.MAX_WHEEL_ACCEL_RAD_S2,
    ):
        self.max_speed_rad_s = _env_float("MAX_WHEEL_SPEED_RAD_S", max_speed_rad_s)
        self.max_accel_rad_s2 = _env_float("MAX_WHEEL_ACCEL_RAD_S2", max_accel_rad_s2)
        self.command_left = 0.0
        self.command_right = 0.0

    def reset(self):
        self.command_left = 0.0
        self.command_right = 0.0

    def update(self, target_left: float, target_right: float, dt: float):
        max_delta = self.max_accel_rad_s2 * max(dt, 0.0)
        left = self._move_toward(self.command_left, target_left, max_delta)
        right = self._move_toward(self.command_right, target_right, max_delta)
        self.command_left = self._clamp_speed(left)
        self.command_right = self._clamp_speed(right)
        return self.command_left, self.command_right

    def set_immediate(self, left: float, right: float):
        self.command_left = self._clamp_speed(left)
        self.command_right = self._clamp_speed(right)
        return self.command_left, self.command_right

    def _clamp_speed(self, speed_rad_s: float):
        return max(-self.max_speed_rad_s, min(self.max_speed_rad_s, speed_rad_s))

    @staticmethod
    def _move_toward(current: float, target: float, max_delta: float):
        if target > current:
            return min(target, current + max_delta)
        return max(target, current - max_delta)


def scan_spin_command(speed_rad_s: float = MotionParams.VICTIM_SCAN_SPIN_SPEED_RAD_S):
    """Wheel command used by the current victim 360-degree spin scan."""

    speed = _env_float("VICTIM_SCAN_SPIN_SPEED_RAD_S", speed_rad_s)
    return speed, -speed


def _angle_diff(angle_rad: float, reference_rad: float):
    return math.atan2(
        math.sin(angle_rad - reference_rad),
        math.cos(angle_rad - reference_rad),
    )


def _clamp(value: float, low: float, high: float):
    return max(low, min(high, value))


def cap_wheel_pair(left: float, right: float, max_abs: float):
    """Scale a wheel pair to a shared cap while preserving its turn ratio."""

    if max_abs <= 0.0:
        return 0.0, 0.0
    peak = max(abs(left), abs(right))
    if peak <= max_abs:
        return left, right
    scale = max_abs / peak
    return left * scale, right * scale
