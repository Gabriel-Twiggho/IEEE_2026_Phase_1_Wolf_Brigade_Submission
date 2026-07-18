"""Local sensing, localisation, terrain, and obstacle-projection logic."""

from __future__ import annotations

import atexit
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from config import LIVE_MAP_VIEWER_PATH, SLAM_LITE_DIR, _env_float, _env_int, _env_flag
from parameters import DepthViewer as DepthViewerParams
from parameters import Localisation as LocalisationParams
from parameters import LocalObstacleProjection as LocalObstacleProjectionParams
from parameters import Terrain as TerrainParams
from shared_types import (
    LocalObstacleObservation,
    LocalObstaclePoint,
    OdomDelta,
    Pose2D,
    TerrainAssessment,
)


class CompassEncoderOdometry:
    """Wheel-encoder distance plus compass-heading odometry."""

    def __init__(self, wheel_radius_m: float):
        self.wheel_radius_m = wheel_radius_m
        self.pose = Pose2D()
        self.total_left_m = 0.0
        self.total_right_m = 0.0
        self._previous_left_angle = None
        self._previous_right_angle = None
        self._start_compass_yaw = None
        self.last_delta = OdomDelta()

    def update(
        self,
        left_angle_rad: float,
        right_angle_rad: float,
        compass_yaw_rad: float,
    ) -> Pose2D:
        if self._previous_left_angle is None:
            self._previous_left_angle = left_angle_rad
            self._previous_right_angle = right_angle_rad
            self._start_compass_yaw = compass_yaw_rad
            self.last_delta = OdomDelta()
            return self.pose

        delta_left_angle = left_angle_rad - self._previous_left_angle
        delta_right_angle = right_angle_rad - self._previous_right_angle
        self._previous_left_angle = left_angle_rad
        self._previous_right_angle = right_angle_rad

        left_m = delta_left_angle * self.wheel_radius_m
        right_m = delta_right_angle * self.wheel_radius_m
        forward_m = 0.5 * (left_m + right_m)

        compass_local_yaw = self._angle_diff(
            compass_yaw_rad,
            self._start_compass_yaw,
        )
        yaw_delta = self._angle_diff(compass_local_yaw, self.pose.yaw)
        mid_yaw = self.pose.yaw + 0.5 * yaw_delta
        self.pose.x += forward_m * math.cos(mid_yaw)
        self.pose.y += forward_m * math.sin(mid_yaw)
        self.pose.yaw = compass_local_yaw

        self.total_left_m += left_m
        self.total_right_m += right_m
        self.last_delta = OdomDelta(left_m, right_m, forward_m, yaw_delta)
        return self.pose

    @staticmethod
    def _angle_diff(angle_rad: float, reference_rad: float) -> float:
        return math.atan2(
            math.sin(angle_rad - reference_rad),
            math.cos(angle_rad - reference_rad),
        )


class CompetitionLocaliser:
    """Convert robot-start odometry into the competition/world frame."""

    START_POSES = LocalisationParams.START_POSES

    def __init__(self, robot_id: str):
        default = self.START_POSES.get(robot_id, Pose2D())
        self.start_pose = Pose2D(
            x=_env_float("LOCALISER_START_X", default.x),
            y=_env_float("LOCALISER_START_Y", default.y),
            yaw=_env_float("LOCALISER_START_YAW", default.yaw),
        )

    def localise(self, odom_pose: Pose2D) -> Pose2D:
        c = math.cos(self.start_pose.yaw)
        s = math.sin(self.start_pose.yaw)
        return Pose2D(
            x=self.start_pose.x + c * odom_pose.x - s * odom_pose.y,
            y=self.start_pose.y + s * odom_pose.x + c * odom_pose.y,
            yaw=self._wrap_angle(self.start_pose.yaw + odom_pose.yaw),
        )

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))


class TerrainAwarenessLayer:
    """Detect sustained robot tilt before changing obstacle interpretation."""

    LEVEL = "LEVEL"
    TILT_CANDIDATE = "TILT_CANDIDATE"
    TILTED = "TILTED"
    UNRELIABLE = "UNRELIABLE"

    ENTER_TILT_RAD = TerrainParams.ENTER_TILT_RAD
    ENTER_TILT_HOLD_S = TerrainParams.ENTER_TILT_HOLD_S
    EXIT_TILT_RAD = TerrainParams.EXIT_TILT_RAD
    LEVEL_HOLD_S = TerrainParams.LEVEL_HOLD_S
    CRAWL_SPEED_RAD_S = TerrainParams.CRAWL_SPEED_RAD_S
    ACCEL_CORRECTION_TAU_S = TerrainParams.ACCEL_CORRECTION_TAU_S
    ACCEL_STARTUP_GRACE_S = TerrainParams.ACCEL_STARTUP_GRACE_S
    IMU_HOLD_S = TerrainParams.IMU_HOLD_S

    def __init__(self, enabled: bool):
        self.enabled = bool(enabled) and _env_flag(
            "TERRAIN_AWARENESS_ENABLED",
            TerrainParams.ENABLED,
        )
        self.enter_tilt_rad = _env_float(
            "TERRAIN_TILT_ENTER_RAD",
            self.ENTER_TILT_RAD,
        )
        self.enter_tilt_hold_s = max(
            0.0,
            _env_float("TERRAIN_TILT_ENTER_HOLD_S", self.ENTER_TILT_HOLD_S),
        )
        self.exit_tilt_rad = _env_float(
            "TERRAIN_TILT_EXIT_RAD",
            self.EXIT_TILT_RAD,
        )
        self.level_hold_s = max(
            0.0,
            _env_float("TERRAIN_LEVEL_HOLD_S", self.LEVEL_HOLD_S),
        )
        self.crawl_speed_rad_s = max(
            0.0,
            _env_float("TERRAIN_CRAWL_SPEED_RAD_S", self.CRAWL_SPEED_RAD_S),
        )
        self.accel_correction_tau_s = max(
            1e-3,
            _env_float(
                "TERRAIN_ACCEL_CORRECTION_TAU_S",
                self.ACCEL_CORRECTION_TAU_S,
            ),
        )
        self.accel_startup_grace_s = max(
            0.0,
            _env_float(
                "TERRAIN_ACCEL_STARTUP_GRACE_S",
                self.ACCEL_STARTUP_GRACE_S,
            ),
        )
        self.imu_hold_s = max(
            0.0,
            _env_float("TERRAIN_IMU_HOLD_S", self.IMU_HOLD_S),
        )

        self._up_vector = np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        self._state = self.UNRELIABLE
        self._tilt_candidate_s = 0.0
        self._level_candidate_s = 0.0
        self._time_without_accel_s = 0.0
        self._elapsed_s = 0.0
        self.assessment = TerrainAssessment()

    def update(self, accelerometer_values, gyro_values, dt: float):
        if not self.enabled:
            self.assessment = TerrainAssessment()
            return self.assessment

        dt = max(0.0, float(dt))
        self._elapsed_s += dt
        accel = self._vector_or_none(accelerometer_values)
        gyro = self._vector_or_none(gyro_values)
        accel_norm = float(np.linalg.norm(accel)) if accel is not None else 0.0
        accel_valid = accel is not None and accel_norm > 1e-6
        accel_startup_gated = (
            accel_valid and self._elapsed_s <= self.accel_startup_grace_s
        )

        if self._up_vector is not None and gyro is not None and dt > 0.0:
            predicted = self._up_vector - np.cross(gyro, self._up_vector) * dt
            self._up_vector = self._normalise(predicted)

        measured_up = accel / accel_norm if accel_valid else None
        accel_usable = accel_valid and not accel_startup_gated
        if accel_usable:
            correction = 1.0 - math.exp(-dt / self.accel_correction_tau_s)
            blended = (1.0 - correction) * self._up_vector + correction * measured_up
            self._up_vector = self._normalise(blended)
            self._time_without_accel_s = 0.0
        else:
            self._time_without_accel_s += dt

        reliable = (
            self._up_vector is not None
            and (
                accel_usable
                or accel_startup_gated
                or (
                    gyro is not None
                    and self._time_without_accel_s <= self.imu_hold_s
                )
            )
        )
        if not reliable:
            self._state = self.UNRELIABLE
            self._tilt_candidate_s = 0.0
            self._level_candidate_s = 0.0
            self.assessment = TerrainAssessment()
            return self.assessment

        up = self._up_vector
        pitch_rad = math.atan2(
            float(up[0]),
            math.hypot(float(up[1]), float(up[2])),
        )
        roll_rad = math.atan2(-float(up[1]), float(up[2]))
        tilt_rad = math.acos(max(-1.0, min(1.0, float(up[2]))))

        if self._elapsed_s <= self.accel_startup_grace_s:
            self._state = self.LEVEL
            self._tilt_candidate_s = 0.0
            self._level_candidate_s = 0.0
        elif self._state == self.TILTED:
            if tilt_rad <= self.exit_tilt_rad:
                self._level_candidate_s += dt
                if self._level_candidate_s >= self.level_hold_s:
                    self._state = self.LEVEL
                    self._tilt_candidate_s = 0.0
                    self._level_candidate_s = 0.0
            else:
                self._level_candidate_s = 0.0
        elif tilt_rad >= self.enter_tilt_rad:
            if self._state != self.TILT_CANDIDATE:
                self._state = self.TILT_CANDIDATE
                self._tilt_candidate_s = 0.0
            self._tilt_candidate_s += dt
            if self._tilt_candidate_s >= self.enter_tilt_hold_s:
                self._state = self.TILTED
                self._tilt_candidate_s = 0.0
                self._level_candidate_s = 0.0
        else:
            self._state = self.LEVEL
            self._tilt_candidate_s = 0.0
            self._level_candidate_s = 0.0

        self.assessment = TerrainAssessment(
            state=self._state,
            pitch_rad=pitch_rad,
            roll_rad=roll_rad,
            tilt_rad=tilt_rad,
            up_vector=tuple(float(value) for value in up),
            reliable=True,
            candidate_s=(
                self._tilt_candidate_s
                if self._state == self.TILT_CANDIDATE
                else 0.0
            ),
            accel_startup_gated=accel_startup_gated,
        )
        return self.assessment

    def limit_wheel_command(self, left_rad_s: float, right_rad_s: float):
        if (
            not self.enabled
            or self.assessment.state != self.TILTED
            or self.crawl_speed_rad_s <= 0.0
        ):
            return left_rad_s, right_rad_s

        largest = max(abs(left_rad_s), abs(right_rad_s))
        if largest <= self.crawl_speed_rad_s or largest <= 1e-9:
            return left_rad_s, right_rad_s
        scale = self.crawl_speed_rad_s / largest
        return left_rad_s * scale, right_rad_s * scale

    @staticmethod
    def level_basis(assessment: TerrainAssessment):
        if (
            assessment is None
            or not assessment.reliable
            or assessment.state != TerrainAwarenessLayer.TILTED
        ):
            return (
                np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
                np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
                np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
            )

        up = TerrainAwarenessLayer._normalise(
            np.asarray(assessment.up_vector, dtype=np.float64)
        )
        body_forward = np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
        forward = body_forward - float(np.dot(body_forward, up)) * up
        if float(np.linalg.norm(forward)) < 1e-6:
            forward = np.asarray((0.0, 1.0, 0.0), dtype=np.float64)
            forward -= float(np.dot(forward, up)) * up
        forward = TerrainAwarenessLayer._normalise(forward)
        left = TerrainAwarenessLayer._normalise(np.cross(up, forward))
        return forward, left, up

    @staticmethod
    def _vector_or_none(values):
        if values is None:
            return None
        vector = np.asarray(values, dtype=np.float64)
        if vector.size != 3 or not np.all(np.isfinite(vector)):
            return None
        return vector.reshape(3)

    @staticmethod
    def _normalise(vector):
        norm = float(np.linalg.norm(vector))
        if norm < 1e-9:
            return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
        return vector / norm

class LocalObstacleDetector:
    """Convert raw depth/IR readings into robot-relative obstacle points."""

    ROBOT_WIDTH_M = LocalObstacleProjectionParams.ROBOT_WIDTH_M
    CAMERA_FORWARD_OFFSET_M = LocalObstacleProjectionParams.CAMERA_FORWARD_OFFSET_M
    CAMERA_HEIGHT_M = LocalObstacleProjectionParams.CAMERA_HEIGHT_M
    SIDE_MARGIN_M = LocalObstacleProjectionParams.SIDE_MARGIN_M
    MIN_COLLISION_HEIGHT_M = LocalObstacleProjectionParams.MIN_COLLISION_HEIGHT_M
    MAX_COLLISION_HEIGHT_M = LocalObstacleProjectionParams.MAX_COLLISION_HEIGHT_M
    SAFETY_DEPTH_NEAR_M = LocalObstacleProjectionParams.SAFETY_DEPTH_NEAR_M
    SAFETY_DEPTH_STOP_M = LocalObstacleProjectionParams.SAFETY_DEPTH_STOP_M
    MAP_DEPTH_NEAR_M = LocalObstacleProjectionParams.MAP_DEPTH_NEAR_M
    MAP_DEPTH_FAR_M = LocalObstacleProjectionParams.MAP_DEPTH_FAR_M
    MAP_DEPTH_STRIDE = LocalObstacleProjectionParams.MAP_DEPTH_STRIDE
    MAP_IR_MAX_RANGE_M = LocalObstacleProjectionParams.MAP_IR_MAX_RANGE_M
    GROUND_MAX_HEIGHT_M = TerrainParams.GROUND_MAX_HEIGHT_M

    IR_MOUNTS = LocalObstacleProjectionParams.IR_MOUNTS

    def __init__(
        self,
        depth_width: int,
        depth_height: int,
        depth_min_range: float,
        depth_max_range: float,
        depth_hfov: float,
    ):
        self.depth_width = int(depth_width)
        self.depth_height = int(depth_height)
        self.depth_min_range = float(depth_min_range)
        self.depth_max_range = float(depth_max_range)
        self.depth_hfov = float(depth_hfov)

        self.robot_width_m = _env_float("SAFETY_ROBOT_WIDTH_M", self.ROBOT_WIDTH_M)
        self.camera_forward_offset_m = _env_float(
            "SAFETY_CAMERA_FORWARD_OFFSET_M",
            self.CAMERA_FORWARD_OFFSET_M,
        )
        self.camera_height_m = _env_float(
            "SAFETY_CAMERA_HEIGHT_M",
            self.CAMERA_HEIGHT_M,
        )
        self.side_margin_m = _env_float("SAFETY_SIDE_MARGIN_M", self.SIDE_MARGIN_M)
        self.min_collision_height_m = _env_float(
            "SAFETY_MIN_COLLISION_HEIGHT_M",
            self.MIN_COLLISION_HEIGHT_M,
        )
        self.max_collision_height_m = _env_float(
            "SAFETY_MAX_COLLISION_HEIGHT_M",
            self.MAX_COLLISION_HEIGHT_M,
        )
        self.safety_depth_near_m = max(
            self.depth_min_range,
            _env_float("SAFETY_DEPTH_NEAR_M", self.SAFETY_DEPTH_NEAR_M),
        )
        self.safety_depth_stop_m = _env_float(
            "SAFETY_DEPTH_STOP_M",
            self.SAFETY_DEPTH_STOP_M,
        )
        self.map_depth_near_m = max(
            self.depth_min_range,
            _env_float("LOCAL_DEPTH_NEAR_M", self.MAP_DEPTH_NEAR_M),
        )
        self.map_depth_far_m = min(
            self.depth_max_range,
            _env_float("LOCAL_DEPTH_FAR_M", self.MAP_DEPTH_FAR_M),
        )
        self.map_depth_stride = max(
            1,
            _env_int("LOCAL_DEPTH_PIXEL_STRIDE", self.MAP_DEPTH_STRIDE),
        )
        self.map_ir_max_range_m = _env_float(
            "LOCAL_IR_MAX_RANGE_M",
            self.MAP_IR_MAX_RANGE_M,
        )
        self.ground_max_height_m = _env_float(
            "TERRAIN_GROUND_MAX_HEIGHT_M",
            self.GROUND_MAX_HEIGHT_M,
        )
        self._build_depth_geometry()

    def _build_depth_geometry(self):
        rows, columns = np.indices(
            (self.depth_height, self.depth_width),
            dtype=np.float32,
        )
        centre_x = 0.5 * (self.depth_width - 1)
        centre_y = 0.5 * (self.depth_height - 1)
        depth_vfov = 2.0 * math.atan(
            math.tan(self.depth_hfov * 0.5)
            * (self.depth_height / self.depth_width)
        )
        x_scale = math.tan(self.depth_hfov * 0.5) / max(centre_x, 1.0)
        y_scale = math.tan(depth_vfov * 0.5) / max(centre_y, 1.0)
        self._lateral_factor = (centre_x - columns) * x_scale
        self._vertical_down_factor = (rows - centre_y) * y_scale

    def observe(
        self,
        depth,
        ir_ranges,
        terrain: TerrainAssessment = None,
    ) -> LocalObstacleObservation:
        ir_ranges = {
            name: float(ir_ranges.get(name, float("inf")))
            for name in self.IR_MOUNTS
        }
        (
            depth_points,
            safety_depth_points,
            depth_pixels,
            depth_columns,
        ) = self._depth_observation(depth, terrain)
        (
            ir_points,
            collision_ir_ranges,
            ground_like_ir_sensors,
        ) = self._ir_observation(ir_ranges, terrain)
        return LocalObstacleObservation(
            ir_ranges=ir_ranges,
            collision_ir_ranges=collision_ir_ranges,
            ground_like_ir_sensors=ground_like_ir_sensors,
            depth_points=depth_points,
            safety_depth_points=safety_depth_points,
            ir_points=ir_points,
            safety_depth_pixels=depth_pixels,
            safety_depth_columns=depth_columns,
            depth_sampled=depth is not None,
            terrain=terrain,
        )

    def _depth_observation(self, depth, terrain):
        masks = self.depth_debug_masks(depth, terrain)
        if masks is None:
            return (), (), 0, 0

        (
            depth_image,
            _valid,
            mapping_mask,
            safety_mask,
            _tilt_rejected_mask,
            forward_m,
            lateral_m,
            height_m,
        ) = masks
        safety_pixels = int(np.count_nonzero(safety_mask))
        safety_columns = int(np.count_nonzero(np.any(safety_mask, axis=0)))

        mapping_points = self._points_from_depth_mask(
            mapping_mask,
            depth_image,
            forward_m,
            lateral_m,
            height_m,
        )
        safety_points = self._points_from_depth_mask(
            safety_mask,
            depth_image,
            forward_m,
            lateral_m,
            height_m,
        )
        return mapping_points, safety_points, safety_pixels, safety_columns

    def depth_debug_masks(
        self,
        depth,
        terrain: TerrainAssessment = None,
    ):
        if depth is None:
            return None

        depth_image = np.asarray(depth, dtype=np.float32)
        if depth_image.size != self.depth_width * self.depth_height:
            return None
        depth_image = depth_image.reshape((self.depth_height, self.depth_width))

        valid = (
            np.isfinite(depth_image)
            & (depth_image >= self.depth_min_range)
            & (depth_image <= self.depth_max_range)
        )
        geometry_depth = np.where(valid, depth_image, 0.0)
        body_forward_m = self.camera_forward_offset_m + geometry_depth
        body_lateral_m = geometry_depth * self._lateral_factor
        body_height_m = self.camera_height_m - geometry_depth * self._vertical_down_factor
        tilt_active = (
            terrain is not None
            and terrain.reliable
            and terrain.state == TerrainAwarenessLayer.TILTED
        )
        if tilt_active:
            forward_axis, left_axis, up_axis = TerrainAwarenessLayer.level_basis(terrain)
            forward_m = (
                forward_axis[0] * body_forward_m
                + forward_axis[1] * body_lateral_m
                + forward_axis[2] * body_height_m
            )
            lateral_m = (
                left_axis[0] * body_forward_m
                + left_axis[1] * body_lateral_m
                + left_axis[2] * body_height_m
            )
            height_m = (
                up_axis[0] * body_forward_m
                + up_axis[1] * body_lateral_m
                + up_axis[2] * body_height_m
            )
        else:
            forward_m = body_forward_m
            lateral_m = body_lateral_m
            height_m = body_height_m
        height_mask = (
            (height_m >= self.min_collision_height_m)
            & (height_m <= self.max_collision_height_m)
        )
        mapping_mask = (
            valid
            & height_mask
            & (forward_m >= self.map_depth_near_m)
            & (forward_m <= self.map_depth_far_m)
        )
        half_width_m = 0.5 * self.robot_width_m + self.side_margin_m
        safety_mask = (
            valid
            & height_mask
            & (forward_m >= self.safety_depth_near_m)
            & (forward_m <= self.safety_depth_stop_m)
            & (np.abs(lateral_m) <= half_width_m)
        )
        if tilt_active:
            uncompensated_height_mask = (
                (body_height_m >= self.min_collision_height_m)
                & (body_height_m <= self.max_collision_height_m)
            )
            uncompensated_mapping_mask = (
                valid
                & uncompensated_height_mask
                & (body_forward_m >= self.map_depth_near_m)
                & (body_forward_m <= self.map_depth_far_m)
            )
            tilt_rejected_mask = uncompensated_mapping_mask & ~mapping_mask
        else:
            tilt_rejected_mask = np.zeros_like(mapping_mask)
        return (
            depth_image,
            valid,
            mapping_mask,
            safety_mask,
            tilt_rejected_mask,
            forward_m,
            lateral_m,
            height_m,
        )

    def _points_from_depth_mask(
        self,
        mask,
        depth_image,
        forward_m,
        lateral_m,
        height_m,
    ):
        stride = self.map_depth_stride
        sampled_mask = mask[::stride, ::stride]
        sampled_depth = depth_image[::stride, ::stride]
        sampled_forward = forward_m[::stride, ::stride]
        sampled_lateral = lateral_m[::stride, ::stride]
        sampled_height = height_m[::stride, ::stride]

        points = []
        for row, column in np.argwhere(sampled_mask):
            points.append(
                LocalObstaclePoint(
                    source="DEPTH",
                    forward_m=float(sampled_forward[row, column]),
                    lateral_m=float(sampled_lateral[row, column]),
                    height_m=float(sampled_height[row, column]),
                    distance_m=float(sampled_depth[row, column]),
                )
            )
        return tuple(points)

    def _ir_observation(self, ir_ranges, terrain):
        points = []
        collision_ranges = dict(ir_ranges)
        ground_like = []
        forward_axis, left_axis, up_axis = TerrainAwarenessLayer.level_basis(terrain)
        for name, (mount_x, mount_y, mount_z, mount_yaw) in self.IR_MOUNTS.items():
            distance_m = ir_ranges[name]
            if not math.isfinite(distance_m) or distance_m < 0.0:
                continue
            body_point = np.asarray(
                (
                    mount_x + distance_m * math.cos(mount_yaw),
                    mount_y + distance_m * math.sin(mount_yaw),
                    mount_z,
                ),
                dtype=np.float64,
            )
            point_height_m = float(np.dot(body_point, up_axis))
            if (
                terrain is not None
                and terrain.reliable
                and terrain.state == TerrainAwarenessLayer.TILTED
                and point_height_m <= self.ground_max_height_m
            ):
                collision_ranges[name] = float("inf")
                ground_like.append(name)
                continue
            if distance_m > self.map_ir_max_range_m:
                continue
            points.append(
                LocalObstaclePoint(
                    source=name,
                    forward_m=float(np.dot(body_point, forward_axis)),
                    lateral_m=float(np.dot(body_point, left_axis)),
                    height_m=point_height_m,
                    distance_m=distance_m,
                )
            )
        return tuple(points), collision_ranges, tuple(ground_like)


class AnnotatedDepthViewer:
    """
    Render the production depth classification as a live diagnostic image.

    Colours:
    - grey: valid raw depth, brighter means closer
    - dark grey: invalid/out-of-range depth
    - red: pixels eligible for local obstacle mapping
    - yellow: pixels inside the narrower emergency-stop corridor
    - cyan: pixels accepted by level-camera geometry but rejected after tilt

    This is a viewer only. It reads masks from LocalObstacleDetector and never
    changes mapping, safety, planning, or motor output.
    """

    ENABLED_FOR_ROBOT1_BY_DEFAULT = DepthViewerParams.ENABLED_FOR_ROBOT1_BY_DEFAULT
    UPDATE_INTERVAL_S = DepthViewerParams.UPDATE_INTERVAL_S
    HEADER_HEIGHT_PX = DepthViewerParams.HEADER_HEIGHT_PX

    def __init__(
        self,
        robot_id: str,
        detector: LocalObstacleDetector,
        enabled_by_default: bool,
    ):
        self.robot_id = robot_id
        self.detector = detector
        viewer_requested = _env_flag("DEPTH_DEBUG_VIEWER", enabled_by_default)
        self.enabled = _env_flag("DEPTH_DEBUG_RENDER", viewer_requested)
        self.viewer_enabled = bool(viewer_requested and self.enabled)
        self.update_interval_s = max(
            0.05,
            _env_float("DEPTH_DEBUG_VIEWER_INTERVAL_S", self.UPDATE_INTERVAL_S),
        )
        output_dir = Path(
            os.environ.get(
                "DEPTH_DEBUG_VIEWER_DIR",
                str(SLAM_LITE_DIR),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{robot_id}_annotated_depth.png"
        self.last_update_time = -self.update_interval_s
        self.error_message = ""
        self._viewer_process = None

        if self.viewer_enabled:
            self._start_viewer()

    def update(
        self,
        sim_time: float,
        depth,
        observation: LocalObstacleObservation,
        pose: Pose2D,
        command_left: float,
        command_right: float,
        confirmed_depth_cells: int,
        accelerometer_values=None,
        gyro_values=None,
    ):
        if (
            not self.enabled
            or depth is None
            or sim_time - self.last_update_time < self.update_interval_s
        ):
            return
        self.last_update_time = sim_time
        self.render(
            sim_time,
            depth,
            observation,
            pose,
            command_left,
            command_right,
            confirmed_depth_cells,
            accelerometer_values,
            gyro_values,
        )

    def render(
        self,
        sim_time: float,
        depth,
        observation: LocalObstacleObservation,
        pose: Pose2D,
        command_left: float,
        command_right: float,
        confirmed_depth_cells: int,
        accelerometer_values=None,
        gyro_values=None,
    ):
        try:
            from PIL import Image, ImageDraw
        except ImportError as exc:
            self._disable(f"PIL unavailable: {exc}")
            return

        terrain = observation.terrain if observation is not None else None
        masks = self.detector.depth_debug_masks(depth, terrain)
        if masks is None:
            return
        (
            depth_image,
            valid,
            mapping_mask,
            safety_mask,
            tilt_rejected_mask,
            _forward_m,
            _lateral_m,
            _height_m,
        ) = masks

        display_max = min(self.detector.map_depth_far_m, self.detector.depth_max_range)
        display_depth = np.where(valid, depth_image, display_max)
        clipped = np.clip(
            display_depth,
            self.detector.depth_min_range,
            display_max,
        )
        normalised = 1.0 - (
            (clipped - self.detector.depth_min_range)
            / max(display_max - self.detector.depth_min_range, 1e-6)
        )
        grey = (normalised * 210.0 + 25.0).astype(np.uint8)
        rgb = np.stack((grey, grey, grey), axis=2)
        rgb[~valid] = (95, 95, 95)
        rgb[tilt_rejected_mask] = (40, 180, 255)
        rgb[mapping_mask] = (230, 40, 40)
        rgb[safety_mask] = (255, 210, 0)

        canvas = Image.new(
            "RGB",
            (
                self.detector.depth_width,
                self.detector.depth_height + self.HEADER_HEIGHT_PX,
            ),
            (245, 245, 245),
        )
        canvas.paste(Image.fromarray(rgb), (0, self.HEADER_HEIGHT_PX))
        draw = ImageDraw.Draw(canvas)

        mapping_pixels = int(np.count_nonzero(mapping_mask))
        safety_pixels = (
            observation.safety_depth_pixels
            if observation is not None
            else int(np.count_nonzero(safety_mask))
        )
        safety_columns = (
            observation.safety_depth_columns
            if observation is not None
            else int(np.count_nonzero(np.any(safety_mask, axis=0)))
        )
        sampled_points = len(observation.depth_points) if observation else 0
        terrain_state = terrain.state if terrain is not None else "UNAVAILABLE"
        pitch_deg = math.degrees(terrain.pitch_rad) if terrain is not None else 0.0
        roll_deg = math.degrees(terrain.roll_rad) if terrain is not None else 0.0
        tilt_deg = math.degrees(terrain.tilt_rad) if terrain is not None else 0.0
        accel = self._vector_or_none(accelerometer_values)
        gyro = self._vector_or_none(gyro_values)
        accel_norm = float(np.linalg.norm(accel)) if accel is not None else 0.0
        gyro_tilt_rate_deg_s = (
            math.degrees(math.hypot(float(gyro[0]), float(gyro[1])))
            if gyro is not None
            else 0.0
        )
        ground_ir = (
            ",".join(observation.ground_like_ir_sensors)
            if observation is not None and observation.ground_like_ir_sensors
            else "-"
        )

        lines = (
            f"{self.robot_id} LIVE DEPTH  t={sim_time:.2f}s",
            (
                f"world x={pose.x:.3f} y={pose.y:.3f} "
                f"yaw={math.degrees(pose.yaw):.2f} deg"
            ),
            (
                f"cmd L={command_left:.2f} R={command_right:.2f} rad/s  "
                f"map pixels={mapping_pixels} sampled points={sampled_points}"
            ),
            (
                f"safety={safety_pixels} px/{safety_columns} cols  "
                f"confirmed red cells={confirmed_depth_cells}"
            ),
            (
                f"terrain={terrain_state} pitch={pitch_deg:+.2f} deg "
                f"roll={roll_deg:+.2f} deg tilt={tilt_deg:+.2f} deg"
            ),
            (
                f"accel={self._format_vector(accel)} |a|={accel_norm:.2f} m/s2 "
                f"startup_gated={terrain.accel_startup_gated if terrain is not None else False}"
            ),
            (
                f"gyro={self._format_vector(gyro)} rad/s "
                f"tilt_rate={gyro_tilt_rate_deg_s:.2f} deg/s ground IR={ground_ir}"
            ),
            "red=map  yellow=safety  cyan=tilt-rejected  grey=invalid",
            "image left=robot left  |  image right=robot right",
        )
        for row, text in enumerate(lines):
            draw.text((10, 8 + row * 20), text, fill=(0, 0, 0))

        tmp_path = self.output_path.with_suffix(".tmp.png")
        try:
            canvas.save(tmp_path)
            os.replace(tmp_path, self.output_path)
        except OSError as exc:
            self._disable(f"could not write image: {exc}")

    def _start_viewer(self):
        script = LIVE_MAP_VIEWER_PATH
        if not script.exists():
            return
        try:
            self._viewer_process = subprocess.Popen(
                [
                    sys.executable,
                    str(script),
                    str(self.output_path),
                    f"{self.robot_id} annotated depth",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            atexit.register(self.stop)
        except OSError as exc:
            self._disable(f"could not start viewer: {exc}")

    def stop(self):
        if self._viewer_process and self._viewer_process.poll() is None:
            self._viewer_process.terminate()

    def _disable(self, reason: str):
        if self.enabled:
            print(f"[{self.robot_id}] Annotated depth viewer disabled: {reason}")
        self.stop()
        self.enabled = False

    @staticmethod
    def _vector_or_none(values):
        if values is None or len(values) < 3:
            return None
        try:
            return np.asarray(values[:3], dtype=np.float32)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _format_vector(vector):
        if vector is None:
            return "(-,-,-)"
        return (
            f"({float(vector[0]):+.2f},"
            f"{float(vector[1]):+.2f},"
            f"{float(vector[2]):+.2f})"
        )
