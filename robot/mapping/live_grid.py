"""Live occupancy grid built from lidar plus confirmed local obstacles."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from config import SLAM_LITE_DIR, _env_float, _env_flag
from parameters import LiveMap as LiveMapParams
from robot.mapping.local_obstacles import LocalObstacleLayer
from robot.sensing import TerrainAwarenessLayer
from shared_types import (
    LiveMapSnapshot,
    LocalObstacleObservation,
    Pose2D,
    TerrainAssessment,
)

if TYPE_CHECKING:
    from robot.mapping.drone_map import DroneExtractionMap


class LiveOccupancyGridMapper:
    """Live occupancy-grid mapper with lidar and confirmed local obstacles."""

    RESOLUTION_M_PER_PX = LiveMapParams.RESOLUTION_M_PER_PX
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1

    def __init__(
        self,
        robot_id: str,
        enabled_by_default: bool,
        drone_map: DroneExtractionMap = None,
    ):
        self.robot_id = robot_id
        self.enabled = _env_flag("LIVE_MAP_ENABLED", enabled_by_default)
        self.lidar_hz = _env_float("LIDAR_HZ", LiveMapParams.LIDAR_HZ)
        self.update_interval_s = (
            0.0 if self.lidar_hz <= 0.0 else 1.0 / self.lidar_hz
        )
        self.render_interval_s = _env_float(
            "LIVE_MAP_RENDER_INTERVAL_S",
            LiveMapParams.RENDER_INTERVAL_S,
        )
        self.render_padding_px = max(
            0,
            int(_env_float("LIVE_MAP_RENDER_PADDING_PX", LiveMapParams.RENDER_PADDING_PX)),
        )
        self.render_enabled = _env_flag("LIVE_MAP_RENDER", True)
        self.lidar_min_range_m = _env_float(
            "LIDAR_MIN_RANGE_M",
            LiveMapParams.LIDAR_MIN_RANGE_M,
        )
        self.lidar_max_range_m = _env_float(
            "LIDAR_MAX_RANGE_M",
            LiveMapParams.LIDAR_MAX_RANGE_M,
        )
        self.lidar_angle_offset_rad = _env_float(
            "LIDAR_ANGLE_OFFSET_RAD",
            LiveMapParams.LIDAR_ANGLE_OFFSET_RAD,
        )
        self.lidar_reverse = _env_flag("LIDAR_REVERSE", LiveMapParams.LIDAR_REVERSE)
        self.yaw_sign = _env_float("LIVE_MAP_YAW_SIGN", LiveMapParams.YAW_SIGN)
        self.sticky_occupied = _env_flag(
            "LIVE_MAP_STICKY_OCCUPIED",
            LiveMapParams.STICKY_OCCUPIED,
        )
        self.mark_inf_free = _env_flag("LIDAR_MARK_INF_FREE", LiveMapParams.MARK_INF_FREE)
        self.lidar_ray_stride = max(
            1,
            int(_env_float("LIDAR_RAY_STRIDE", LiveMapParams.LIDAR_RAY_STRIDE)),
        )
        self.outlier_filter_enabled = _env_flag(
            "LIDAR_OUTLIER_FILTER",
            LiveMapParams.OUTLIER_FILTER,
        )
        self.outlier_threshold_m = _env_float(
            "LIDAR_OUTLIER_THRESHOLD_M",
            LiveMapParams.OUTLIER_THRESHOLD_M,
        )
        self.terrain_pause_enabled = _env_flag(
            "LIDAR_TERRAIN_PAUSE_ENABLED",
            LiveMapParams.TERRAIN_PAUSE_ENABLED,
        )
        self.lidar_mapping_paused = False
        self.lidar_pause_reason = ""

        self.hit_confidence = _env_float(
            "LIVE_MAP_HIT_CONFIDENCE",
            LiveMapParams.HIT_CONFIDENCE,
        )
        self.free_confidence = _env_float(
            "LIVE_MAP_FREE_CONFIDENCE",
            LiveMapParams.FREE_CONFIDENCE,
        )
        self.free_threshold = _env_float(
            "LIVE_MAP_FREE_THRESHOLD",
            LiveMapParams.FREE_THRESHOLD,
        )
        self.occupied_threshold = _env_float(
            "LIVE_MAP_OCCUPIED_THRESHOLD",
            LiveMapParams.OCCUPIED_THRESHOLD,
        )
        self.confidence_min = _env_float(
            "LIVE_MAP_CONFIDENCE_MIN",
            LiveMapParams.CONFIDENCE_MIN,
        )
        self.confidence_max = _env_float(
            "LIVE_MAP_CONFIDENCE_MAX",
            LiveMapParams.CONFIDENCE_MAX,
        )

        self.scan_match_enabled = _env_flag("LIVE_MAP_SCAN_MATCH", LiveMapParams.SCAN_MATCH)
        self.scan_match_xy_step_m = _env_float(
            "SCAN_MATCH_XY_STEP_M",
            LiveMapParams.SCAN_MATCH_XY_STEP_M,
        )
        self.scan_match_yaw_step_rad = _env_float(
            "SCAN_MATCH_YAW_STEP_RAD",
            LiveMapParams.SCAN_MATCH_YAW_STEP_RAD,
        )
        self.scan_match_stride = max(
            1,
            int(_env_float("SCAN_MATCH_STRIDE", LiveMapParams.SCAN_MATCH_STRIDE)),
        )
        self.scan_match_radius_px = max(
            0,
            int(
                _env_float(
                    "SCAN_MATCH_OCCUPIED_RADIUS_PX",
                    LiveMapParams.SCAN_MATCH_OCCUPIED_RADIUS_PX,
                )
            ),
        )
        self.scan_match_odom_penalty = _env_float(
            "SCAN_MATCH_ODOM_PENALTY",
            LiveMapParams.SCAN_MATCH_ODOM_PENALTY,
        )
        self.scan_match_min_hits = max(
            1,
            int(_env_float("SCAN_MATCH_MIN_HITS", LiveMapParams.SCAN_MATCH_MIN_HITS)),
        )

        self.drone_map = drone_map
        self.seed_from_drone = (
            drone_map is not None
            and drone_map.ready
            and _env_flag("LIVE_MAP_SEED_FROM_DRONE", LiveMapParams.SEED_FROM_DRONE)
        )
        self.geometry = drone_map.geometry if self.seed_from_drone else None
        self.seeded_map_size = drone_map.map_size if self.seed_from_drone else None
        self.seed_occupied_confidence = _env_float(
            "DRONE_MAP_OCCUPIED_CONFIDENCE",
            LiveMapParams.DRONE_OCCUPIED_CONFIDENCE,
        )
        self.seed_free_confidence = _env_float(
            "DRONE_MAP_FREE_CONFIDENCE",
            LiveMapParams.DRONE_FREE_CONFIDENCE,
        )
        self.seed_threshold = int(
            _env_float(
                "DRONE_MAP_OCCUPIED_PIXEL_THRESHOLD",
                LiveMapParams.DRONE_OCCUPIED_PIXEL_THRESHOLD,
            )
        )

        self.requested_start_x = self._optional_env_float("LIVE_MAP_START_X")
        self.requested_start_y = self._optional_env_float("LIVE_MAP_START_Y")
        self.requested_start_yaw = self._optional_env_float("LIVE_MAP_START_YAW")
        if self.seed_from_drone and not _env_flag(
            "LIVE_MAP_USE_START_OVERRIDE_WITH_DRONE",
            LiveMapParams.USE_START_OVERRIDE_WITH_DRONE,
        ):
            self.requested_start_x = None
            self.requested_start_y = None
            self.requested_start_yaw = None
        self.pose_offset_ready = False
        self.pose_offset_x = 0.0
        self.pose_offset_y = 0.0
        self.pose_offset_yaw = 0.0

        output_dir = Path(
            os.environ.get(
                "LIVE_MAP_DIR",
                str(SLAM_LITE_DIR),
            )
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / f"{robot_id}_live_map.png"
        self._viewer_process = None
        viewer_default = self.enabled and self.render_enabled and robot_id == "robot1"
        if _env_flag("LIVE_MAP_VIEWER", viewer_default):
            self._start_viewer()

        self.confidence_grid = {}
        self.occupied_cells = set()
        local_resolution = (
            self.geometry.resolution
            if self.geometry is not None
            else self.RESOLUTION_M_PER_PX
        )
        self.local_obstacles = LocalObstacleLayer(
            local_resolution,
            enabled_by_default=self.enabled,
        )
        self.revision = 0
        self._revision_dirty = False
        if self.seed_from_drone:
            self._seed_confidence_from_drone_map()
        if self.confidence_grid:
            self.revision = 1

        self.latest_hit_cells = []
        self.latest_ray_cells = []
        self.last_update_time = -self.update_interval_s
        self.last_render_time = -self.render_interval_s
        self.last_pose = Pose2D()
        self.last_scan_match_dx = 0.0
        self.last_scan_match_dy = 0.0
        self.last_scan_match_dyaw = 0.0
        self.last_scan_match_score = 0.0
        self._render_available = True

    def should_read_lidar(
        self,
        sim_time: float,
        terrain: TerrainAssessment = None,
    ) -> bool:
        if not self.enabled:
            return False
        if self._terrain_lidar_pause_reason(terrain):
            return False
        return (
            self.update_interval_s <= 0.0
            or sim_time - self.last_update_time >= self.update_interval_s
        )

    def update(
        self,
        pose: Pose2D,
        sim_time: float,
        lidar_ranges=(),
        lidar_fov: float = 2.0 * math.pi,
        observation: LocalObstacleObservation = None,
        terrain: TerrainAssessment = None,
    ):
        if not self.enabled:
            return False

        if terrain is None and observation is not None:
            terrain = observation.terrain
        self.last_pose = self._pose_for_map(pose)
        map_changed = self.local_obstacles.update(
            observation,
            self.last_pose,
            sim_time,
            self.world_to_grid,
            self._cell_in_bounds,
        )
        pause_reason = self._terrain_lidar_pause_reason(terrain)
        was_paused = self.lidar_mapping_paused
        self.lidar_mapping_paused = bool(pause_reason)
        self.lidar_pause_reason = pause_reason
        if self.lidar_mapping_paused and not was_paused:
            self.latest_hit_cells = []
            self.latest_ray_cells = []
            self.last_scan_match_dx = 0.0
            self.last_scan_match_dy = 0.0
            self.last_scan_match_dyaw = 0.0
            self.last_scan_match_score = 0.0

        if (
            not self.lidar_mapping_paused
            and lidar_ranges
            and (
                self.update_interval_s <= 0.0
                or sim_time - self.last_update_time >= self.update_interval_s
            )
        ):
            self._revision_dirty = False
            self._update_from_lidar_scan(pose, lidar_ranges, lidar_fov)
            map_changed |= self._revision_dirty
            self.last_update_time = sim_time
        if map_changed:
            self.revision += 1

        rendered = False
        if self.render_enabled and sim_time - self.last_render_time >= self.render_interval_s:
            self.render()
            self.last_render_time = sim_time
            rendered = True
        return rendered

    def _terrain_lidar_pause_reason(self, terrain: TerrainAssessment):
        if not self.terrain_pause_enabled or terrain is None:
            return ""
        if terrain.accel_startup_gated:
            return "startup accel gated"
        if terrain.state == TerrainAwarenessLayer.TILT_CANDIDATE:
            return "tilt candidate"
        if terrain.state == TerrainAwarenessLayer.TILTED:
            return "tilted / waiting for stable level"
        return ""

    def render(self):
        from robot.mapping.render import render_live_grid

        render_live_grid(self)

    def world_to_grid(self, x_m: float, y_m: float):
        if self.geometry is not None:
            return self.geometry.world_to_pixel(x_m, y_m)

        gx = int(round(x_m / self.RESOLUTION_M_PER_PX))
        gy = int(round(-y_m / self.RESOLUTION_M_PER_PX))
        return gx, gy

    def planning_snapshot(self):
        if (
            not self.enabled
            or self.geometry is None
            or self.seeded_map_size is None
        ):
            return None
        width, height = self.seeded_map_size
        occupied_cells = self.occupied_cells | self.local_obstacles.occupied_cells
        return LiveMapSnapshot(
            geometry=self.geometry,
            width=width,
            height=height,
            occupied_cells=frozenset(occupied_cells),
            revision=self.revision,
        )

    def confirm_safety_obstacle(
        self,
        observation: LocalObstacleObservation,
        pose: Pose2D,
        sim_time: float,
        source: str,
    ):
        if not self.enabled:
            return False
        map_pose = self._pose_for_map(pose)
        changed = self.local_obstacles.force_confirm(
            observation,
            map_pose,
            sim_time,
            source,
            self.world_to_grid,
            self._cell_in_bounds,
        )
        if changed:
            self.revision += 1
            if self.render_enabled:
                self.render()
        return changed

    def _start_viewer(self):
        from robot.mapping.render import start_viewer

        self._viewer_process = start_viewer(
            self.output_path,
            f"{self.robot_id} live map",
            f"[{self.robot_id}] Could not start live map viewer",
        )

    def _stop_viewer(self):
        from robot.mapping.render import stop_viewer

        stop_viewer(self._viewer_process)

    def _seed_confidence_from_drone_map(self):
        try:
            from PIL import Image
        except ImportError as exc:
            print(f"[{self.robot_id}] Could not seed live map from drone map: {exc}")
            self.seed_from_drone = False
            self.geometry = None
            self.seeded_map_size = None
            return

        try:
            image = Image.open(self.drone_map.map_path).convert("L")
        except OSError as exc:
            print(f"[{self.robot_id}] Could not open drone map seed: {exc}")
            self.seed_from_drone = False
            self.geometry = None
            self.seeded_map_size = None
            return

        self.seeded_map_size = image.size
        width, height = image.size
        for py in range(height):
            for px in range(width):
                pixel = image.getpixel((px, py))
                if pixel < self.seed_threshold:
                    self.confidence_grid[(px, py)] = self.seed_occupied_confidence
                    if self.seed_occupied_confidence >= self.occupied_threshold:
                        self.occupied_cells.add((px, py))
                else:
                    self.confidence_grid[(px, py)] = self.seed_free_confidence

    def _pose_for_map(self, pose: Pose2D):
        if not self.pose_offset_ready:
            if self.requested_start_x is not None:
                self.pose_offset_x = self.requested_start_x - pose.x
            if self.requested_start_y is not None:
                self.pose_offset_y = self.requested_start_y - pose.y
            if self.requested_start_yaw is not None:
                self.pose_offset_yaw = self._wrap_angle(self.requested_start_yaw - pose.yaw)
            self.pose_offset_ready = True

        return Pose2D(
            x=pose.x + self.pose_offset_x,
            y=pose.y + self.pose_offset_y,
            yaw=self._wrap_angle(pose.yaw + self.pose_offset_yaw),
        )

    def _update_from_lidar_scan(self, pose: Pose2D, ranges, fov: float):
        ranges = tuple(ranges)
        if not ranges:
            return

        filtered_ranges = self._filtered_ranges(ranges)
        map_base_pose = self._pose_for_map(pose)
        map_pose = self._scan_matched_pose(map_base_pose, filtered_ranges, fov)
        self.last_pose = map_pose
        start = self.world_to_grid(map_pose.x, map_pose.y)
        self.latest_hit_cells = []
        self.latest_ray_cells = []

        for i, range_m in enumerate(filtered_ranges):
            if i % self.lidar_ray_stride != 0:
                continue

            ray_angle = self.yaw_sign * map_pose.yaw + self._lidar_angle(i, len(ranges), fov)
            is_hit = range_m is not None
            if not is_hit and not self.mark_inf_free:
                continue

            ray_range_m = range_m if is_hit else self.lidar_max_range_m
            end_x = map_pose.x + ray_range_m * math.cos(ray_angle)
            end_y = map_pose.y + ray_range_m * math.sin(ray_angle)
            end = self.world_to_grid(end_x, end_y)
            cells = self._bresenham(start[0], start[1], end[0], end[1])
            if not cells:
                continue

            free_cells = cells[:-1] if is_hit else cells
            for gx, gy in free_cells:
                self._set_cell(gx, gy, self.FREE)

            if is_hit:
                gx, gy = cells[-1]
                self._set_cell(gx, gy, self.OCCUPIED)
                self.latest_hit_cells.append((gx, gy))

    def _filtered_ranges(self, ranges):
        filtered = []
        for range_m in ranges:
            if (
                math.isfinite(range_m)
                and self.lidar_min_range_m <= range_m <= self.lidar_max_range_m
            ):
                filtered.append(range_m)
            else:
                filtered.append(None)

        if not self.outlier_filter_enabled:
            return filtered

        cleaned = list(filtered)
        for i, range_m in enumerate(filtered):
            if range_m is None:
                continue

            neighbours = []
            for j in (i - 2, i - 1, i + 1, i + 2):
                if 0 <= j < len(filtered) and filtered[j] is not None:
                    neighbours.append(filtered[j])

            if len(neighbours) < 2:
                continue

            neighbours.sort()
            neighbour_median = neighbours[len(neighbours) // 2]
            if abs(range_m - neighbour_median) > self.outlier_threshold_m:
                cleaned[i] = None

        return cleaned

    def _scan_matched_pose(self, base_pose: Pose2D, filtered_ranges, fov: float):
        self.last_scan_match_dx = 0.0
        self.last_scan_match_dy = 0.0
        self.last_scan_match_dyaw = 0.0
        self.last_scan_match_score = 0.0

        if not self.scan_match_enabled:
            return base_pose

        sampled_hits = [
            (i, range_m)
            for i, range_m in enumerate(filtered_ranges)
            if range_m is not None and i % self.scan_match_stride == 0
        ]
        if len(sampled_hits) < self.scan_match_min_hits:
            return base_pose

        xy_offsets = (-self.scan_match_xy_step_m, 0.0, self.scan_match_xy_step_m)
        yaw_offsets = (
            -self.scan_match_yaw_step_rad,
            0.0,
            self.scan_match_yaw_step_rad,
        )
        best_pose = base_pose
        best_score = None
        best_offsets = (0.0, 0.0, 0.0)

        for dx in xy_offsets:
            for dy in xy_offsets:
                for dyaw in yaw_offsets:
                    candidate = Pose2D(
                        x=base_pose.x + dx,
                        y=base_pose.y + dy,
                        yaw=self._wrap_angle(base_pose.yaw + dyaw),
                    )
                    score = self._score_scan_candidate(
                        candidate,
                        sampled_hits,
                        len(filtered_ranges),
                        fov,
                    )
                    odom_penalty = self.scan_match_odom_penalty * (
                        abs(dx) / max(self.scan_match_xy_step_m, 1e-6)
                        + abs(dy) / max(self.scan_match_xy_step_m, 1e-6)
                        + abs(dyaw) / max(self.scan_match_yaw_step_rad, 1e-6)
                    )
                    score -= odom_penalty
                    if best_score is None or score > best_score:
                        best_score = score
                        best_pose = candidate
                        best_offsets = (dx, dy, dyaw)

        self.last_scan_match_dx = best_offsets[0]
        self.last_scan_match_dy = best_offsets[1]
        self.last_scan_match_dyaw = best_offsets[2]
        self.last_scan_match_score = best_score if best_score is not None else 0.0
        return best_pose

    def _score_scan_candidate(self, candidate: Pose2D, sampled_hits, count: int, fov: float):
        total_score = 0.0
        for i, range_m in sampled_hits:
            ray_angle = self.yaw_sign * candidate.yaw + self._lidar_angle(i, count, fov)
            hit_x = candidate.x + range_m * math.cos(ray_angle)
            hit_y = candidate.y + range_m * math.sin(ray_angle)
            gx, gy = self.world_to_grid(hit_x, hit_y)
            total_score += self._neighbourhood_confidence(gx, gy)
        return total_score / max(len(sampled_hits), 1)

    def _neighbourhood_confidence(self, gx: int, gy: int):
        best = 0.0
        for yy in range(gy - self.scan_match_radius_px, gy + self.scan_match_radius_px + 1):
            for xx in range(gx - self.scan_match_radius_px, gx + self.scan_match_radius_px + 1):
                best = max(best, self.confidence_grid.get((xx, yy), 0.0))
        return best

    def _lidar_angle(self, index: int, count: int, fov: float):
        logical_index = count - 1 - index if self.lidar_reverse else index
        return (
            -0.5 * fov
            + ((logical_index + 0.5) / count) * fov
            + self.lidar_angle_offset_rad
        )

    def _set_cell(self, gx: int, gy: int, value: int):
        if not self._cell_in_bounds(gx, gy):
            return

        cell = (gx, gy)
        old_confidence = self.confidence_grid.get(cell, 0.0)
        old_occupied = old_confidence >= self.occupied_threshold
        if (
            self.sticky_occupied
            and old_confidence >= self.occupied_threshold
            and value == self.FREE
        ):
            return

        if value == self.OCCUPIED:
            delta = self.hit_confidence
        elif value == self.FREE:
            delta = -self.free_confidence
        else:
            delta = 0.0

        new_confidence = self._clamp(
            old_confidence + delta,
            self.confidence_min,
            self.confidence_max,
        )
        if abs(new_confidence) < 1e-9:
            self.confidence_grid.pop(cell, None)
        else:
            self.confidence_grid[cell] = new_confidence

        new_occupied = new_confidence >= self.occupied_threshold
        if new_occupied:
            self.occupied_cells.add(cell)
        else:
            self.occupied_cells.discard(cell)
        if new_occupied != old_occupied:
            self._revision_dirty = True

    def _cell_in_bounds(self, gx: int, gy: int):
        return self.geometry is None or self.geometry.in_bounds(gx, gy)

    @staticmethod
    def _bresenham(x0: int, y0: int, x1: int, y1: int):
        cells = []
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        x, y = x0, y0

        while True:
            cells.append((x, y))
            if x == x1 and y == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x += sx
            if e2 <= dx:
                err += dx
                y += sy
        return cells

    @staticmethod
    def _wrap_angle(angle_rad: float):
        return math.atan2(math.sin(angle_rad), math.cos(angle_rad))

    @staticmethod
    def _clamp(value: float, low: float, high: float):
        return max(low, min(high, value))

    @staticmethod
    def _optional_env_float(name: str):
        text = os.environ.get(name)
        if text is None or text == "":
            return None
        try:
            return float(text)
        except ValueError:
            print(f"[CONFIG] Ignoring invalid {name}={text!r}; using no override")
            return None
