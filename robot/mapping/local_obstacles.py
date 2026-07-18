"""Confirmed depth/IR obstacle cells for a robot live map."""

from __future__ import annotations

import math

from config import _env_float, _env_int, _env_flag
from parameters import LocalObstacleMap as LocalObstacleMapParams
from shared_types import LocalObstacleObservation, Pose2D


class LocalObstacleLayer:
    """Confirm and store depth/IR obstacle evidence for one robot's live map."""

    DEPTH_CONFIRM_FRAMES = LocalObstacleMapParams.DEPTH_CONFIRM_FRAMES
    IR_CONFIRM_FRAMES = LocalObstacleMapParams.IR_CONFIRM_FRAMES
    IR_PATCH_RADIUS_M = LocalObstacleMapParams.IR_PATCH_RADIUS_M
    OBSTACLE_TTL_S = LocalObstacleMapParams.OBSTACLE_TTL_S

    def __init__(self, resolution_m: float, enabled_by_default: bool):
        self.enabled = _env_flag(
            "LOCAL_OBSTACLE_MAPPING_ENABLED",
            enabled_by_default,
        )
        self.resolution_m = float(resolution_m)
        self.depth_confirm_frames = max(
            1,
            _env_int("LOCAL_DEPTH_CONFIRM_FRAMES", self.DEPTH_CONFIRM_FRAMES),
        )
        self.ir_confirm_frames = max(
            1,
            _env_int("LOCAL_IR_CONFIRM_FRAMES", self.IR_CONFIRM_FRAMES),
        )
        self.ir_patch_radius_m = max(
            0.0,
            _env_float("LOCAL_IR_PATCH_RADIUS_M", self.IR_PATCH_RADIUS_M),
        )
        self.sticky = _env_flag("LOCAL_OBSTACLE_STICKY", LocalObstacleMapParams.STICKY)
        self.ttl_s = max(
            0.0,
            _env_float("LOCAL_OBSTACLE_TTL_S", self.OBSTACLE_TTL_S),
        )

        self.depth_cells = set()
        self.ir_cells = set()
        self._depth_candidates = {}
        self._ir_candidates = {}
        self._depth_last_seen = {}
        self._ir_last_seen = {}

    @property
    def occupied_cells(self):
        return self.depth_cells | self.ir_cells

    def update(
        self,
        observation: LocalObstacleObservation,
        pose: Pose2D,
        sim_time: float,
        world_to_grid,
        in_bounds,
    ):
        if not self.enabled or observation is None:
            return False

        depth_cells = self._project_points(
            observation.depth_points,
            pose,
            world_to_grid,
            in_bounds,
        )
        ir_centres = self._project_points(
            observation.ir_points,
            pose,
            world_to_grid,
            in_bounds,
        )
        ir_cells = self._expand_cells(
            ir_centres,
            self.ir_patch_radius_m,
            in_bounds,
        )

        changed = False
        if observation.depth_sampled:
            changed = self._confirm_cells(
                depth_cells,
                self._depth_candidates,
                self.depth_cells,
                self._depth_last_seen,
                self.depth_confirm_frames,
                sim_time,
            )
        changed |= self._confirm_cells(
            ir_cells,
            self._ir_candidates,
            self.ir_cells,
            self._ir_last_seen,
            self.ir_confirm_frames,
            sim_time,
        )
        if not self.sticky:
            changed |= self._expire_cells(sim_time)
        return changed

    def force_confirm(
        self,
        observation: LocalObstacleObservation,
        pose: Pose2D,
        sim_time: float,
        source: str,
        world_to_grid,
        in_bounds,
    ):
        if not self.enabled or observation is None:
            return False

        changed = False
        if source == "DEPTH":
            cells = self._project_points(
                observation.safety_depth_points,
                pose,
                world_to_grid,
                in_bounds,
            )
            changed |= self._add_confirmed(
                cells,
                self.depth_cells,
                self._depth_last_seen,
                sim_time,
            )
        else:
            allowed = {
                "FRONT_IR": {"fl_range", "fr_range"},
                "REAR_IR": {"rl_range", "rr_range"},
                "TURN_IR": {"fl_range", "fr_range", "rl_range", "rr_range"},
            }.get(source, {source})
            points = tuple(
                point
                for point in observation.ir_points
                if point.source in allowed
            )
            centres = self._project_points(
                points,
                pose,
                world_to_grid,
                in_bounds,
            )
            cells = self._expand_cells(
                centres,
                self.ir_patch_radius_m,
                in_bounds,
            )
            changed |= self._add_confirmed(
                cells,
                self.ir_cells,
                self._ir_last_seen,
                sim_time,
            )
        return changed

    def _project_points(self, points, pose, world_to_grid, in_bounds):
        c = math.cos(pose.yaw)
        s = math.sin(pose.yaw)
        cells = set()
        for point in points:
            world_x = pose.x + c * point.forward_m - s * point.lateral_m
            world_y = pose.y + s * point.forward_m + c * point.lateral_m
            cell = world_to_grid(world_x, world_y)
            if in_bounds(*cell):
                cells.add(cell)
        return cells

    def _expand_cells(self, centres, radius_m: float, in_bounds):
        radius_px = int(math.ceil(radius_m / max(self.resolution_m, 1e-9)))
        if radius_px <= 0:
            return set(centres)

        cells = set()
        radius2 = radius_px * radius_px
        for centre_x, centre_y in centres:
            for dy in range(-radius_px, radius_px + 1):
                for dx in range(-radius_px, radius_px + 1):
                    if dx * dx + dy * dy > radius2:
                        continue
                    cell = (centre_x + dx, centre_y + dy)
                    if in_bounds(*cell):
                        cells.add(cell)
        return cells

    @staticmethod
    def _confirm_cells(
        observed,
        candidates,
        confirmed,
        last_seen,
        required_frames: int,
        sim_time: float,
    ):
        next_candidates = {}
        changed = False
        for cell in observed:
            last_seen[cell] = sim_time
            if cell in confirmed:
                continue
            count = candidates.get(cell, 0) + 1
            if count >= required_frames:
                confirmed.add(cell)
                changed = True
            else:
                next_candidates[cell] = count
        candidates.clear()
        candidates.update(next_candidates)
        return changed

    @staticmethod
    def _add_confirmed(cells, confirmed, last_seen, sim_time: float):
        changed = False
        for cell in cells:
            last_seen[cell] = sim_time
            if cell not in confirmed:
                confirmed.add(cell)
                changed = True
        return changed

    def _expire_cells(self, sim_time: float):
        changed = False
        for cells, last_seen in (
            (self.depth_cells, self._depth_last_seen),
            (self.ir_cells, self._ir_last_seen),
        ):
            expired = {
                cell
                for cell in cells
                if sim_time - last_seen.get(cell, sim_time) > self.ttl_s
            }
            if expired:
                cells.difference_update(expired)
                for cell in expired:
                    last_seen.pop(cell, None)
                changed = True
        return changed
