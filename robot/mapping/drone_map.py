"""Static drone-map loading and victim-prior extraction."""

from __future__ import annotations

import csv
import json
import os

from config import (
    CONTROLLER_DIR,
    DRONE_MAP_IMAGE_PATH,
    DRONE_MAP_INFO_PATH,
    DRONE_MAP_VICTIMS_PATH,
    SIM_LOGS_DIR,
    SLAM_LITE_DIR,
    resolve_runtime_path,
    _env_flag,
)
from parameters import DroneMap as DroneMapParams
from robot.mapping.geometry import MapGeometry


class DroneExtractionMap:
    """
    Load the drone map, metadata, and victim estimates.

    Local robot runtimes may load this silently for path planning. The debug
    overlay image belongs to the coordinator host, so normal robot clients can
    set render_overlay=False and still get the map geometry/victim data.
    """

    ROBOT_STARTS = DroneMapParams.ROBOT_STARTS

    def __init__(
        self,
        robot_id: str,
        enabled_by_default: bool,
        render_overlay: bool = True,
    ):
        self.robot_id = robot_id
        self.enabled = _env_flag("DRONE_MAP_ENABLED", enabled_by_default)
        self.render_overlay_enabled = bool(render_overlay) and _env_flag(
            "DRONE_MAP_RENDER",
            bool(render_overlay),
        )
        self.base_dir = CONTROLLER_DIR
        self.sim_logs_dir = resolve_runtime_path(
            os.environ.get("DRONE_MAP_SIM_LOGS_DIR", str(SIM_LOGS_DIR))
        )
        self.map_path = resolve_runtime_path(
            os.environ.get(
                "DRONE_MAP_IMAGE",
                str(self.sim_logs_dir / DRONE_MAP_IMAGE_PATH.name),
            )
        )
        self.info_path = resolve_runtime_path(
            os.environ.get(
                "DRONE_MAP_INFO",
                str(self.sim_logs_dir / DRONE_MAP_INFO_PATH.name),
            )
        )
        self.victims_path = resolve_runtime_path(
            os.environ.get(
                "DRONE_MAP_VICTIMS",
                str(self.sim_logs_dir / DRONE_MAP_VICTIMS_PATH.name),
            )
        )
        output_dir = resolve_runtime_path(
            os.environ.get("DRONE_MAP_DIR", str(SLAM_LITE_DIR))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_path = output_dir / "drone_map_overlay.png"

        self.geometry = None
        self.victim_estimates = []
        self.map_size = None
        self.origin_px_from_formula = None
        self.ready = False
        self.error_message = ""
        self._viewer_process = None

        if self.enabled:
            self.ready = self.render_overlay()
            viewer_default = (
                self.ready
                and self.robot_id in ("robot1", "coordinator")
                and self.render_overlay_enabled
            )
            if _env_flag("DRONE_MAP_VIEWER", viewer_default):
                self._start_viewer()

    @property
    def victim_count(self):
        return len(self.victim_estimates)

    def render_overlay(self):
        try:
            from PIL import Image
        except ImportError as exc:
            self.error_message = f"PIL unavailable: {exc}"
            print(f"[{self.robot_id}] Drone map disabled: {self.error_message}")
            return False

        try:
            with open(self.info_path, "r", encoding="utf-8") as f:
                info = json.load(f)
            self.geometry = MapGeometry(info)
            self.victim_estimates = self._load_victim_estimates()
            image = Image.open(self.map_path).convert("RGB")
            self.map_image = image.copy()
            self.map_size = image.size
        except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
            self.error_message = str(exc)
            print(f"[{self.robot_id}] Drone map disabled: {self.error_message}")
            return False

        if not self.render_overlay_enabled:
            return True

        from robot.mapping.render import render_drone_map_overlay

        render_drone_map_overlay(self, image)
        return True

    def _load_victim_estimates(self):
        estimates = []
        with open(self.victims_path, "r", encoding="utf-8", newline="") as f:
            for row in csv.reader(f):
                if len(row) < 2:
                    continue
                try:
                    estimates.append((float(row[0]), float(row[1])))
                except ValueError:
                    continue
        return estimates

    def _start_viewer(self):
        from robot.mapping.render import start_viewer

        self._viewer_process = start_viewer(
            self.output_path,
            f"{self.robot_id} drone map overlay",
            f"[{self.robot_id}] Could not start drone map viewer",
        )

    def _stop_viewer(self):
        from robot.mapping.render import stop_viewer

        stop_viewer(self._viewer_process)
