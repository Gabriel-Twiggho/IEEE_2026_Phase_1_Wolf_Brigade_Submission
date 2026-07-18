"""Passive local robot state container for the flat architecture."""

from __future__ import annotations

from dataclasses import dataclass, field

from shared_types import (
    MapSummary,
    Pose2D,
    RouteSummary,
    SafetySummary,
    TerrainAssessment,
    VictimSummary,
)


@dataclass
class RobotState:
    """
    One robot's current state.

    This class stores facts only. It does not choose missions, plan routes,
    filter safety, or command motors.
    """

    robot_id: str
    sim_time_s: float = 0.0
    odom_pose: Pose2D = field(default_factory=Pose2D)
    world_pose: Pose2D = field(default_factory=Pose2D)
    terrain: TerrainAssessment = field(default_factory=TerrainAssessment)
    map: MapSummary = field(default_factory=MapSummary)
    victim: VictimSummary = field(default_factory=VictimSummary)
    route: RouteSummary = field(default_factory=RouteSummary)
    safety: SafetySummary = field(default_factory=SafetySummary)
    recovery: SafetySummary = field(default_factory=lambda: SafetySummary(state="IDLE"))
    latest_directive: object = None

    def update_pose(
        self,
        sim_time_s: float,
        odom_pose: Pose2D,
        world_pose: Pose2D,
        terrain: TerrainAssessment = None,
    ):
        self.sim_time_s = float(sim_time_s)
        self.odom_pose = odom_pose
        self.world_pose = world_pose
        if terrain is not None:
            self.terrain = terrain

    def update_map_summary(
        self,
        ready: bool,
        revision: int,
        occupied_count: int,
        victim_count: int = 0,
        error: str = "",
    ):
        self.map = MapSummary(
            ready=bool(ready),
            revision=int(revision),
            occupied_count=int(occupied_count),
            victim_count=int(victim_count),
            error=str(error or ""),
        )

    def update_victim_summary(
        self,
        state: str,
        selected_track_id: str = "",
        reason: str = "",
        found_prior_ids=(),
        exhausted_prior_ids=(),
        report_ready: bool = False,
        confidence: float = 0.0,
    ):
        self.victim = VictimSummary(
            state=state,
            selected_track_id=selected_track_id,
            reason=reason,
            found_prior_ids=tuple(found_prior_ids),
            exhausted_prior_ids=tuple(exhausted_prior_ids),
            report_ready=bool(report_ready),
            confidence=float(confidence),
        )

    def update_victim_summary_from_mission(self, mission):
        snapshot = mission.snapshot() if hasattr(mission, "snapshot") else {}
        found = []
        exhausted = []
        tracker = getattr(mission, "tracker", None)
        for track in getattr(tracker, "tracks", ()):
            if getattr(track, "reported", False) or getattr(track, "status", "") == "FOUND":
                found.append(track.track_id)
            elif getattr(track, "status", "") == "SEARCH_EXHAUSTED":
                exhausted.append(track.track_id)
        self.update_victim_summary(
            state=snapshot.get("state", getattr(mission, "state", "IDLE")),
            selected_track_id=snapshot.get(
                "selected_track_id",
                getattr(mission, "selected_track_id", ""),
            ),
            reason=snapshot.get("reason", getattr(mission, "reason", "")),
            found_prior_ids=found,
            exhausted_prior_ids=exhausted,
            report_ready=snapshot.get("report_status", "") == "ready",
            confidence=snapshot.get("report_confidence", 0.0),
        )

    def update_route_summary(
        self,
        active: bool,
        state: str,
        waypoint_index: int = 0,
        waypoint_count: int = 0,
        route_id: str = "",
        target_id: str = "",
    ):
        self.route = RouteSummary(
            active=bool(active),
            route_id=str(route_id or ""),
            target_id=str(target_id or ""),
            state=str(state or "IDLE"),
            waypoint_index=int(waypoint_index),
            waypoint_count=int(waypoint_count),
        )

    def update_safety_summary(self, safety, recovery=None):
        self.safety = SafetySummary(
            state=getattr(safety, "state", "CLEAR"),
            source=getattr(safety, "source", ""),
            reason=getattr(safety, "reason", ""),
        )
        if recovery is not None:
            self.recovery = SafetySummary(
                state=getattr(recovery, "state", "IDLE"),
                source=getattr(recovery, "trigger_source", ""),
                reason=getattr(recovery, "reason", ""),
            )

    def snapshot(self):
        return {
            "robot_id": self.robot_id,
            "sim_time_s": self.sim_time_s,
            "odom": {
                "x": self.odom_pose.x,
                "y": self.odom_pose.y,
                "yaw": self.odom_pose.yaw,
            },
            "world": {
                "x": self.world_pose.x,
                "y": self.world_pose.y,
                "yaw": self.world_pose.yaw,
            },
            "terrain": {
                "state": self.terrain.state,
                "pitch_rad": self.terrain.pitch_rad,
                "roll_rad": self.terrain.roll_rad,
                "tilt_rad": self.terrain.tilt_rad,
                "reliable": self.terrain.reliable,
            },
            "map": {
                "ready": self.map.ready,
                "revision": self.map.revision,
                "occupied_count": self.map.occupied_count,
                "victim_count": self.map.victim_count,
                "error": self.map.error,
            },
            "victim": {
                "state": self.victim.state,
                "selected_track_id": self.victim.selected_track_id,
                "reason": self.victim.reason,
                "found_prior_ids": self.victim.found_prior_ids,
                "exhausted_prior_ids": self.victim.exhausted_prior_ids,
                "report_ready": self.victim.report_ready,
                "confidence": self.victim.confidence,
            },
            "route": {
                "active": self.route.active,
                "route_id": self.route.route_id,
                "target_id": self.route.target_id,
                "state": self.route.state,
                "waypoint_index": self.route.waypoint_index,
                "waypoint_count": self.route.waypoint_count,
            },
            "safety": {
                "state": self.safety.state,
                "source": self.safety.source,
                "reason": self.safety.reason,
            },
            "recovery": {
                "state": self.recovery.state,
                "source": self.recovery.source,
                "reason": self.recovery.reason,
            },
            "directive": self._directive_snapshot(),
        }

    def _directive_snapshot(self):
        directive = self.latest_directive
        if directive is None:
            return {
                "id": "",
                "robot_id": "",
                "kind": "NONE",
                "target_prior_id": "",
                "target_index": 0,
                "reason": "",
            }
        return {
            "id": getattr(directive, "directive_id", ""),
            "robot_id": getattr(directive, "robot_id", ""),
            "kind": getattr(directive, "kind", "NONE"),
            "target_prior_id": getattr(directive, "target_prior_id", ""),
            "target_index": getattr(directive, "target_index", 0),
            "reason": getattr(directive, "reason", ""),
        }
