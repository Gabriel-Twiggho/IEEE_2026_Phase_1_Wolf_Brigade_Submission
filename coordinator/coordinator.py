"""Coordinator status/directive messages and simple target assignment."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math

from config import _env_flag, _env_float
from parameters import Coordinator as CoordinatorParams
from coordinator.planning import CoordinatorPlanner
from robot.mapping.drone_map import DroneExtractionMap
from robot.planning.path_planner import PathPlanner
from shared_types import Pose2D


@dataclass(frozen=True)
class CoordinatorDirective:
    """One high-level route request from Coordinator to a robot mission."""

    directive_id: str = ""
    robot_id: str = "robot1"
    kind: str = "NONE"
    target_pose: Pose2D = None
    target_prior_id: str = ""
    target_index: int = 0
    pixel_path: tuple = ()
    waypoints: tuple = ()
    waypoint_speed_caps: tuple = ()
    planning_mode: str = ""
    physical_path_cost_m: float = 0.0
    weighted_path_cost_m: float = 0.0
    reason: str = ""
    created_time_s: float = 0.0
    confidence: float = 0.0
    target_records: dict = None

    def to_dict(self):
        return {
            "directive_id": self.directive_id,
            "robot_id": self.robot_id,
            "kind": self.kind,
            "target_pose": _pose_to_dict(self.target_pose),
            "target_prior_id": self.target_prior_id,
            "target_index": self.target_index,
            "pixel_path": _points_to_list(self.pixel_path),
            "waypoints": _poses_to_list(self.waypoints),
            "waypoint_speed_caps": list(self.waypoint_speed_caps or ()),
            "planning_mode": self.planning_mode,
            "physical_path_cost_m": self.physical_path_cost_m,
            "weighted_path_cost_m": self.weighted_path_cost_m,
            "reason": self.reason,
            "created_time_s": self.created_time_s,
            "confidence": self.confidence,
            "target_records": self.target_records or {},
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            directive_id=str(data.get("directive_id", "")),
            robot_id=str(data.get("robot_id", "robot1")),
            kind=str(data.get("kind", "NONE")),
            target_pose=_pose_from_dict(data.get("target_pose")),
            target_prior_id=str(data.get("target_prior_id", "")),
            target_index=int(data.get("target_index", 0) or 0),
            pixel_path=_points_from_list(data.get("pixel_path")),
            waypoints=_poses_from_list(data.get("waypoints")),
            waypoint_speed_caps=tuple(data.get("waypoint_speed_caps") or ()),
            planning_mode=str(data.get("planning_mode", "")),
            physical_path_cost_m=float(data.get("physical_path_cost_m", 0.0) or 0.0),
            weighted_path_cost_m=float(data.get("weighted_path_cost_m", 0.0) or 0.0),
            reason=str(data.get("reason", "")),
            created_time_s=float(data.get("created_time_s", 0.0) or 0.0),
            confidence=float(data.get("confidence", 0.0) or 0.0),
            target_records=dict(data.get("target_records") or {}),
        )


@dataclass
class CoordinatorStatus:
    """Small robot status message consumed by the minimal coordinator."""

    robot_id: str
    sim_time_s: float = 0.0
    world_pose: Pose2D = None
    route_state: str = "IDLE"
    directive_id: str = ""
    directive_state: str = "IDLE"
    reason: str = ""
    victim_state: str = "IDLE"
    selected_track_id: str = ""
    assigned_prior_id: str = ""
    report_ready: bool = False
    report_confidence: float = 0.0
    robot_operational: bool = True
    mission_active: bool = False
    encountered_prior_id: str = ""

    def to_dict(self):
        return {
            "robot_id": self.robot_id,
            "sim_time_s": self.sim_time_s,
            "world_pose": _pose_to_dict(self.world_pose),
            "route_state": self.route_state,
            "directive_id": self.directive_id,
            "directive_state": self.directive_state,
            "reason": self.reason,
            "victim_state": self.victim_state,
            "selected_track_id": self.selected_track_id,
            "assigned_prior_id": self.assigned_prior_id,
            "report_ready": self.report_ready,
            "report_confidence": self.report_confidence,
            "robot_operational": self.robot_operational,
            "mission_active": self.mission_active,
            "encountered_prior_id": self.encountered_prior_id,
        }

    @classmethod
    def from_dict(cls, data):
        data = data or {}
        return cls(
            robot_id=str(data.get("robot_id", "")),
            sim_time_s=float(data.get("sim_time_s", 0.0) or 0.0),
            world_pose=_pose_from_dict(data.get("world_pose")),
            route_state=str(data.get("route_state", "IDLE")),
            directive_id=str(data.get("directive_id", "")),
            directive_state=str(data.get("directive_state", "IDLE")),
            reason=str(data.get("reason", "")),
            victim_state=str(data.get("victim_state", "IDLE")),
            selected_track_id=str(data.get("selected_track_id", "")),
            assigned_prior_id=str(data.get("assigned_prior_id", "")),
            report_ready=bool(data.get("report_ready", False)),
            report_confidence=float(data.get("report_confidence", 0.0) or 0.0),
            robot_operational=bool(data.get("robot_operational", True)),
            mission_active=bool(data.get("mission_active", False)),
            encountered_prior_id=str(data.get("encountered_prior_id", "")),
        )


def _pose_to_dict(pose):
    if pose is None:
        return None
    return {"x": float(pose.x), "y": float(pose.y), "yaw": float(pose.yaw)}


def _pose_from_dict(data):
    if not data:
        return None
    return Pose2D(
        x=float(data.get("x", 0.0) or 0.0),
        y=float(data.get("y", 0.0) or 0.0),
        yaw=float(data.get("yaw", 0.0) or 0.0),
    )


def _poses_to_list(poses):
    return [
        _pose_to_dict(pose)
        for pose in (poses or ())
        if pose is not None
    ]


def _poses_from_list(items):
    return tuple(
        pose
        for pose in (_pose_from_dict(item) for item in (items or ()))
        if pose is not None
    )


def _points_to_list(points):
    return [[int(x), int(y)] for x, y in (points or ())]


def _points_from_list(items):
    return tuple((int(item[0]), int(item[1])) for item in (items or ()) if len(item) >= 2)


def encode_coordinator_message(sender_id: str, kind: str, payload):
    if hasattr(payload, "to_dict"):
        payload = payload.to_dict()
    return json.dumps(
        {
            "schema": "sar_coord_v1",
            "sender": sender_id,
            "kind": kind,
            "payload": payload,
        }
    ).encode("utf-8")


def decode_coordinator_message(raw):
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", errors="replace")
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return None
    if data.get("schema") != "sar_coord_v1":
        return None
    return data


class CoordinatorClient:
    """Follower-side coordinator port for robots that receive radio directives."""

    def __init__(self, robot_id: str, enabled: bool = True):
        self.robot_id = robot_id
        self.enabled = bool(enabled) and _env_flag(
            "COORDINATOR_ENABLED",
            CoordinatorParams.ENABLED,
        )
        self.pending_directives = []
        self.seen_directive_ids = set()
        self.target_records = {}
        self.control = {
            "collision_hold": False,
            "denied_encounter_prior_id": "",
            "reason": "",
        }

    def receive_message(self, message):
        if not self.enabled or message is None:
            return
        if message.get("sender") == self.robot_id:
            return
        if message.get("kind") == "TARGETS":
            self.update_target_records(message.get("payload"))
            return
        if message.get("kind") == "CONTROL":
            payload = message.get("payload") or {}
            if payload.get("robot_id") == self.robot_id:
                self.control = {
                    "collision_hold": bool(payload.get("collision_hold", False)),
                    "denied_encounter_prior_id": str(
                        payload.get("denied_encounter_prior_id", "")
                    ),
                    "reason": str(payload.get("reason", "")),
                }
            return
        if message.get("kind") != "DIRECTIVE":
            return
        directive = CoordinatorDirective.from_dict(message.get("payload"))
        self.update_target_records(directive.target_records)
        if directive.robot_id != self.robot_id:
            return
        if directive.directive_id in self.seen_directive_ids:
            return
        self.seen_directive_ids.add(directive.directive_id)
        self.pending_directives.append(directive)

    def update_target_records(self, records):
        if isinstance(records, dict):
            self.target_records = records

    def next_directive(self):
        if not self.pending_directives:
            return None
        return self.pending_directives.pop(0)


class CoordinatorLeader:
    """
    Robot1-hosted coordinator V1.

    V1 assigns victim priors, approves reports, and gives the configured
    priority robot right of way when the robots are too close.
    """

    TARGET_STATES = (
        "unassigned",
        "assigned",
        "found",
        "report_ready",
        "reported",
        "route_failed",
        "exhausted",
    )

    def __init__(
        self,
        robot_ids=("robot2", "robot1"),
        enabled: bool = True,
        victims=None,
        drone_map=None,
        path_planner=None,
    ):
        self.enabled = bool(enabled) and _env_flag(
            "COORDINATOR_ENABLED",
            CoordinatorParams.ENABLED,
        )
        self.robot_ids = tuple(robot_ids)
        self.drone_map = drone_map
        self.path_planner = path_planner
        if self.enabled and self.drone_map is None and victims is None:
            self.drone_map = DroneExtractionMap(
                "coordinator",
                enabled_by_default=True,
                render_overlay=True,
            )
        if self.enabled and self.path_planner is None and self.drone_map is not None:
            self.path_planner = PathPlanner(
                "coordinator",
                self.drone_map,
                enabled_by_default=True,
                auto_plan=False,
                render_overlay=True,
            )
        if victims is None:
            victims = (
                self.drone_map.victim_estimates
                if self.drone_map is not None and self.drone_map.ready
                else ()
            )
        self.planner = CoordinatorPlanner(
            robot_ids=self.robot_ids,
            robot_starts=DroneExtractionMap.ROBOT_STARTS,
            victims=victims,
            drone_map=self.drone_map,
            path_planner=self.path_planner,
            cluster_radius_m=CoordinatorParams.CLUSTER_RADIUS_M,
            conservative_shortcuts=CoordinatorParams.CONSERVATIVE_SHORTCUTS,
        )
        self.targets = self._refresh_targets()
        self.status_by_robot = {}
        self.active_directives = {}
        self.initial_live_poses_ready = False
        self.team_collision_enabled = _env_flag(
            "COORDINATOR_TEAM_COLLISION_AVOIDANCE_ENABLED",
            CoordinatorParams.TEAM_COLLISION_AVOIDANCE_ENABLED,
        )
        self.team_hold_distance_m = max(
            0.0,
            _env_float(
                "COORDINATOR_TEAM_COLLISION_HOLD_DISTANCE_M",
                CoordinatorParams.TEAM_COLLISION_HOLD_DISTANCE_M,
            ),
        )
        self.team_release_distance_m = max(
            self.team_hold_distance_m,
            _env_float(
                "COORDINATOR_TEAM_COLLISION_RELEASE_DISTANCE_M",
                CoordinatorParams.TEAM_COLLISION_RELEASE_DISTANCE_M,
            ),
        )
        self.team_priority_robot_id = str(
            CoordinatorParams.TEAM_COLLISION_PRIORITY_ROBOT_ID
        )
        self.yielding_robots = set()
        self.encounter_denials = {}
        self.render_assigned_routes = _env_flag(
            "COORDINATOR_RENDER_ASSIGNED_ROUTES",
            False,
        )
        self.initial_plan_rendered = False
        self.sequence = 0
        self.reason = "waiting for robot status"

    def update_status(self, status: CoordinatorStatus):
        if not self.enabled or not status.robot_id:
            return
        self.status_by_robot[status.robot_id] = status
        self._update_team_collision_yield()
        self._process_terminal_status(status)
        self._process_report_ready(status)
        self._process_encounter(status)

    def tick(self, sim_time_s: float):
        if not self.enabled:
            return
        if not self._prepare_initial_live_poses():
            return
        self._update_team_collision_yield()
        for robot_id in self.robot_ids:
            if robot_id not in self.active_directives:
                self._assign_next_target(robot_id, sim_time_s)

    def next_directive(self, robot_id: str):
        if not self.enabled:
            return None
        directive = self.active_directives.get(robot_id)
        return self._with_target_records(directive, self._refresh_targets())

    def directives_for_broadcast(self, local_robot_id: str):
        if not self.enabled:
            return ()
        targets = self._refresh_targets()
        return tuple(
            self._with_target_records(directive, targets)
            for directive in self.active_directives.values()
            if directive.robot_id != local_robot_id
        )

    def target_board_message(self):
        if not self.enabled:
            return {}
        return self._refresh_targets()

    def should_yield(self, robot_id: str):
        return robot_id in self.yielding_robots

    def control_for(self, robot_id: str):
        denied_prior = self.encounter_denials.get(robot_id, "")
        return {
            "robot_id": robot_id,
            "collision_hold": robot_id in self.yielding_robots,
            "denied_encounter_prior_id": denied_prior,
            "reason": (
                f"yielding to active {self.team_priority_robot_id}"
                if robot_id in self.yielding_robots
                else (
                    f"encountered victim {denied_prior} not reassigned"
                    if denied_prior
                    else ""
                )
            ),
        }

    def controls_for_broadcast(self, local_robot_id: str):
        return tuple(
            self.control_for(robot_id)
            for robot_id in self.robot_ids
            if robot_id != local_robot_id
        )

    def _process_terminal_status(self, status: CoordinatorStatus):
        directive = self.active_directives.get(status.robot_id)
        if directive is None or status.directive_id != directive.directive_id:
            return
        prior_id = directive.target_prior_id

        if directive.kind == "REPORT_VICTIM" and status.directive_state == "DONE":
            self._set_target_state(prior_id, "reported", "")
            print(f"[coordinator] victim reported: {prior_id}")
            self.active_directives.pop(status.robot_id, None)
            return

        if directive.kind == "REPORT_DENIED" and status.directive_state == "DONE":
            self.active_directives.pop(status.robot_id, None)
            self.planner.release_robot(status.robot_id)
            self._refresh_targets()
            return

        if directive.kind == "TARGET_VICTIM":
            if status.directive_state == "DONE":
                self.active_directives.pop(status.robot_id, None)
                self.planner.release_robot(status.robot_id)
                self._refresh_targets()
                return
            if status.directive_state == "FAILED":
                self.planner.record_prior_failure(
                    prior_id,
                    status.robot_id,
                    failure_kind="route_failed",
                )
                self._refresh_targets()
                print(
                    f"[coordinator] route failed: {prior_id}; "
                    "moved to low-priority teammate retry"
                )
                self.active_directives.pop(status.robot_id, None)

    def _process_report_ready(self, status: CoordinatorStatus):
        if not status.report_ready:
            return
        prior_id = status.selected_track_id or status.assigned_prior_id
        active = self.active_directives.get(status.robot_id)
        if active is not None and active.kind in ("REPORT_VICTIM", "REPORT_DENIED"):
            return
        target = self.targets.get(prior_id)
        if target is None:
            self._deny_report_ready(
                status,
                prior_id,
                "report denied: victim prior is not in coordinator target board",
            )
            return
        if target["state"] == "reported":
            self._deny_report_ready(
                status,
                prior_id,
                f"report denied: {prior_id} already reported",
            )
            return
        owner = target.get("assigned_robot", "")
        if target["state"] in ("found", "report_ready") and owner not in ("", status.robot_id):
            self._deny_report_ready(
                status,
                prior_id,
                f"report denied: {prior_id} already found by {owner or 'teammate'}",
            )
            return
        if (
            active is not None
            and active.kind == "TARGET_VICTIM"
            and active.target_prior_id
            and active.target_prior_id != prior_id
        ):
            self._release_superseded_prior(active.target_prior_id, status.robot_id)
        self._set_target_state(prior_id, "found", status.robot_id)
        target = self.targets.get(prior_id, target)
        self._queue_directive(
            CoordinatorDirective(
                directive_id=self._directive_id(status.robot_id, "report", prior_id),
                robot_id=status.robot_id,
                kind="REPORT_VICTIM",
                target_prior_id=prior_id,
                target_index=target["index"],
                confidence=status.report_confidence,
                reason=f"report approved for {prior_id}",
                created_time_s=status.sim_time_s,
            )
        )

    def _process_encounter(self, status: CoordinatorStatus):
        prior_id = status.encountered_prior_id
        if not prior_id:
            self.encounter_denials.pop(status.robot_id, None)
            return
        active = self.active_directives.get(status.robot_id)
        if active is None or active.kind != "TARGET_VICTIM":
            self._deny_encounter(status.robot_id, prior_id)
            return
        if active.target_prior_id == prior_id:
            self.encounter_denials.pop(status.robot_id, None)
            return

        target = self.targets.get(prior_id)
        if target is None or target.get("state") in ("found", "report_ready", "reported"):
            self._deny_encounter(status.robot_id, prior_id)
            return
        if status.world_pose is None:
            self._deny_encounter(status.robot_id, prior_id)
            return

        owner = target.get("assigned_robot", "")
        if owner and owner != status.robot_id:
            owner_status = self.status_by_robot.get(owner)
            if owner_status is None or owner_status.world_pose is None:
                self._deny_encounter(status.robot_id, prior_id)
                return
            if owner_status.robot_operational:
                target_x, target_y = target["world"]
                observer_distance = math.hypot(
                    status.world_pose.x - target_x,
                    status.world_pose.y - target_y,
                )
                owner_distance = math.hypot(
                    owner_status.world_pose.x - target_x,
                    owner_status.world_pose.y - target_y,
                )
                if observer_distance >= owner_distance:
                    self._deny_encounter(status.robot_id, prior_id)
                    return

        choice = self.planner.choice_for_prior(
            status.robot_id,
            status.world_pose,
            prior_id,
        )
        if choice is None:
            self._deny_encounter(status.robot_id, prior_id)
            return

        previous_prior = active.target_prior_id
        if owner and owner != status.robot_id:
            self.active_directives.pop(owner, None)
            self.planner.release_robot(owner)
        self.active_directives.pop(status.robot_id, None)
        self.planner.release_robot(status.robot_id)
        self.planner.update_victim_state(previous_prior, "unassigned", "")
        self.planner.update_victim_state(prior_id, "unassigned", "")
        self.encounter_denials.pop(status.robot_id, None)
        self._refresh_targets()
        self._assign_choice(choice, status.sim_time_s)
        print(
            f"[coordinator] reassigned {status.robot_id} from "
            f"{previous_prior} to encountered victim {prior_id}"
        )

    def _deny_encounter(self, robot_id: str, prior_id: str):
        self.encounter_denials[robot_id] = prior_id

    def _deny_report_ready(
        self,
        status: CoordinatorStatus,
        prior_id: str,
        reason: str,
    ):
        active = self.active_directives.get(status.robot_id)
        if active is not None and active.kind == "TARGET_VICTIM":
            self.planner.release_robot(status.robot_id)
        self._queue_directive(
            CoordinatorDirective(
                directive_id=self._directive_id(
                    status.robot_id,
                    "deny-report",
                    prior_id or "unknown",
                ),
                robot_id=status.robot_id,
                kind="REPORT_DENIED",
                target_prior_id=prior_id,
                reason=reason,
                created_time_s=status.sim_time_s,
            )
        )

    def _assign_next_target(self, robot_id: str, sim_time_s: float):
        status = self.status_by_robot.get(robot_id)
        if status is not None and not status.robot_operational:
            self.planner.unavailable_robots.add(robot_id)
            return
        self.planner.unavailable_robots.discard(robot_id)
        pose = status.world_pose if status is not None else None
        choice = self.planner.choose_next_for_robot(
            robot_id,
            pose,
            busy_robots=self.active_directives.keys(),
            robot_poses={
                known_robot_id: known_status.world_pose
                for known_robot_id, known_status in self.status_by_robot.items()
            },
        )
        if choice is None:
            self._refresh_targets()
            self.reason = "no planned cluster assignment available"
            return
        self._assign_choice(choice, sim_time_s)

    def _prepare_initial_live_poses(self):
        if self.initial_live_poses_ready:
            return True
        robot_poses = {
            robot_id: self.status_by_robot[robot_id].world_pose
            for robot_id in self.robot_ids
            if robot_id in self.status_by_robot
            and self.status_by_robot[robot_id].world_pose is not None
        }
        if len(robot_poses) != len(self.robot_ids):
            self.reason = "waiting for initial robot poses"
            return False
        self.planner.set_initial_robot_poses(robot_poses)
        self.initial_live_poses_ready = True
        self._refresh_targets()
        return True

    def _assign_choice(self, choice, sim_time_s: float):
        target = self.targets.get(choice.prior_id) or self._refresh_targets().get(choice.prior_id)
        if target is None:
            return
        self.planner.assign_choice(choice, choice.robot_id)
        self._refresh_targets()
        planned_path = choice.planned_path
        waypoints = planned_path.waypoints
        speed_caps = planned_path.waypoint_speed_caps
        pixel_path = planned_path.pixel_path
        planning_mode = planned_path.planning_mode
        physical_cost = planned_path.physical_path_cost_m
        weighted_cost = planned_path.weighted_path_cost_m
        self._queue_directive(
            CoordinatorDirective(
                directive_id=self._directive_id(choice.robot_id, "target", choice.prior_id),
                robot_id=choice.robot_id,
                kind="TARGET_VICTIM",
                target_prior_id=choice.prior_id,
                target_index=target["index"],
                target_pose=Pose2D(
                    x=target["world"][0],
                    y=target["world"][1],
                    yaw=0.0,
                ),
                pixel_path=tuple(pixel_path),
                waypoints=tuple(waypoints),
                waypoint_speed_caps=tuple(speed_caps),
                planning_mode=planning_mode,
                physical_path_cost_m=physical_cost,
                weighted_path_cost_m=weighted_cost,
                reason=(
                    f"assigned victim prior {choice.prior_id} "
                    f"in cluster {choice.cluster_id}"
                ),
                created_time_s=sim_time_s,
            )
        )
        self._render_initial_assignment(choice)
        print(f"[coordinator] assigned {choice.robot_id} -> {choice.prior_id} mode={planning_mode}")

    def _render_initial_assignment(self, choice):
        if not self.render_assigned_routes:
            return
        self.initial_plan_rendered = True
        self.planner.render_choice(choice)

    def _set_target_state(self, prior_id: str, state: str, robot_id: str):
        if prior_id not in self.targets:
            return
        if state == "reported":
            assigned_robot = self.targets[prior_id].get("assigned_robot", "")
            self.planner.update_victim_state(prior_id, state, assigned_robot)
            cluster_id = self.planner.cluster_id_for_prior(prior_id)
            if self.planner.cluster_complete(cluster_id):
                self.planner.release_robot(assigned_robot)
        else:
            self.planner.update_victim_state(prior_id, state, robot_id)
        self._refresh_targets()

    def _release_superseded_prior(self, prior_id: str, robot_id: str):
        target = self.targets.get(prior_id)
        if target is None:
            return
        if target.get("state") in ("reported", "found"):
            return
        if target.get("assigned_robot") not in ("", robot_id):
            return
        self.planner.release_robot(robot_id)
        self.planner.update_victim_state(prior_id, "unassigned", "")
        self._refresh_targets()

    def _update_team_collision_yield(self):
        if not self.team_collision_enabled:
            self.yielding_robots.clear()
            return
        priority_status = self.status_by_robot.get(self.team_priority_robot_id)
        if (
            priority_status is None
            or priority_status.world_pose is None
            or not priority_status.robot_operational
            or not priority_status.mission_active
        ):
            self.yielding_robots.clear()
            return
        yield_robot_id = next(
            (
                robot_id
                for robot_id in self.robot_ids
                if robot_id != self.team_priority_robot_id
            ),
            None,
        )
        yield_status = self.status_by_robot.get(yield_robot_id)
        if (
            yield_robot_id is None
            or yield_status is None
            or yield_status.world_pose is None
            or not yield_status.robot_operational
        ):
            self.yielding_robots.clear()
            return

        distance_m = math.hypot(
            priority_status.world_pose.x - yield_status.world_pose.x,
            priority_status.world_pose.y - yield_status.world_pose.y,
        )
        if yield_robot_id in self.yielding_robots:
            if distance_m >= self.team_release_distance_m:
                self.yielding_robots.discard(yield_robot_id)
        elif distance_m < self.team_hold_distance_m:
            self.yielding_robots.add(yield_robot_id)

    def _queue_directive(self, directive: CoordinatorDirective):
        self.active_directives[directive.robot_id] = self._with_target_records(
            directive,
            self._refresh_targets(),
        )
        self.reason = directive.reason

    def _directive_id(self, robot_id: str, action: str, prior_id: str):
        self.sequence += 1
        return f"coord-v1-{self.sequence:03d}-{robot_id}-{action}-{prior_id}"

    def _refresh_targets(self):
        self.targets = self.planner.target_records()
        return self.targets

    def _with_target_records(self, directive, targets=None):
        if directive is None:
            return None
        return replace(directive, target_records=targets or self._refresh_targets())
