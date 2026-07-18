"""Coordinator-level clustering, assignment, and route planning.

This module keeps team planning policy out of coordinator.coordinator.  The
coordinator still owns radio/status/directive bookkeeping; this file owns the
question "which robot should go to which victim next, and by what route?".
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import heapq
from itertools import permutations
import math
import os
from pathlib import Path

from config import SIM_LOGS_DIR, resolve_runtime_path, _env_float, _env_flag
from parameters import Coordinator as CoordinatorParams
from shared_types import PlannedPath, Pose2D


DRONE_PATH_GUIDED = "DRONE_PATH_GUIDED"
DRONE_CORRIDOR_ASTAR = "DRONE_CORRIDOR_ASTAR"
MAP_ASTAR = "MAP_ASTAR"
FAILED = "FAILED"
TERMINAL_VICTIM_STATES = frozenset(("reported",))
FOUND_VICTIM_STATES = frozenset(("found", "report_ready"))
DEFERRED_VICTIM_STATES = frozenset(("route_failed", "exhausted"))
HANDLED_VICTIM_STATES = TERMINAL_VICTIM_STATES | FOUND_VICTIM_STATES
PRIMARY_BLOCKED_STATES = HANDLED_VICTIM_STATES | DEFERRED_VICTIM_STATES | frozenset(("assigned",))
GLOBAL_BLOCKED_STATES = HANDLED_VICTIM_STATES | DEFERRED_VICTIM_STATES | frozenset(("assigned",))


@dataclass(frozen=True)
class VictimPrior:
    prior_id: str
    index: int
    world: tuple


@dataclass
class VictimCluster:
    cluster_id: str
    prior_ids: list
    victim_indices: list
    victim_world_positions: list
    centre: tuple
    nearest_drone_path_index: int = -1
    state: str = "unassigned"
    assigned_robot: str = ""


@dataclass(frozen=True)
class AssignmentChoice:
    robot_id: str
    cluster_id: str
    prior_id: str
    planned_path: PlannedPath


def load_drone_path_csv(path) -> tuple:
    """Load world/metre path points from a drone-path CSV.

    Supports the generated header format with x_m/y_m columns and simple
    headerless x,y rows used by tests or hand-written fixtures.
    """

    if path in (None, ""):
        return ()
    path = Path(path)
    if not path.exists():
        return ()

    points = []
    with path.open("r", encoding="utf-8", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(f)
            for row in reader:
                point = _point_from_dict_row(row)
                if point is not None:
                    points.append(point)
        else:
            reader = csv.reader(f)
            for row in reader:
                point = _point_from_numeric_row(row)
                if point is not None:
                    points.append(point)

    return _deduplicate_points(points)


def _point_from_dict_row(row):
    for x_key, y_key in (("x_m", "y_m"), ("x", "y"), ("world_x", "world_y")):
        if x_key in row and y_key in row:
            try:
                return float(row[x_key]), float(row[y_key])
            except (TypeError, ValueError):
                return None
    return _point_from_numeric_row(list(row.values()))


def _point_from_numeric_row(row):
    try:
        values = [float(item) for item in row if str(item).strip() != ""]
    except (TypeError, ValueError):
        return None
    if len(values) >= 5:
        return values[3], values[4]
    if len(values) >= 2:
        return values[0], values[1]
    return None


def build_victim_clusters(victims, radius_m: float = 3.0, drone_path=()) -> dict:
    priors = [
        VictimPrior(f"P{i}", i, (float(victim[0]), float(victim[1])))
        for i, victim in enumerate(victims or (), start=1)
    ]
    clusters = []
    for prior in priors:
        matching = [
            index
            for index, cluster in enumerate(clusters)
            if _prior_touches_cluster(prior, cluster, radius_m)
        ]
        if not matching:
            clusters.append([prior])
            continue
        first = matching[0]
        clusters[first].append(prior)
        for index in reversed(matching[1:]):
            clusters[first].extend(clusters.pop(index))
        _merge_touching_clusters(clusters, radius_m)

    result = {}
    for index, members in enumerate(clusters, start=1):
        members = sorted(members, key=lambda prior: prior.index)
        positions = [prior.world for prior in members]
        centre = _centroid(positions)
        nearest = _nearest_path_index(centre, drone_path)
        cluster = VictimCluster(
            cluster_id=f"C{index}",
            prior_ids=[prior.prior_id for prior in members],
            victim_indices=[prior.index for prior in members],
            victim_world_positions=positions,
            centre=centre,
            nearest_drone_path_index=nearest,
        )
        result[cluster.cluster_id] = cluster
    return result


def _prior_touches_cluster(prior, cluster_members, radius_m):
    if _distance(prior.world, _centroid([item.world for item in cluster_members])) <= radius_m:
        return True
    return any(_distance(prior.world, item.world) <= radius_m for item in cluster_members)


def _merge_touching_clusters(clusters, radius_m):
    changed = True
    while changed:
        changed = False
        for i in range(len(clusters)):
            if changed:
                break
            for j in range(i + 1, len(clusters)):
                if _clusters_touch(clusters[i], clusters[j], radius_m):
                    clusters[i].extend(clusters.pop(j))
                    changed = True
                    break


def _clusters_touch(a, b, radius_m):
    centre_a = _centroid([item.world for item in a])
    centre_b = _centroid([item.world for item in b])
    if _distance(centre_a, centre_b) <= radius_m:
        return True
    return any(_distance(left.world, right.world) <= radius_m for left in a for right in b)


class CoordinatorPlanner:
    """Cluster-aware, drone-path-guided planner for coordinator decisions."""

    def __init__(
        self,
        robot_ids=("robot2", "robot1"),
        robot_starts=None,
        victims=None,
        drone_map=None,
        path_planner=None,
        drone_path=None,
        drone_path_path=None,
        cluster_radius_m: float = 3.0,
        conservative_shortcuts: bool = True,
    ):
        self.robot_ids = tuple(robot_ids)
        self.robot_starts = dict(robot_starts or {})
        self.drone_map = drone_map
        self.path_planner = path_planner
        self.cluster_radius_m = float(cluster_radius_m)
        self.conservative_shortcuts = bool(conservative_shortcuts)
        self.drone_corridor_enabled = _env_flag(
            "COORDINATOR_DRONE_CORRIDOR_ENABLED",
            CoordinatorParams.DRONE_CORRIDOR_ENABLED,
        )
        self.drone_corridor_radius_m = max(
            0.0,
            _env_float(
                "COORDINATOR_DRONE_CORRIDOR_RADIUS_M",
                CoordinatorParams.DRONE_CORRIDOR_RADIUS_M,
            ),
        )
        self.drone_corridor_cost_multiplier = min(
            1.0,
            max(
                0.05,
                _env_float(
                    "COORDINATOR_DRONE_CORRIDOR_COST_MULTIPLIER",
                    CoordinatorParams.DRONE_CORRIDOR_COST_MULTIPLIER,
                ),
            ),
        )
        self.drone_path = tuple(drone_path or ())
        if not self.drone_path:
            self.drone_path = load_drone_path_csv(
                resolve_runtime_path(
                    drone_path_path
                    or os.environ.get(
                        "DRONE_PATH_CSV",
                        str(SIM_LOGS_DIR / "drone_path.csv"),
                    )
                )
            )
        self.victims = tuple((float(v[0]), float(v[1])) for v in (victims or ()))
        self.priors = {
            f"P{i}": VictimPrior(f"P{i}", i, victim)
            for i, victim in enumerate(self.victims, start=1)
        }
        self.victim_states = {prior_id: "unassigned" for prior_id in self.priors}
        self.failed_prior_robots = {prior_id: set() for prior_id in self.priors}
        self.unavailable_robots = set()
        self.clusters = build_victim_clusters(
            self.victims,
            self.cluster_radius_m,
            self.drone_path,
        )
        self.prior_cluster_ids = {
            prior_id: cluster_id
            for cluster_id, cluster in self.clusters.items()
            for prior_id in cluster.prior_ids
        }
        self.robot_cluster = {}
        self.robot_queues = {robot_id: [] for robot_id in self.robot_ids}
        self.planned_prior_owner = {}
        self.prior_assigned_robot = {prior_id: "" for prior_id in self.priors}
        self.deferred_prior_ids = []
        self.robot_prior_attempts = {}
        self.route_cache = {}
        self._drone_corridor_cache = {}
        if self.drone_path:
            print(
                "[coordinator-planner] loaded drone path backbone: "
                f"{len(self.drone_path)} point(s)"
            )
        else:
            print("[coordinator-planner] no drone path; map fallback will be used")

    def target_records(self):
        return {
            prior_id: {
                "index": prior.index,
                "world": tuple(prior.world),
                "state": self.victim_states.get(prior_id, "unassigned"),
                "assigned_robot": self._assigned_robot_for_prior(prior_id),
                "cluster_id": self.cluster_id_for_prior(prior_id),
                "failed_by": tuple(sorted(self.failed_prior_robots.get(prior_id, ()))),
                "deferred": prior_id in self.deferred_prior_ids,
            }
            for prior_id, prior in self.priors.items()
        }

    def update_victim_state(self, prior_id: str, state: str, robot_id: str = ""):
        if prior_id not in self.victim_states:
            return
        self.victim_states[prior_id] = state
        if state in ("assigned", "found", "report_ready"):
            self.prior_assigned_robot[prior_id] = robot_id
        elif state in ("unassigned", "reported", "route_failed", "exhausted"):
            self.prior_assigned_robot[prior_id] = ""
        if state in HANDLED_VICTIM_STATES:
            self._remove_deferred_prior(prior_id)
        cluster_id = self.cluster_id_for_prior(prior_id)
        if cluster_id:
            self._refresh_cluster_state(cluster_id, robot_id)

    def mark_route_failed(self, prior_id: str, robot_id: str):
        self.record_prior_failure(
            prior_id,
            robot_id,
            failure_kind="route_failed",
        )

    def defer_prior(self, prior_id: str, robot_id: str, state: str = "exhausted"):
        self.record_prior_failure(
            prior_id,
            robot_id,
            failure_kind=state,
        )

    def record_prior_failure(
        self,
        prior_id: str,
        robot_id: str,
        failure_kind: str = "route_failed",
    ):
        if prior_id not in self.victim_states:
            return
        if self.victim_states.get(prior_id) in HANDLED_VICTIM_STATES:
            return
        if robot_id:
            self.failed_prior_robots.setdefault(prior_id, set()).add(robot_id)

        self.victim_states[prior_id] = failure_kind
        self.release_robot(robot_id)
        attempts = self._bump_attempt(robot_id, prior_id)
        self._defer_prior(prior_id, failure_kind)
        print(
            "[coordinator-planner] "
            f"{prior_id} moved to low-priority teammate retry after {attempts} "
            f"failed attempt(s) on {robot_id}"
        )

    def _defer_prior(self, prior_id: str, state: str):
        self.victim_states[prior_id] = state
        if prior_id not in self.deferred_prior_ids:
            self.deferred_prior_ids.append(prior_id)
        cluster_id = self.cluster_id_for_prior(prior_id)
        if cluster_id:
            self._refresh_cluster_state(cluster_id, "")

    def release_robot(self, robot_id: str):
        self.robot_cluster.pop(robot_id, None)
        for cluster in self.clusters.values():
            if cluster.assigned_robot == robot_id:
                for prior_id in cluster.prior_ids:
                    if self.victim_states.get(prior_id) == "assigned":
                        self.victim_states[prior_id] = "unassigned"
                        self.prior_assigned_robot[prior_id] = ""
                cluster.assigned_robot = ""
                self._refresh_cluster_state(cluster.cluster_id, "")

    def choose_next_for_robot(self, robot_id: str, robot_pose=None, busy_robots=(), robot_poses=None):
        busy_robots = set(busy_robots)
        if robot_id in busy_robots or robot_id in self.unavailable_robots:
            return None

        queued = self._next_choice_from_queue(robot_id, robot_pose)
        if queued is not None:
            return queued

        normal = self._next_global_remaining_choice(
            robot_id,
            robot_pose,
            busy_robots,
        )
        if normal is not None:
            return normal

        return self._next_deferred_teammate_choice(
            robot_id,
            robot_pose,
            busy_robots,
            robot_poses,
        )

    def set_initial_robot_poses(self, robot_poses):
        """Rebuild the initial victim queues from first reported robot poses."""
        self.robot_starts.update(
            {
                robot_id: pose
                for robot_id, pose in (robot_poses or {}).items()
                if robot_id in self.robot_ids and pose is not None
            }
        )
        self.route_cache.clear()
        self._build_global_queues()

    def choice_for_prior(self, robot_id: str, robot_pose, prior_id: str):
        """Plan one explicit coordinator assignment without mutating ownership."""
        return self._choice_for_prior(robot_id, robot_pose, prior_id)

    def plan_for_prior(
        self,
        robot_id: str,
        start_pose,
        prior: VictimPrior,
        route_policy: str = "default",
    ):
        cache_key = _cache_key(robot_id, start_pose, prior.prior_id, route_policy)
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        planned = self._plan_route(robot_id, start_pose, prior, route_policy)
        if planned.success:
            print(
                f"[coordinator-planner] {planned.planning_mode}: "
                f"{robot_id} -> {prior.prior_id} "
                f"policy={route_policy} "
                f"physical={planned.physical_path_cost_m:.2f}m "
                f"weighted={planned.weighted_path_cost_m:.2f}m"
            )
        self.route_cache[cache_key] = planned
        return planned

    def assign_choice(self, choice: AssignmentChoice, robot_id: str):
        self._remove_deferred_prior(choice.prior_id)
        cluster = self.clusters.get(choice.cluster_id)
        if cluster is not None:
            cluster.assigned_robot = robot_id
            cluster.state = "assigned"
        self.robot_cluster[robot_id] = choice.cluster_id
        self.victim_states[choice.prior_id] = "assigned"
        self.prior_assigned_robot[choice.prior_id] = robot_id

    def cluster_id_for_prior(self, prior_id: str):
        return self.prior_cluster_ids.get(prior_id, "")

    def cluster_complete(self, cluster_id: str):
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            return True
        return all(
            self.victim_states.get(prior_id, "unassigned") == "reported"
            for prior_id in cluster.prior_ids
        )

    def render_choice(self, choice: AssignmentChoice):
        if (
            self.path_planner is not None
            and getattr(self.path_planner, "ready", False)
            and hasattr(self.path_planner, "render_plan_overlay")
        ):
            self.path_planner.render_plan_overlay(choice.planned_path)

    def _build_global_queues(self):
        prior_ids = sorted(
            self.priors,
            key=lambda prior_id: self.priors[prior_id].index,
        )
        if len(self.robot_ids) != 2 or not prior_ids:
            if len(self.robot_ids) != 2:
                print(
                    "[coordinator-planner] expected exactly two robots for "
                    "initial victim assignment"
                )
            return

        assignment = self._balanced_route_cost_assignment(prior_ids)

        self.robot_queues = {}
        self.planned_prior_owner = {}
        for robot_id in self.robot_ids:
            assigned = list(assignment.get(robot_id, ()))
            self.robot_queues[robot_id] = assigned
            for prior_id in assigned:
                self.planned_prior_owner[prior_id] = robot_id

        summary = ", ".join(
            f"{robot_id}:{'->'.join(self.robot_queues.get(robot_id, ())) or '-'}"
            for robot_id in self.robot_ids
        )
        print(f"[coordinator-planner] global victim queues: {summary}")

    def _balanced_route_cost_assignment(self, prior_ids):
        """Choose complete robot queues that minimise team completion distance."""

        start_costs = {
            (robot_id, prior_id): self._assignment_leg_cost(
                robot_id,
                self.robot_starts.get(robot_id),
                prior_id,
            )
            for robot_id in self.robot_ids
            for prior_id in prior_ids
        }
        pair_costs = {}
        route_robot_id = self.robot_ids[0]
        for source_id in prior_ids:
            source = self.priors[source_id].world
            source_pose = Pose2D(source[0], source[1], 0.0)
            for target_id in prior_ids:
                if source_id == target_id:
                    continue
                pair_costs[(source_id, target_id)] = self._assignment_leg_cost(
                    route_robot_id,
                    source_pose,
                    target_id,
                )
        return self._optimise_balanced_routes(prior_ids, start_costs, pair_costs)

    def _assignment_leg_cost(self, robot_id, start_pose, prior_id):
        prior = self.priors[prior_id]
        if self.path_planner is not None and getattr(self.path_planner, "ready", False):
            planned = self.plan_for_prior(robot_id, start_pose, prior)
            if not planned.success:
                return float("inf")
            return planned.physical_path_cost_m or planned.weighted_path_cost_m
        start_xy = _pose_xy(start_pose)
        if start_xy is None:
            return float("inf")
        return _distance(start_xy, prior.world)

    def _optimise_balanced_routes(self, prior_ids, start_costs, pair_costs):
        robot_a, robot_b = self.robot_ids[:2]
        best_score = None
        best_assignment = None

        def route_cost(robot_id, route):
            if not route:
                return 0.0
            total = start_costs.get((robot_id, route[0]), float("inf"))
            for source_id, target_id in zip(route, route[1:]):
                total += pair_costs.get((source_id, target_id), float("inf"))
            return total

        for owner_mask in range(1 << len(prior_ids)):
            assigned_a = tuple(
                prior_id
                for index, prior_id in enumerate(prior_ids)
                if owner_mask & (1 << index)
            )
            assigned_b = tuple(
                prior_id for prior_id in prior_ids if prior_id not in assigned_a
            )
            if len(prior_ids) > 1 and (not assigned_a or not assigned_b):
                continue
            for route_a in permutations(assigned_a):
                cost_a = route_cost(robot_a, route_a)
                if not math.isfinite(cost_a):
                    continue
                for route_b in permutations(assigned_b):
                    cost_b = route_cost(robot_b, route_b)
                    if not math.isfinite(cost_b):
                        continue
                    score = (
                        max(cost_a, cost_b),
                        cost_a + cost_b,
                        abs(cost_a - cost_b),
                        -len(route_a),
                        route_a,
                        route_b,
                    )
                    if best_score is None or score < best_score:
                        best_score = score
                        best_assignment = {
                            robot_a: list(route_a),
                            robot_b: list(route_b),
                        }

        if best_assignment is None:
            print("[coordinator-planner] no complete balanced route assignment found")
            return {robot_id: [] for robot_id in self.robot_ids}
        print(
            "[coordinator-planner] balanced route assignment "
            f"completion={best_score[0]:.2f}m total={best_score[1]:.2f}m"
        )
        return best_assignment

    def _attempt_count(self, robot_id: str, prior_id: str) -> int:
        return self.robot_prior_attempts.get((robot_id, prior_id), 0)

    def _bump_attempt(self, robot_id: str, prior_id: str) -> int:
        key = (robot_id, prior_id)
        self.robot_prior_attempts[key] = self._attempt_count(robot_id, prior_id) + 1
        return self.robot_prior_attempts[key]

    def _choice_for_prior(
        self,
        robot_id: str,
        start_pose,
        prior_id: str,
        route_policy: str = "default",
    ):
        prior = self.priors.get(prior_id)
        if prior is None:
            return None
        if robot_id in self.failed_prior_robots.get(prior_id, set()):
            return None
        cluster_id = self.cluster_id_for_prior(prior_id)
        planned_path = self.plan_for_prior(
            robot_id,
            start_pose,
            prior,
            route_policy=route_policy,
        )
        if not planned_path.success:
            return None
        return AssignmentChoice(robot_id, cluster_id, prior_id, planned_path)

    def _plan_route(self, robot_id: str, start_pose, prior: VictimPrior, route_policy: str):
        if route_policy == "guided":
            planned = self._plan_drone_path_guided(robot_id, start_pose, prior)
            if planned.success:
                return planned
            route_policy = "default"
        return self._plan_map_astar(robot_id, start_pose, prior, allow_corridor=True)

    def _next_choice_from_queue(self, robot_id: str, robot_pose):
        queue = self.robot_queues.setdefault(robot_id, [])
        while queue:
            prior_id = queue[0]
            if prior_id not in self.priors:
                queue.pop(0)
                continue
            state = self.victim_states.get(prior_id, "unassigned")
            if state in PRIMARY_BLOCKED_STATES:
                queue.pop(0)
                continue
            choice = self._choice_for_prior(
                robot_id,
                robot_pose,
                prior_id,
            )
            if choice is not None:
                return choice
            self.defer_prior(prior_id, robot_id, "route_failed")
            queue.pop(0)
        return None

    def _next_deferred_teammate_choice(self, robot_id: str, robot_pose, busy_robots=(), robot_poses=None):
        candidates = self._deferred_candidate_prior_ids()
        if not candidates:
            return None
        robot_poses = robot_poses or {}
        available_robots = [
            candidate_robot
            for candidate_robot in self.robot_ids
            if candidate_robot not in busy_robots
            and candidate_robot not in self.unavailable_robots
        ]
        if robot_id not in available_robots:
            return None

        current_point = _pose_xy(robot_pose) or self._robot_start_xy(robot_id)
        ordered = sorted(
            candidates,
            key=lambda prior_id: (
                _distance(current_point or self.priors[prior_id].world, self.priors[prior_id].world),
                self.priors[prior_id].index,
            ),
        )
        for prior_id in ordered:
            closest_robot = self._closest_robot_for_prior(
                prior_id,
                available_robots,
                robot_poses,
            )
            if closest_robot != robot_id:
                continue
            choice = self._choice_for_prior(
                robot_id,
                robot_pose,
                prior_id,
                route_policy="default",
            )
            if choice is not None:
                return choice
        return None

    def _closest_robot_for_prior(self, prior_id, robot_ids, robot_poses=None):
        robot_poses = robot_poses or {}
        prior = self.priors[prior_id]
        eligible_robot_ids = tuple(
            robot_id
            for robot_id in robot_ids
            if robot_id not in self.failed_prior_robots.get(prior_id, set())
        )
        if not eligible_robot_ids:
            return None
        return min(
            eligible_robot_ids,
            key=lambda robot_id: (
                _distance(
                    _pose_xy(robot_poses.get(robot_id))
                    or self._robot_start_xy(robot_id)
                    or prior.world,
                    prior.world,
                ),
                robot_id,
            ),
        )

    def _next_global_remaining_choice(self, robot_id: str, robot_pose, busy_robots=()):
        if robot_id in busy_robots:
            return None
        candidates = []
        start_point = _pose_xy(robot_pose) or self._robot_start_xy(robot_id)
        for prior_id, prior in self.priors.items():
            state = self.victim_states.get(prior_id, "unassigned")
            if state in GLOBAL_BLOCKED_STATES:
                continue
            planned_owner = self.planned_prior_owner.get(prior_id, "")
            if (
                planned_owner
                and planned_owner != robot_id
                and planned_owner not in self.unavailable_robots
            ):
                continue
            choice = self._choice_for_prior(robot_id, robot_pose, prior_id)
            if choice is None:
                continue
            candidates.append(
                (
                    _route_score(choice.planned_path),
                    _distance(start_point or prior.world, prior.world),
                    prior.index,
                    choice,
                )
            )
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[:3])[3]

    def _primary_prior_ids(self, cluster_id: str):
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            return ()
        return tuple(
            prior_id
            for prior_id in cluster.prior_ids
            if self.victim_states.get(prior_id, "unassigned") not in PRIMARY_BLOCKED_STATES
        )

    def _deferred_candidate_prior_ids(self):
        return tuple(
            prior_id
            for prior_id in self.deferred_prior_ids
            if prior_id in self.priors
            and self.victim_states.get(prior_id, "unassigned") in DEFERRED_VICTIM_STATES
        )

    def _remove_deferred_prior(self, prior_id):
        if prior_id in self.deferred_prior_ids:
            self.deferred_prior_ids = [
                candidate
                for candidate in self.deferred_prior_ids
                if candidate != prior_id
            ]

    def _robot_start_xy(self, robot_id):
        return _pose_xy(self.robot_starts.get(robot_id))

    def _plan_drone_path_guided(self, robot_id: str, start_pose, prior: VictimPrior):
        if not self.drone_path:
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="no drone path loaded",
            )
        start_point = _pose_xy(start_pose) or prior.world
        start_index = _nearest_path_index(start_point, self.drone_path)
        target_index = _nearest_path_index(prior.world, self.drone_path)
        if start_index < 0 or target_index < 0:
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="could not connect to drone path",
            )

        if start_index <= target_index:
            backbone = list(self.drone_path[start_index:target_index + 1])
        else:
            backbone = list(reversed(self.drone_path[target_index:start_index + 1]))
        points = _deduplicate_points([start_point] + backbone + [prior.world])
        points = self._resample_route(points)
        if self.conservative_shortcuts:
            points = self._shortcut_if_verified(points)
        points = self._resample_route(points)
        if len(points) < 2:
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="guided route collapsed",
            )

        waypoints = self._waypoints_from_points(points)
        if not waypoints:
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="guided route has no waypoints",
            )
        pixel_path = self._pixel_path(points)
        physical_cost = _polyline_length(points)
        connection_cost = _distance(start_point, self.drone_path[start_index]) + _distance(
            self.drone_path[target_index],
            prior.world,
        )
        weighted_cost = physical_cost + 0.25 * connection_cost
        return PlannedPath(
            robot_id=robot_id,
            victim_index=prior.index,
            victim_world=prior.world,
            pixel_path=pixel_path,
            world_path=tuple(points),
            waypoints=tuple(waypoints),
            waypoint_speed_caps=tuple(None for _ in waypoints),
            planning_mode=DRONE_PATH_GUIDED,
            physical_path_cost_m=physical_cost,
            weighted_path_cost_m=weighted_cost,
            success=True,
        )

    def _plan_map_astar(self, robot_id: str, start_pose, prior: VictimPrior, allow_corridor: bool = True):
        if self.path_planner is None or not getattr(self.path_planner, "ready", False):
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="coordinator planner unavailable",
            )
        pose = start_pose or self.robot_starts.get(robot_id)
        if pose is None:
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="robot start pose missing",
            )
        planned_path = self.path_planner.plan_to_fixed_victim(
            robot_id,
            pose,
            prior.index,
            prior.world,
            log_result=False,
        )
        if planned_path.success:
            if planned_path.planning_mode == "GLOBAL":
                planned_path.planning_mode = MAP_ASTAR
        if allow_corridor:
            corridor_path = self._plan_drone_corridor_astar(
                robot_id,
                pose,
                prior,
            )
            if corridor_path is not None and corridor_path.success:
                if planned_path.success:
                    corridor_path.global_comparison_pixel_path = planned_path.pixel_path
                return corridor_path
            return PlannedPath(
                robot_id=robot_id,
                victim_index=prior.index,
                victim_world=prior.world,
                success=False,
                error_reason="drone corridor astar unavailable",
            )
        return planned_path

    def _plan_drone_corridor_astar(
        self,
        robot_id: str,
        start_pose,
        prior: VictimPrior,
    ):
        planner = self.path_planner
        if (
            not self.drone_corridor_enabled
            or self.drone_corridor_radius_m <= 0.0
            or self.drone_corridor_cost_multiplier >= 1.0
            or not self.drone_path
            or planner is None
            or not getattr(planner, "ready", False)
            or getattr(planner, "free_grid", None) is None
            or getattr(planner, "geometry", None) is None
        ):
            return None

        geometry = planner.geometry
        raw_start = geometry.world_to_pixel(start_pose.x, start_pose.y)
        raw_goal = geometry.world_to_pixel(prior.world[0], prior.world[1])
        start = planner._nearest_free_cell(raw_start)
        goal = planner._nearest_free_cell(raw_goal)
        if start is None or goal is None:
            return None

        corridor_cells = self._drone_corridor_cells()
        if not corridor_cells:
            return None

        pixel_path, weighted_cost_px = _weighted_astar_on_grid(
            planner.free_grid,
            planner.width,
            planner.height,
            start,
            goal,
            corridor_cells,
            self.drone_corridor_cost_multiplier,
        )
        if not pixel_path:
            return None

        world_path = tuple(geometry.pixel_to_world(px, py) for px, py in pixel_path)
        waypoints = self._waypoints_from_points(world_path)
        physical_cost = _polyline_length(world_path)
        path = PlannedPath(
            robot_id=robot_id,
            victim_index=prior.index,
            victim_world=prior.world,
            pixel_path=tuple(pixel_path),
            world_path=world_path,
            waypoints=tuple(waypoints),
            waypoint_speed_caps=tuple(None for _ in waypoints),
            planning_mode=DRONE_CORRIDOR_ASTAR,
            physical_path_cost_m=physical_cost,
            weighted_path_cost_m=weighted_cost_px * geometry.resolution,
            success=True,
        )
        return path

    def _drone_corridor_cells(self):
        planner = self.path_planner
        geometry = getattr(planner, "geometry", None)
        if planner is None or geometry is None:
            return frozenset()
        radius_px = int(
            math.ceil(
                self.drone_corridor_radius_m
                * geometry.pixels_per_metre
            )
        )
        if radius_px <= 0:
            return frozenset()

        pixel_points = []
        for point in self.drone_path:
            try:
                pixel = geometry.world_to_pixel(float(point[0]), float(point[1]))
            except (TypeError, ValueError, IndexError):
                continue
            if not pixel_points or pixel_points[-1] != pixel:
                pixel_points.append(pixel)
        if not pixel_points:
            return frozenset()

        key = (
            planner.width,
            planner.height,
            radius_px,
            tuple(pixel_points),
        )
        cached = self._drone_corridor_cache.get(key)
        if cached is not None:
            return cached

        offsets = planner._inflation_offsets(radius_px)
        path_cells = set()
        if len(pixel_points) == 1:
            path_cells.add(pixel_points[0])
        else:
            for start, end in zip(pixel_points, pixel_points[1:]):
                path_cells.update(_bresenham_cells(start, end))

        cells = set()
        for cx, cy in path_cells:
            for dx, dy in offsets:
                px = cx + dx
                py = cy + dy
                if 0 <= px < planner.width and 0 <= py < planner.height:
                    cells.add((px, py))

        corridor = frozenset(cells)
        self._drone_corridor_cache[key] = corridor
        return corridor

    def _refresh_cluster_state(self, cluster_id: str, robot_id: str = ""):
        cluster = self.clusters.get(cluster_id)
        if cluster is None:
            return
        states = [self.victim_states.get(prior_id, "unassigned") for prior_id in cluster.prior_ids]
        if all(state in TERMINAL_VICTIM_STATES for state in states):
            cluster.state = "complete"
            cluster.assigned_robot = ""
            if robot_id:
                self.robot_cluster.pop(robot_id, None)
        elif all(state in HANDLED_VICTIM_STATES for state in states):
            cluster.state = "found"
            if robot_id:
                cluster.assigned_robot = robot_id
        elif any(state == "assigned" for state in states):
            cluster.state = "assigned"
            if robot_id:
                cluster.assigned_robot = robot_id
        elif cluster.assigned_robot and any(
            state not in (TERMINAL_VICTIM_STATES | DEFERRED_VICTIM_STATES)
            for state in states
        ):
            cluster.state = "reserved"
        else:
            cluster.state = "unassigned"

    def _assigned_robot_for_prior(self, prior_id: str):
        assigned_robot = self.prior_assigned_robot.get(prior_id, "")
        if assigned_robot:
            return assigned_robot
        return self.planned_prior_owner.get(prior_id, "")

    def _resample_route(self, points):
        spacing = max(
            0.05,
            _env_float(
                "COORDINATOR_DRONE_ROUTE_SPACING_M",
                CoordinatorParams.DRONE_ROUTE_SPACING_M,
            ),
        )
        result = [points[0]]
        for start, end in zip(points, points[1:]):
            length = _distance(start, end)
            steps = max(1, int(math.ceil(length / spacing)))
            for step in range(1, steps + 1):
                ratio = step / steps
                result.append(
                    (
                        start[0] + ratio * (end[0] - start[0]),
                        start[1] + ratio * (end[1] - start[1]),
                    )
                )
        return _deduplicate_points(result)

    def _shortcut_if_verified(self, points):
        if len(points) <= 2 or not self._has_map_safety_checks():
            return points
        result = [points[0]]
        anchor = 0
        last = len(points) - 1
        while anchor < last:
            candidate = last
            while candidate > anchor + 1 and not self._segment_is_verified_free(
                points[anchor],
                points[candidate],
            ):
                candidate -= 1
            result.append(points[candidate])
            anchor = candidate
        return _deduplicate_points(result)

    def _has_map_safety_checks(self):
        planner = self.path_planner
        geometry = getattr(planner, "geometry", None) or getattr(self.drone_map, "geometry", None)
        return geometry is not None and (
            getattr(planner, "free_grid", None) is not None
            or getattr(planner, "source_occupied_cells", None)
        )

    def _segment_is_verified_free(self, start, end):
        planner = self.path_planner
        geometry = getattr(planner, "geometry", None) or getattr(self.drone_map, "geometry", None)
        if geometry is None:
            return False
        start_px = geometry.world_to_pixel(start[0], start[1])
        end_px = geometry.world_to_pixel(end[0], end[1])
        for cell in _bresenham_cells(start_px, end_px):
            if hasattr(planner, "_is_free") and getattr(planner, "free_grid", None) is not None:
                if not planner._is_free(cell):
                    return False
            else:
                px, py = cell
                if not geometry.in_bounds(px, py):
                    return False
                if cell in getattr(planner, "source_occupied_cells", frozenset()):
                    return False
        return True

    def _pixel_path(self, points):
        geometry = getattr(self.path_planner, "geometry", None) or getattr(self.drone_map, "geometry", None)
        if geometry is None:
            return ()
        return tuple(geometry.world_to_pixel(point[0], point[1]) for point in points)

    @staticmethod
    def _waypoints_from_points(points):
        waypoints = []
        for index in range(1, len(points)):
            current = points[index]
            if index + 1 < len(points):
                next_point = points[index + 1]
            else:
                next_point = current
                current = points[index]
            previous = points[index - 1]
            heading_to = next_point if next_point != current else current
            yaw = math.atan2(heading_to[1] - previous[1], heading_to[0] - previous[0])
            waypoints.append(Pose2D(current[0], current[1], yaw))
        return waypoints


def _cache_key(robot_id, start_pose, prior_id, route_policy="default"):
    xy = _pose_xy(start_pose)
    if xy is None:
        return robot_id, None, prior_id, route_policy
    return robot_id, round(xy[0], 2), round(xy[1], 2), prior_id, route_policy


def _pose_xy(pose):
    if pose is None:
        return None
    return float(pose.x), float(pose.y)


def _route_score(path):
    if path is None or not path.success:
        return float("inf")
    return path.weighted_path_cost_m or path.physical_path_cost_m


def _nearest_path_index(point, path):
    if not path:
        return -1
    best_index = 0
    best_distance = float("inf")
    for index, candidate in enumerate(path):
        distance = _distance(point, candidate)
        if distance < best_distance:
            best_index = index
            best_distance = distance
    return best_index


def _centroid(points):
    if not points:
        return 0.0, 0.0
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )


def _distance(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _polyline_length(points):
    return sum(_distance(start, end) for start, end in zip(points, points[1:]))


def _deduplicate_points(points):
    result = []
    for point in points:
        point = (float(point[0]), float(point[1]))
        if not result or _distance(result[-1], point) > 1e-6:
            result.append(point)
    return tuple(result)


def _bresenham_cells(start, end):
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    x, y = x0, y0
    while True:
        yield x, y
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def _weighted_astar_on_grid(
    free_grid,
    width,
    height,
    start,
    goal,
    corridor_cells,
    corridor_cost_multiplier,
):
    corridor_cells = corridor_cells or frozenset()
    corridor_cost_multiplier = min(1.0, max(0.05, float(corridor_cost_multiplier)))
    heuristic_multiplier = corridor_cost_multiplier if corridor_cells else 1.0
    open_heap = []
    counter = 0
    heapq.heappush(
        open_heap,
        (
            _octile_distance(start, goal) * heuristic_multiplier,
            0.0,
            counter,
            start,
        ),
    )
    came_from = {}
    g_score = {start: 0.0}
    closed = set()
    neighbours = (
        (-1, -1, math.sqrt(2.0)),
        (0, -1, 1.0),
        (1, -1, math.sqrt(2.0)),
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (-1, 1, math.sqrt(2.0)),
        (0, 1, 1.0),
        (1, 1, math.sqrt(2.0)),
    )

    while open_heap:
        _, current_g, _, current = heapq.heappop(open_heap)
        if current in closed:
            continue
        if current == goal:
            return _reconstruct_path(came_from, current), current_g

        closed.add(current)
        cx, cy = current
        for dx, dy, move_cost in neighbours:
            neighbour = (cx + dx, cy + dy)
            nx, ny = neighbour
            if (
                nx < 0
                or nx >= width
                or ny < 0
                or ny >= height
                or not free_grid[ny][nx]
                or neighbour in closed
            ):
                continue

            step_cost = (
                move_cost * corridor_cost_multiplier
                if neighbour in corridor_cells
                else move_cost
            )
            tentative_g = current_g + step_cost
            if tentative_g >= g_score.get(neighbour, float("inf")):
                continue

            came_from[neighbour] = current
            g_score[neighbour] = tentative_g
            counter += 1
            f_score = tentative_g + (
                _octile_distance(neighbour, goal) * heuristic_multiplier
            )
            heapq.heappush(open_heap, (f_score, tentative_g, counter, neighbour))

    return (), float("inf")


def _reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return tuple(path)


def _octile_distance(a, b):
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy)
