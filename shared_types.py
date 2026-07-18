"""Pure dataclasses shared by the flat SAR controller modules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Pose2D:
    """2D pose in either odom or competition/world frame."""

    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class OdomDelta:
    """Per-step encoder/compass odometry movement summary."""

    left: float = 0.0
    right: float = 0.0
    forward: float = 0.0
    yaw_delta: float = 0.0


@dataclass
class PlannedPath:
    """Result container for map/path planning."""

    robot_id: str = ""
    victim_index: int = 0
    victim_world: tuple = None
    pixel_path: tuple = ()
    world_path: tuple = ()
    waypoints: tuple = ()
    planning_mode: str = "GLOBAL"
    waypoint_speed_caps: tuple = ()
    global_comparison_pixel_path: tuple = ()
    final_target_yaw: float = None
    physical_path_cost_m: float = 0.0
    weighted_path_cost_m: float = 0.0
    planning_time_s: float = 0.0
    success: bool = False
    error_reason: str = ""


@dataclass
class SafetyDecision:
    """Final wheel command and diagnostic result from SafetyCollisionLayer."""

    left_rad_s: float = 0.0
    right_rad_s: float = 0.0
    state: str = "CLEAR"
    source: str = ""
    reason: str = ""
    new_stop: bool = False
    depth_pixels: int = 0
    depth_columns: int = 0
    trigger_sensor: str = ""


@dataclass(frozen=True)
class TerrainAssessment:
    """IMU attitude estimate plus terrain state."""

    state: str = "UNRELIABLE"
    pitch_rad: float = 0.0
    roll_rad: float = 0.0
    tilt_rad: float = 0.0
    up_vector: tuple = (0.0, 0.0, 1.0)
    reliable: bool = False
    candidate_s: float = 0.0
    accel_startup_gated: bool = False


@dataclass(frozen=True)
class LocalObstaclePoint:
    """One depth or IR obstacle point in robot-local coordinates."""

    source: str
    forward_m: float
    lateral_m: float
    height_m: float
    distance_m: float


@dataclass
class LocalObstacleObservation:
    """Shared depth/IR interpretation consumed by mapping and later safety."""

    ir_ranges: dict = None
    collision_ir_ranges: dict = None
    ground_like_ir_sensors: tuple = ()
    depth_points: tuple = ()
    safety_depth_points: tuple = ()
    ir_points: tuple = ()
    safety_depth_pixels: int = 0
    safety_depth_columns: int = 0
    depth_sampled: bool = True
    terrain: object = None


@dataclass(frozen=True)
class LiveMapSnapshot:
    """Immutable occupied-cell view used by future planning/replanning."""

    geometry: object
    width: int
    height: int
    occupied_cells: frozenset
    revision: int


@dataclass
class LiveReplanEvent:
    """One state transition requested by LiveReplanner."""

    action: str = "NONE"
    path: PlannedPath = None
    reason: str = ""


@dataclass
class RecoveryDecision:
    """One command or state transition from SafetyRecoveryController."""

    left_rad_s: float = 0.0
    right_rad_s: float = 0.0
    action: str = "NONE"
    reason: str = ""


@dataclass
class MapSummary:
    """Small map status summary for RobotState."""

    ready: bool = False
    revision: int = 0
    occupied_count: int = 0
    victim_count: int = 0
    error: str = ""


@dataclass
class VictimSummary:
    """Small victim status summary for RobotState."""

    state: str = "IDLE"
    selected_track_id: str = ""
    reason: str = ""
    found_prior_ids: tuple = ()
    exhausted_prior_ids: tuple = ()
    report_ready: bool = False
    confidence: float = 0.0


@dataclass
class RouteSummary:
    """Compact route status stored in RobotState and coordinator messages."""

    active: bool = False
    route_id: str = ""
    target_id: str = ""
    state: str = "IDLE"
    waypoint_index: int = 0
    waypoint_count: int = 0


@dataclass
class SafetySummary:
    """Compact local safety/recovery status stored in RobotState."""

    state: str = "CLEAR"
    source: str = ""
    reason: str = ""
