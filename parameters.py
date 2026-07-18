"""Human-facing tuning board for the SAR controller.

Edit this file when you want to change normal defaults in the codebase.
Environment variables still override these values at runtime through
config._env_float/_env_int/_env_flag, so benchmark one-offs can stay in the
terminal without changing source.

This module intentionally does not contain root paths, Webots device names,
state names, or radio protocol labels. Those remain in the modules that own
them.
"""

from __future__ import annotations

import math

from shared_types import Pose2D


class Controller:
    """Launcher mode defaults and temporary compatibility aliases."""

    DEFAULT_MODE = "normal"
    MODE_ALIASES = {
        "team_v2": "normal",
        "robot1_v2": "normal",
        "manual_v2": "manual",
        "legacy": "normal",
    }
    VALID_MODES = frozenset({"normal", "manual", "victim_test"})
    VICTIM_TEST_AUTONOMY_ENABLED = False
    VICTIM_TEST_REPORT_ENABLED = False


# Popup/window defaults used by proposed_solution.apply_visual_defaults().
VISUALS = {
    # Coordinator-owned global views. Only robot1 hosts the coordinator in V1.
    "coordinator_drone_map": False,
    "coordinator_path_plan": False,
    # Per-robot local views.
    "robot1_live_map":  False,
    "robot2_live_map": False,
    "robot1_live_replan": False,
    "robot2_live_replan": False,
    "robot1_depth_debug": False,
    "robot2_depth_debug": False,
    "robot1_victim_debug": False,
    "robot2_victim_debug": False,
}


# PNG output defaults used by proposed_solution.apply_visual_defaults().
# These control whether diagnostic images are written at all.  VISUALS only
# controls whether a popup window watches those images.
PNG_OUTPUTS = {
    # Coordinator-owned global outputs. Only robot1 hosts the coordinator in V1.
    "coordinator_drone_map": False,
    "coordinator_path_plan": False,
    # Per-robot local outputs.
    "robot1_live_map": False,
    "robot2_live_map": False,
    "robot1_live_replan": False,
    "robot2_live_replan": False,
    "robot1_depth_debug": False,
    "robot2_depth_debug": False,
    "robot1_victim_debug": False,
    "robot2_victim_debug": False,
    "safety_escape_debug": False,
    "route_start_debug": False,
}


class RobotGeometry:
    """Physical ROSbot dimensions shared by mapping, planning, and motion.

    These are the real robot body dimensions from the competition model.  Keep
    subsystem-specific clearances, margins, and collision envelopes in their
    owning classes below; those values are derived safety behaviour, not raw
    robot geometry.
    """

    LENGTH_M = 0.200
    WIDTH_M = 0.235
    HEIGHT_M = 0.220
    WHEEL_RADIUS_M = 0.043
    EFFECTIVE_TRACK_M = 0.271


class Coordinator:
    """Coordinator defaults and initial directive knobs."""

    ENABLED = True
    TEAM_COLLISION_AVOIDANCE_ENABLED = True
    TEAM_COLLISION_PRIORITY_ROBOT_ID = "robot2"
    TEAM_COLLISION_HOLD_DISTANCE_M = 0.5
    TEAM_COLLISION_RELEASE_DISTANCE_M = 0.65
    CLUSTER_RADIUS_M = 3.0
    DRONE_ROUTE_SPACING_M = 0.50
    CONSERVATIVE_SHORTCUTS = True
    DRONE_CORRIDOR_ENABLED = True
    DRONE_CORRIDOR_RADIUS_M = 2.0
    DRONE_CORRIDOR_COST_MULTIPLIER = 0.15


class Safety:
    """Last-line depth/IR collision thresholds."""

    ENABLED = True
    DEPTH_MIN_PIXELS = 300
    DEPTH_MIN_COLUMNS = 8
    DEPTH_CONFIRM_FRAMES = 1
    IR_SLOW_M = 0.30
    IR_STOP_M = 0.05
    IR_SLOW_SPEED_RAD_S = 3.0


class SafetyRecovery:
    """Post-latch yaw-aware local escape defaults."""

    ENABLED = True
    # Planning-only circle cleared around the latched robot in the fine map.
    START_CLEAR_RADIUS_M = 0.15
    # Number of discrete headings searched in the local (x, y, yaw) escape grid.
    YAW_BINS = 16
    # Length of one small straight primitive in the 1 cm/px local escape map.
    PRIMITIVE_STEP_M = 0.05
    # Radius used by left/right arc primitives during local escape search.
    PRIMITIVE_ARC_RADIUS_M = 0.18
    # Heading change used by in-place rotate primitives.
    ROTATE_STEP_RAD = math.radians(15.0)
    # Extra padding around the robot rectangle when checking footprint collision.
    FOOTPRINT_MARGIN_M = 0.00
    # Search guard so a bad local map cannot stall the controller.
    MAX_EXPANSIONS = 8000
    # If an escape route runs this long without relatching safety, treat any
    # later latch as a new incident with a fresh single recovery attempt.
    RESET_CLEAR_TIME_S = 0.5


class Localisation:
    """Competition-frame start poses used by encoder/compass localisation."""

    START_POSES = {
        "robot1": Pose2D(x=-0.375, y=0.375, yaw=0.0),
        "robot2": Pose2D(x=-0.375, y=0.0, yaw=0.0),
    }


class Terrain:
    """Tilt and IMU filtering defaults."""

    ENABLED = True
    ENTER_TILT_RAD = math.radians(1.0)
    ENTER_TILT_HOLD_S = 0.2
    EXIT_TILT_RAD = math.radians(3.0)
    LEVEL_HOLD_S = 0.50
    CRAWL_SPEED_RAD_S = 3.0
    ACCEL_CORRECTION_TAU_S = 1.50
    ACCEL_STARTUP_GRACE_S = 0.5
    IMU_HOLD_S = 0.50
    # Ground-like returns below this height are treated as floor/bumps rather
    # than collision obstacles.
    GROUND_MAX_HEIGHT_M = 0.03


class LocalObstacleProjection:
    """Depth/IR projection geometry used by sensing, safety, and local mapping."""

    ROBOT_WIDTH_M = RobotGeometry.WIDTH_M
    CAMERA_FORWARD_OFFSET_M = -0.027
    CAMERA_HEIGHT_M = 0.1991
    SIDE_MARGIN_M = 0.01
    # Collision band in robot/world height coordinates.  The upper limit is a
    # clearance envelope above the 0.220 m robot height, not a second robot
    # height definition.
    MIN_COLLISION_HEIGHT_M = 0.015
    MAX_COLLISION_HEIGHT_M = 0.24
    SAFETY_DEPTH_NEAR_M = 0.60
    SAFETY_DEPTH_STOP_M = 0.01
    MAP_DEPTH_NEAR_M = 0.60
    MAP_DEPTH_FAR_M = 1.6
    MAP_DEPTH_STRIDE = 8
    MAP_IR_MAX_RANGE_M = 0.7
    IR_MOUNTS = {
        "fl_range": (0.100, 0.050, 0.053, 0.130),
        "fr_range": (0.100, -0.050, 0.053, -0.130),
        "rl_range": (-0.100, 0.050, 0.053, 3.010),
        "rr_range": (-0.100, -0.050, 0.053, 3.270),
    }


class DepthViewer:
    """Annotated depth debug viewer defaults."""

    ENABLED_FOR_ROBOT1_BY_DEFAULT = True
    UPDATE_INTERVAL_S = 0.20
    HEADER_HEIGHT_PX = 192


class DepthCamera:
    """Depth-camera read cadence for safety, local mapping, and diagnostics."""

    HZ = 1.0


class Diagnostics:
    """Standalone diagnostic helper defaults."""

    LIVE_MAP_VIEWER_POLL_MS = 200


class DroneMap:
    """Drone extraction map loading/overlay defaults."""

    ROBOT_STARTS = Localisation.START_POSES


class LocalObstacleMap:
    """Confirmation and lifetime defaults for depth/IR map obstacles."""

    DEPTH_CONFIRM_FRAMES = 1
    IR_CONFIRM_FRAMES = 1
    IR_PATCH_RADIUS_M = 0.01
    STICKY = True
    OBSTACLE_TTL_S = 0.7


class LiveMap:
    """Lidar occupancy-grid, confidence, scan-match, and seed defaults."""

    RESOLUTION_M_PER_PX = 0.05
    LIDAR_HZ = 20
    RENDER_INTERVAL_S = 0.50
    RENDER_PADDING_PX = 40.0
    LIDAR_MIN_RANGE_M = 0.20
    LIDAR_MAX_RANGE_M = 10.0
    LIDAR_ANGLE_OFFSET_RAD = 0.0
    LIDAR_REVERSE = True
    YAW_SIGN = 1.0
    STICKY_OCCUPIED = False
    MARK_INF_FREE = False
    LIDAR_RAY_STRIDE = 1.0
    OUTLIER_FILTER = True
    OUTLIER_THRESHOLD_M = 0.40
    TERRAIN_PAUSE_ENABLED = True
    HIT_CONFIDENCE = 1.0
    FREE_CONFIDENCE = 0.25
    FREE_THRESHOLD = -0.5
    OCCUPIED_THRESHOLD = 3.0
    CONFIDENCE_MIN = -3.0
    CONFIDENCE_MAX = 5.0
    SCAN_MATCH = True
    SCAN_MATCH_XY_STEP_M = 0.04
    SCAN_MATCH_YAW_STEP_RAD = math.radians(2.0)
    SCAN_MATCH_STRIDE = 4.0
    SCAN_MATCH_OCCUPIED_RADIUS_PX = 2.0
    SCAN_MATCH_ODOM_PENALTY = 0.05
    SCAN_MATCH_MIN_HITS = 10.0
    SEED_FROM_DRONE = True
    USE_START_OVERRIDE_WITH_DRONE = False
    DRONE_OCCUPIED_CONFIDENCE = 3.0
    DRONE_FREE_CONFIDENCE = -1.0
    DRONE_OCCUPIED_PIXEL_THRESHOLD = 128.0


class Planning:
    """Global A* and waypoint simplification defaults."""

    PRINT_PLAN_SUMMARY_ENABLED = False
    ROBOT_WIDTH_M = RobotGeometry.WIDTH_M
    ROBOT_LENGTH_M = RobotGeometry.LENGTH_M
    OCCUPIED_PIXEL_THRESHOLD = 128
    WALL_CLEARANCE_M = 0.12
    WAYPOINT_SPACING_M = 0.0
    MIN_WAYPOINT_SEPARATION_M = 0.1
    SNAP_RADIUS_M = 1.0
    VICTIM_VIEWPOINT_CLEARANCE_M = 0.0


class EscapePlanning:
    """Fine-grid escape planner defaults used after safety latch."""

    FINE_RESOLUTION_M = 0.01
    FINE_CLEARANCE_M = 0.5 * RobotGeometry.WIDTH_M
    GOAL_CLEARANCE_M = 0.25
    PATCH_SIZES_M = (1.5, 2.5, 3.5)
    SPEED_CAP_RAD_S = 3.0


class LiveReplanning:
    """Live path obstruction/replan visual defaults."""

    ROBOT_LENGTH_M = RobotGeometry.LENGTH_M
    ROBOT_WIDTH_M = RobotGeometry.WIDTH_M
    REPLAN_INTERVAL_S = 1.0
    # Initial coordinator routes get one normal local A* attempt, then at most
    # this many planning-only retries before the directive is failed.
    COORDINATOR_LOCAL_RETRY_COUNT = 3
    # Ignore mapped occupancy under the robot centre only while constructing
    # those initial retries. The live occupancy map itself is never modified.
    COORDINATOR_START_CLEAR_RADIUS_M = 0.15
    REQUIRE_APPROVAL = False
    INFO_PANEL_WIDTH_PX = 370


class Motion:
    """Smooth lookahead route follower, slew, and scan-spin defaults."""

    POSITION_TOLERANCE_M = 0.05
    FINAL_YAW_TOLERANCE_RAD = math.radians(5.0)
    LOOKAHEAD_M = 0.15
    MAX_LINEAR_M_S = 0.25
    MAX_ANGULAR_RAD_S = 2.2
    MAX_PROFILE_WHEEL_RAD_S = 8.0
    KP_HEADING = 2.4
    KP_DISTANCE = 0.85
    ROTATE_IN_PLACE_HEADING_RAD = math.radians(40.0)
    HEADING_SLOWDOWN_MIN = 0.18
    NARROW_LOOKAHEAD_M = 0.14
    NARROW_MAX_LINEAR_M_S = 0.14
    NARROW_MAX_ANGULAR_RAD_S = 1.8
    NARROW_MAX_PROFILE_WHEEL_RAD_S = 6.0
    NARROW_ROTATE_IN_PLACE_HEADING_RAD = math.radians(40.0)
    NARROW_HEADING_SLOWDOWN_MIN = 0.10
    MAX_WHEEL_SPEED_RAD_S = 26.0
    MAX_WHEEL_ACCEL_RAD_S2 = 26.0
    VICTIM_SCAN_SPIN_SPEED_RAD_S = 2.5

    #Gabriel twigg-ho Krish
class Victim:
    """Victim detector, tracker, search, reporting, and viewer defaults."""

    AUTONOMY_ENABLED = True
    DETECTOR_ENABLED = True
    MODEL_IMAGE_SIZE = 832
    MODEL_CONFIDENCE = 0.7
    MAX_RANGE_M = 4.0
    RESULT_MAX_AGE_S = 1.0
    MIN_DEPTH_PIXELS = 50
    MIN_DEPTH_FRACTION = 0.20
    MASK_ERODE_PX = 5
    TRANSIT_HZ = 2.0
    SCAN_HZ = 5.0
    MODEL_DEVICE = "cuda"
    PRIOR_ASSOCIATION_M = 1.50
    VISUAL_ASSOCIATION_M = 0.75
    TRACK_FUSION_ALPHA = 0.35
    PRIOR_AMBIGUITY_MARGIN_M = 0.25
    TEMP_TRACK_TTL_S = 2.0
    PRIOR_SELECTION_MODE = "nearest"
    SEARCH_TIMEOUT_S = 300.0
    REQUIRED_APPROACH_DISTANCE_M = 1.0
    REPORT_DISTANCE_M = 0.35
    CLOSE_IN_FINAL_DISTANCE_M = 0.35
    # Wheel-speed cap for the single continuous close-in route.
    CLOSE_IN_SPEED_CAP_RAD_S = 5.0
    CLOSE_IN_MIN_ADVANCE_M = 0.08
    # Maximum age of a report-quality visual lock used to commit close-in.
    REPORT_LOCK_MAX_AGE_S = 12.0
    REPORT_MIN_CONFIDENCE = 0.70
    REPORT_ENABLED = True
    DEBUG_PRINT_ENABLED = False
    # One-shot side-view verification for wide, low victim masks near 2 m.
    SIDE_VIEW_ENABLED = True
    SIDE_VIEW_RANGE_MIN_M = 1.75
    SIDE_VIEW_RANGE_MAX_M = 2.10
    SIDE_VIEW_MIN_MASK_WIDTH_PX = 230 #245 #250
    SIDE_VIEW_MIN_MASK_HEIGHT_PX = 100 #100 #110
    SIDE_VIEW_MAX_MASK_HEIGHT_PX = 190 #190 #180
    # Radius from the fused victim position used for left/right side-view poses.
    SIDE_VIEW_DISTANCE_M = 0.4
    # Print and save selected-mask size once when crossing 3 m, 2 m, and 1 m.
    VIEWER_ENABLED_ROBOT1_DEFAULT = False


class Mission:
    """Runtime loop, benchmark, manual, and coordinator timing defaults."""

    WHEEL_RADIUS_M = RobotGeometry.WHEEL_RADIUS_M
    PRINT_STATUS_ENABLED = False
    PRINT_INTERVAL_S = 0.5
    ROBOT1_STRAIGHT_TEST_DISTANCE_M = 8.0
    DRIVE_SPEED_RAD_S = 26.0
    TURN_SPEED_RAD_S = 26.0
    MAX_WHEEL_SPEED_RAD_S = Motion.MAX_WHEEL_SPEED_RAD_S
    MAX_WHEEL_ACCEL_RAD_S2 = Motion.MAX_WHEEL_ACCEL_RAD_S2
    COORDINATOR_STATUS_INTERVAL_S = 0.2
    BENCHMARK_ENABLED = False
    PROFILE_ENABLED = False
    PROFILE_SECONDS = 60.0
    BENCHMARK_TARGET_X = 1.0
    BENCHMARK_TARGET_Y = 0.0
    BENCHMARK_TARGET_YAW = 0.0
