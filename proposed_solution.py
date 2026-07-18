"""Select a controller mode and start the flat local robot runtime."""

from __future__ import annotations

from dataclasses import dataclass
import os

from config import _env_flag
from parameters import Controller as ControllerParams
from parameters import Mission as MissionParams
from parameters import PNG_OUTPUTS
from parameters import VISUALS


DEFAULT_CONTROLLER_MODE = ControllerParams.DEFAULT_MODE
MODE_ALIASES = ControllerParams.MODE_ALIASES
VALID_CONTROLLER_MODES = ControllerParams.VALID_MODES

@dataclass(frozen=True)
class RuntimeOptions:
    """Resolved launcher options for Robot1FlatRuntime."""

    mode: str
    coordinator_enabled: bool
    manual_only: bool
    victim_test: bool


def requested_controller_mode():
    """
    Read the desired mode.

    SAR_CONTROLLER_MODE is the new primary switch. SAR_ARCH_MODE is accepted as
    a temporary compatibility alias only when SAR_CONTROLLER_MODE is unset.
    """

    controller_mode = os.environ.get("SAR_CONTROLLER_MODE")
    if controller_mode not in (None, ""):
        return controller_mode
    arch_mode = os.environ.get("SAR_ARCH_MODE")
    if arch_mode not in (None, ""):
        return arch_mode
    return DEFAULT_CONTROLLER_MODE


def normalise_controller_mode(mode_text: str):
    requested = (mode_text or DEFAULT_CONTROLLER_MODE).strip().lower()
    mode = MODE_ALIASES.get(requested, requested)
    if requested != mode:
        print(
            f"[CONTROLLER] Mode {requested!r} is deprecated; "
            f"using SAR_CONTROLLER_MODE={mode}"
        )
    if mode not in VALID_CONTROLLER_MODES:
        print(
            f"[CONTROLLER] Unknown SAR_CONTROLLER_MODE={requested!r}; "
            f"using {DEFAULT_CONTROLLER_MODE}"
        )
        return DEFAULT_CONTROLLER_MODE
    return mode


def runtime_options_for(mode_text: str, robot_id: str):
    """Resolve controller behavior for this Webots robot instance."""

    mode = normalise_controller_mode(mode_text)
    if mode == "manual":
        return RuntimeOptions(
            mode=mode,
            coordinator_enabled=False,
            manual_only=True,
            victim_test=False,
        )
    if mode == "victim_test":
        return RuntimeOptions(
            mode=mode,
            coordinator_enabled=False,
            manual_only=True,
            victim_test=True,
        )
    return RuntimeOptions(
        mode=mode,
        coordinator_enabled=True,
        manual_only=False,
        victim_test=False,
    )


def _bool_env(value: bool):
    return "1" if bool(value) else "0"


def _visual_default(name: str, benchmark_mode: bool):
    if benchmark_mode:
        return False
    return bool(VISUALS.get(name, False))


def _png_output_default(name: str, benchmark_mode: bool):
    if benchmark_mode:
        return False
    return bool(PNG_OUTPUTS.get(name, True))


def apply_visual_defaults(robot_id: str, options: RuntimeOptions):
    """
    Set default viewer/render env vars before Robot1FlatRuntime constructs modules.

    These are setdefault() calls: shell env vars still win.  Examples:
      LIVE_MAP_VIEWER=0 disables the local live-map viewer for that robot.
      LIVE_MAP_RENDER=0 disables the local live-map PNG writer for that robot.
      DEPTH_DEBUG_VIEWER=1 forces the annotated depth viewer on.
    """

    benchmark_mode = _env_flag("SAR_BENCHMARK", MissionParams.BENCHMARK_ENABLED)
    robot_key = "robot2" if robot_id == "robot2" else "robot1"
    is_coordinator_host = (
        robot_id == "robot1"
        and options.mode == "normal"
        and options.coordinator_enabled
    )

    os.environ.setdefault(
        "DRONE_MAP_VIEWER",
        _bool_env(
            is_coordinator_host
            and _visual_default("coordinator_drone_map", benchmark_mode)
        ),
    )
    os.environ.setdefault(
        "DRONE_MAP_RENDER",
        _bool_env(
            is_coordinator_host
            and _png_output_default("coordinator_drone_map", benchmark_mode)
        ),
    )
    os.environ.setdefault(
        "PATH_PLANNER_VIEWER",
        _bool_env(
            is_coordinator_host
            and _visual_default("coordinator_path_plan", benchmark_mode)
        ),
    )
    os.environ.setdefault(
        "PATH_PLANNER_RENDER",
        _bool_env(
            is_coordinator_host
            and _png_output_default("coordinator_path_plan", benchmark_mode)
        ),
    )
    os.environ.setdefault(
        "COORDINATOR_RENDER_ASSIGNED_ROUTES",
        _bool_env(
            is_coordinator_host
            and _png_output_default("coordinator_path_plan", benchmark_mode)
        ),
    )
    os.environ.setdefault(
        "LIVE_MAP_VIEWER",
        _bool_env(_visual_default(f"{robot_key}_live_map", benchmark_mode)),
    )
    os.environ.setdefault(
        "LIVE_MAP_RENDER",
        _bool_env(_png_output_default(f"{robot_key}_live_map", benchmark_mode)),
    )
    os.environ.setdefault(
        "LIVE_REPLAN_VIEWER",
        _bool_env(_visual_default(f"{robot_key}_live_replan", benchmark_mode)),
    )
    os.environ.setdefault(
        "LIVE_REPLAN_RENDER",
        _bool_env(_png_output_default(f"{robot_key}_live_replan", benchmark_mode)),
    )
    os.environ.setdefault(
        "DEPTH_DEBUG_VIEWER",
        _bool_env(_visual_default(f"{robot_key}_depth_debug", benchmark_mode)),
    )
    os.environ.setdefault(
        "DEPTH_DEBUG_RENDER",
        _bool_env(_png_output_default(f"{robot_key}_depth_debug", benchmark_mode)),
    )
    os.environ.setdefault(
        "VICTIM_VIEWER_ENABLED",
        _bool_env(_visual_default(f"{robot_key}_victim_debug", benchmark_mode)),
    )
    os.environ.setdefault(
        "VICTIM_DEBUG_RENDER",
        _bool_env(_png_output_default(f"{robot_key}_victim_debug", benchmark_mode)),
    )
    os.environ.setdefault(
        "SAFETY_ESCAPE_DEBUG_RENDER",
        _bool_env(_png_output_default("safety_escape_debug", benchmark_mode)),
    )
    os.environ.setdefault(
        "ROUTE_START_DEBUG_RENDER",
        _bool_env(_png_output_default("route_start_debug", benchmark_mode)),
    )


def build_controller(robot=None, mode_text: str = None):
    if robot is None:
        from controller import Robot

        robot = Robot()

    from robot.mission import Robot1FlatRuntime

    selected_mode = requested_controller_mode() if mode_text is None else mode_text
    options = runtime_options_for(selected_mode, robot.getName())
    apply_visual_defaults(robot.getName(), options)

    if options.victim_test:
        os.environ.setdefault(
            "VICTIM_AUTONOMY_ENABLED",
            _bool_env(ControllerParams.VICTIM_TEST_AUTONOMY_ENABLED),
        )
        os.environ.setdefault(
            "VICTIM_REPORT_ENABLED",
            _bool_env(ControllerParams.VICTIM_TEST_REPORT_ENABLED),
        )

    print(
        f"[CONTROLLER] SAR_CONTROLLER_MODE={options.mode} "
        f"robot={robot.getName()} coordinator={options.coordinator_enabled} "
        f"manual={options.manual_only} victim_test={options.victim_test}"
    )
    return Robot1FlatRuntime(
        robot,
        coordinator_enabled=options.coordinator_enabled,
        manual_only=options.manual_only,
        victim_test=options.victim_test,
    )


def main():
    controller = build_controller()
    controller.run()


if __name__ == "__main__":
    main()
