"""Planning package with lazy public compatibility exports."""

from __future__ import annotations

__all__ = ("PathPlanner", "LiveReplanner")


def __getattr__(name: str):
    if name == "PathPlanner":
        from robot.planning.path_planner import PathPlanner

        return PathPlanner
    if name == "LiveReplanner":
        from robot.planning.live_replanner import LiveReplanner

        return LiveReplanner
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
