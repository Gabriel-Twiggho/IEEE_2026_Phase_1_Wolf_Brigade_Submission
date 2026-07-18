"""Mapping package with lazy public compatibility exports."""

from __future__ import annotations

__all__ = (
    "MapGeometry",
    "DroneExtractionMap",
    "LocalObstacleLayer",
    "LiveOccupancyGridMapper",
)


def __getattr__(name: str):
    if name == "MapGeometry":
        from robot.mapping.geometry import MapGeometry

        return MapGeometry
    if name == "DroneExtractionMap":
        from robot.mapping.drone_map import DroneExtractionMap

        return DroneExtractionMap
    if name == "LocalObstacleLayer":
        from robot.mapping.local_obstacles import LocalObstacleLayer

        return LocalObstacleLayer
    if name == "LiveOccupancyGridMapper":
        from robot.mapping.live_grid import LiveOccupancyGridMapper

        return LiveOccupancyGridMapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
