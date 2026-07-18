"""Waypoint and polyline helpers for PathPlanner."""

from __future__ import annotations

import math

def polyline_length(points):
    return sum(math.hypot(end[0] - start[0], end[1] - start[1]) for start, end in zip(points, points[1:]))
