"""World/pixel coordinate transforms for drone and live maps."""

from __future__ import annotations


class MapGeometry:
    """Coordinate transform for the drone-extracted map."""

    def __init__(self, info: dict):
        self.resolution = float(info["resolution"])
        self.pixels_per_metre = float(info["pixels_per_metre"])
        self.canvas_size = int(info["canvas_size"])
        self.center_x_m = float(info["center_x_m"])
        self.center_y_m = float(info["center_y_m"])
        self.image_x_sign = float(info["image_x_sign"])
        self.image_y_sign = float(info["image_y_sign"])
        origin_px = info.get("origin_px", (None, None))
        self.origin_px = (int(origin_px[0]), int(origin_px[1]))

    def world_to_pixel_float(self, x_m: float, y_m: float):
        px = (
            self.canvas_size / 2.0
            + self.image_x_sign * (y_m - self.center_y_m) * self.pixels_per_metre
        )
        py = (
            self.canvas_size / 2.0
            + self.image_y_sign * (x_m - self.center_x_m) * self.pixels_per_metre
        )
        return px, py

    def world_to_pixel(self, x_m: float, y_m: float):
        px, py = self.world_to_pixel_float(x_m, y_m)
        return int(round(px)), int(round(py))

    def pixel_to_world(self, px: float, py: float):
        x_m = self.center_x_m + (
            ((py - self.canvas_size / 2.0) * self.resolution) / self.image_y_sign
        )
        y_m = self.center_y_m + (
            ((px - self.canvas_size / 2.0) * self.resolution) / self.image_x_sign
        )
        return x_m, y_m

    def in_bounds(self, px: int, py: int):
        return 0 <= px < self.canvas_size and 0 <= py < self.canvas_size
