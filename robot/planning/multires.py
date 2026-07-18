"""Fine-grid clearance helpers for yaw-aware safety escape planning."""

from __future__ import annotations


def euclidean_clearance_pixels(occupied):
    import numpy as np

    height, width = occupied.shape
    maximum = float(width * width + height * height)
    distances = np.where(occupied, 0.0, maximum)

    for py in range(height):
        distances[py, :] = distance_transform_1d(distances[py, :])
    for px in range(width):
        distances[:, px] = distance_transform_1d(distances[:, px])

    return np.sqrt(distances)


def distance_transform_1d(values):
    import numpy as np

    length = len(values)
    sites = np.zeros(length, dtype=np.int32)
    boundaries = np.empty(length + 1, dtype=np.float64)
    result = np.empty(length, dtype=np.float64)
    k = 0
    sites[0] = 0
    boundaries[0] = -float("inf")
    boundaries[1] = float("inf")

    for q in range(1, length):
        while True:
            p = sites[k]
            numerator = (values[q] + q * q) - (values[p] + p * p)
            denominator = 2 * (q - p)
            split = numerator / denominator if denominator else float("inf")
            if split > boundaries[k]:
                break
            k -= 1
        k += 1
        sites[k] = q
        boundaries[k] = split
        boundaries[k + 1] = float("inf")

    k = 0
    for q in range(length):
        while boundaries[k + 1] < q:
            k += 1
        p = sites[k]
        result[q] = (q - p) * (q - p) + values[p]
    return result
