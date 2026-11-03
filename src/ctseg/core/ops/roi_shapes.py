"""Rasterize editable 2D ROI geometry into boolean masks."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def rasterize_contour(
    vertices_xy: Sequence[tuple[float, float]], shape_yx: tuple[int, int]
) -> np.ndarray:
    """Fill a closed polygon using pixel-center ray casting.

    Keeping all ROI tools on this mask representation means segmentation
    operations do not need tool-specific polygon, lasso, or widget logic.
    """
    height, width = shape_yx
    if height < 0 or width < 0:
        raise ValueError("image shape cannot contain negative dimensions")
    points = np.asarray(vertices_xy, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 3:
        raise ValueError("a polygon requires at least three x,y vertices")
    if not np.all(np.isfinite(points)):
        raise ValueError("contour vertices must be finite")
    output = np.zeros((height, width), dtype=bool)
    min_x = max(0, int(np.floor(points[:, 0].min())))
    max_x = min(width - 1, int(np.ceil(points[:, 0].max())))
    min_y = max(0, int(np.floor(points[:, 1].min())))
    max_y = min(height - 1, int(np.ceil(points[:, 1].max())))
    if min_x > max_x or min_y > max_y:
        return output
    yy, xx = np.mgrid[min_y : max_y + 1, min_x : max_x + 1]
    px = xx + 0.5
    py = yy + 0.5
    inside = np.zeros(px.shape, dtype=bool)
    previous = points[-1]
    for current in points:
        x1, y1 = current
        x2, y2 = previous
        crosses = (y1 > py) != (y2 > py)
        crossing_x = (x2 - x1) * (py - y1) / ((y2 - y1) + np.finfo(float).eps) + x1
        inside ^= crosses & (px < crossing_x)
        previous = current
    output[min_y : max_y + 1, min_x : max_x + 1] = inside
    return output


def rectangle_mask(
    first_xy: tuple[float, float], second_xy: tuple[float, float], shape_yx: tuple[int, int]
) -> np.ndarray:
    """Create an inclusive pixel-center rectangle from two drag corners."""
    x1, y1 = first_xy
    x2, y2 = second_xy
    return rasterize_contour([(x1, y1), (x2, y1), (x2, y2), (x1, y2)], shape_yx)


def ellipse_mask(
    first_xy: tuple[float, float], second_xy: tuple[float, float], shape_yx: tuple[int, int]
) -> np.ndarray:
    """Rasterize an ellipse bounded by a drag rectangle."""
    height, width = shape_yx
    x1, y1 = first_xy
    x2, y2 = second_xy
    center_x = (x1 + x2) / 2.0
    center_y = (y1 + y2) / 2.0
    radius_x = abs(x2 - x1) / 2.0
    radius_y = abs(y2 - y1) / 2.0
    yy, xx = np.ogrid[:height, :width]
    if radius_x == 0 or radius_y == 0:
        return np.zeros((height, width), dtype=bool)
    return ((xx + 0.5 - center_x) / radius_x) ** 2 + ((yy + 0.5 - center_y) / radius_y) ** 2 <= 1.0
