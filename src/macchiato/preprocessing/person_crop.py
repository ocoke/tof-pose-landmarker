"""Person bounding-box helpers."""

from __future__ import annotations

import numpy as np


def padded_keypoint_box(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    width: int,
    height: int,
    padding_fraction: float = 0.10,
) -> tuple[float, float, float, float]:
    """Return a clipped ``(x_min, y_min, x_max, y_max)`` box.

    Padding is applied independently to each side using the visible-keypoint
    span on that axis. A one-pixel minimum keeps the box positive when points
    share an x or y coordinate.
    """

    points = np.asarray(keypoints, dtype=np.float32)[np.asarray(valid_mask, dtype=bool)]
    if points.size == 0:
        raise ValueError("At least one valid keypoint is required to create a box")
    if padding_fraction < 0:
        raise ValueError("padding_fraction cannot be negative")

    x_min, y_min = points.min(axis=0)
    x_max, y_max = points.max(axis=0)
    pad_x = max(1.0, float(x_max - x_min) * padding_fraction)
    pad_y = max(1.0, float(y_max - y_min) * padding_fraction)

    x_min = max(0.0, float(x_min) - pad_x)
    y_min = max(0.0, float(y_min) - pad_y)
    x_max = min(float(width), float(x_max) + pad_x)
    y_max = min(float(height), float(y_max) + pad_y)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("Keypoints produced a degenerate person box")
    return x_min, y_min, x_max, y_max
