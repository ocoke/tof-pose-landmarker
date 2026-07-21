"""Joint image and COCO-keypoint transforms."""

from __future__ import annotations

import random

import numpy as np

from macchiato.adapters.arducam_adapter import COCO_FLIP_INDICES


def horizontal_flip(channels: np.ndarray, keypoints: np.ndarray, valid_mask: np.ndarray):
    """Flip channels and swap COCO left/right keypoint identities."""

    flipped = channels[:, :, ::-1].copy()
    points = keypoints.copy()
    points[:, 0] = channels.shape[2] - 1 - points[:, 0]
    indices = np.asarray(COCO_FLIP_INDICES)
    return flipped, points[indices], valid_mask[indices]


def random_basic_transform(
    channels: np.ndarray,
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    max_rotation_degrees: float = 15.0,
):
    """Apply the canonical v7 horizontal flip and small rotation."""

    if random.random() < 0.5:
        channels, keypoints, valid_mask = horizontal_flip(channels, keypoints, valid_mask)

    angle = random.uniform(-max_rotation_degrees, max_rotation_degrees)
    if abs(angle) < 1e-6:
        return channels, keypoints, valid_mask
    try:
        import cv2
    except ModuleNotFoundError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("OpenCV is required for U-Net training augmentation") from exc

    height, width = channels.shape[1:]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0).astype(np.float32)
    warped = np.stack(
        [
            cv2.warpAffine(channel, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=0.0)
            for channel in channels
        ]
    ).astype(np.float32)
    homogeneous = np.concatenate([keypoints.astype(np.float32), np.ones((len(keypoints), 1), dtype=np.float32)], axis=1)
    transformed = (matrix @ homogeneous.T).T.astype(np.float32)
    return warped, transformed, valid_mask


def generate_heatmaps(
    keypoints: np.ndarray,
    valid_mask: np.ndarray,
    output_shape: tuple[int, int],
    sigma: float = 4.0,
) -> np.ndarray:
    """Generate one Gaussian heatmap per keypoint."""

    height, width = output_shape
    x_grid, y_grid = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    heatmaps = np.zeros((len(keypoints), height, width), dtype=np.float32)
    for index, ((x_coord, y_coord), valid) in enumerate(zip(keypoints, valid_mask)):
        if valid and 0 <= x_coord < width and 0 <= y_coord < height:
            heatmaps[index] = np.exp(-((x_grid - x_coord) ** 2 + (y_grid - y_coord) ** 2) / (2.0 * sigma**2))
    return heatmaps
