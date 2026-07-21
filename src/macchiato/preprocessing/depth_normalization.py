"""Depth clipping, invalid-value handling, and normalization utilities."""

from __future__ import annotations

import numpy as np


def normalize_depth(depth: np.ndarray, maximum_mm: float = 4000.0) -> np.ndarray:
    """Return a finite float32 depth map normalized to ``[0, 1]``."""

    if maximum_mm <= 0:
        raise ValueError("maximum_mm must be positive")
    clean = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    return np.clip(clean, 0.0, maximum_mm) / float(maximum_mm)


def depth_to_rgb_uint8(depth: np.ndarray, maximum_mm: float = 4000.0) -> np.ndarray:
    """Convert raw depth to a lossless three-channel uint8 representation."""

    gray = np.rint(normalize_depth(depth, maximum_mm) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def normalize_confidence(confidence: np.ndarray, maximum: float = 350.0) -> np.ndarray:
    """Apply the canonical clipped log normalization used by the v7 U-Net."""

    if maximum <= 0:
        raise ValueError("maximum must be positive")
    clean = np.nan_to_num(confidence, nan=0.0, posinf=maximum, neginf=0.0).astype(np.float32, copy=False)
    clean = np.clip(clean, 0.0, maximum)
    return np.log1p(clean) / np.log1p(maximum)
