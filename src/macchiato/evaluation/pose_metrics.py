"""Model-neutral pose metrics in original Arducam pixel coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PoseMetricAccumulator:
    """Accumulate PCK, detected-only MPJPE, and per-keypoint accuracy."""

    num_keypoints: int = 17
    pck5_threshold: float = 12.0
    pck10_threshold: float = 24.0
    sample_count: int = 0
    detected_count: int = 0
    visible_count: int = 0
    pck5_hits: int = 0
    pck10_hits: int = 0
    detected_errors: list[float] = field(default_factory=list)
    per_keypoint_total: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.int64))
    per_keypoint_pck5: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.int64))
    per_keypoint_pck10: np.ndarray = field(default_factory=lambda: np.zeros(17, dtype=np.int64))

    def __post_init__(self) -> None:
        self.per_keypoint_total = np.zeros(self.num_keypoints, dtype=np.int64)
        self.per_keypoint_pck5 = np.zeros(self.num_keypoints, dtype=np.int64)
        self.per_keypoint_pck10 = np.zeros(self.num_keypoints, dtype=np.int64)

    def update(self, predicted: np.ndarray | None, target: np.ndarray, valid_mask: np.ndarray) -> None:
        target = np.asarray(target, dtype=np.float32)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if target.shape != (self.num_keypoints, 2) or valid_mask.shape != (self.num_keypoints,):
            raise ValueError("Unexpected target or valid-mask shape")

        self.sample_count += 1
        visible_indices = np.flatnonzero(valid_mask)
        self.visible_count += len(visible_indices)
        self.per_keypoint_total[visible_indices] += 1
        if predicted is None:
            return

        predicted = np.asarray(predicted, dtype=np.float32)
        if predicted.shape != target.shape:
            raise ValueError(f"Unexpected prediction shape {predicted.shape}")
        self.detected_count += 1
        distances = np.linalg.norm(predicted[valid_mask] - target[valid_mask], axis=1)
        finite = np.isfinite(distances)
        self.detected_errors.extend(distances[finite].tolist())
        hits5 = finite & (distances <= self.pck5_threshold)
        hits10 = finite & (distances <= self.pck10_threshold)
        self.pck5_hits += int(hits5.sum())
        self.pck10_hits += int(hits10.sum())
        self.per_keypoint_pck5[visible_indices] += hits5.astype(np.int64)
        self.per_keypoint_pck10[visible_indices] += hits10.astype(np.int64)

    def summarize(self) -> dict[str, object]:
        """Return JSON-serializable aggregate and per-keypoint metrics."""

        denominator = max(1, self.visible_count)
        per5 = np.divide(
            self.per_keypoint_pck5,
            self.per_keypoint_total,
            out=np.full(self.num_keypoints, np.nan),
            where=self.per_keypoint_total > 0,
        )
        per10 = np.divide(
            self.per_keypoint_pck10,
            self.per_keypoint_total,
            out=np.full(self.num_keypoints, np.nan),
            where=self.per_keypoint_total > 0,
        )
        return {
            "samples": self.sample_count,
            "visible_keypoints": self.visible_count,
            "detection_coverage_percent": 100.0 * self.detected_count / max(1, self.sample_count),
            "pck5_percent": 100.0 * self.pck5_hits / denominator,
            "pck10_percent": 100.0 * self.pck10_hits / denominator,
            "mpjpe_detected_pixels": float(np.mean(self.detected_errors)) if self.detected_errors else None,
            "per_keypoint_pck5_percent": [None if np.isnan(value) else float(value * 100.0) for value in per5],
            "per_keypoint_pck10_percent": [None if np.isnan(value) else float(value * 100.0) for value in per10],
        }
