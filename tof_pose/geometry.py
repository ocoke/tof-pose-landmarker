from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .types import CameraIntrinsics, DepthFrame, FloorPlane, TrackedPerson


@dataclass(slots=True)
class GeometryConfig:
    min_depth_m: float = 0.4
    max_depth_m: float = 4.0
    confidence_threshold: float = 30.0
    background_threshold_m: float = 0.12
    floor_distance_threshold_m: float = 0.04
    min_component_pixels: int = 120
    roi_margin_px: int = 18
    max_missing_background_frames: int = 45
    min_human_height_m: float = 0.55
    max_human_height_m: float = 2.3
    min_human_width_m: float = 0.15
    max_human_width_m: float = 1.5
    max_cluster_depth_m: float = 4.5
    track_roi_alpha: float = 0.35
    fallback_near_percentile: float = 12.0
    fallback_depth_window_m: float = 0.35
    fallback_max_component_fraction: float = 0.45
    fallback_max_roi_fraction: float = 0.85
    track_max_depth_jump_m: float = 0.45
    track_depth_jump_penalty: float = 3200.0
    track_reacquire_after_frames: int = 5
    track_hold_frames: int = 3
    track_max_center_step_px: float = 24.0
    track_max_size_step_fraction: float = 0.12
    max_candidate_roi_fraction: float = 0.85
    full_height_roi_fraction: float = 0.95
    full_height_min_fill_ratio: float = 0.16
    min_previous_roi_iou: float = 0.08


@dataclass(slots=True)
class TrackingDiagnostics:
    track_source: str = "none"
    held: bool = False
    roi_area_frac: float = 0.0
    mask_pixels: int = 0
    candidate_count: int = 0
    reject_counts: dict[str, int] = field(default_factory=dict)
    raw_roi_px: tuple[int, int, int, int] | None = None
    final_roi_px: tuple[int, int, int, int] | None = None

    def to_json(self) -> dict[str, object]:
        return {
            "track_source": self.track_source,
            "held": self.held,
            "roi_area_frac": round(self.roi_area_frac, 4),
            "mask_pixels": self.mask_pixels,
            "candidate_count": self.candidate_count,
            "reject_counts": dict(self.reject_counts),
            "raw_roi_px": list(self.raw_roi_px) if self.raw_roi_px is not None else None,
            "final_roi_px": list(self.final_roi_px) if self.final_roi_px is not None else None,
        }


@dataclass(slots=True)
class FloorCalibrationDiagnostics:
    frames_total: int = 0
    frames_with_points: int = 0
    strict_points_total: int = 0
    lower_band_points_total: int = 0
    used_points_total: int = 0
    frames_relaxed_confidence: int = 0
    frames_full_frame_fallback: int = 0
    confidence_p95: float = 0.0
    confidence_max: float = 0.0

    def to_json(self) -> dict[str, float | int]:
        return {
            "frames_total": self.frames_total,
            "frames_with_points": self.frames_with_points,
            "strict_points_total": self.strict_points_total,
            "lower_band_points_total": self.lower_band_points_total,
            "used_points_total": self.used_points_total,
            "frames_relaxed_confidence": self.frames_relaxed_confidence,
            "frames_full_frame_fallback": self.frames_full_frame_fallback,
            "confidence_p95": round(self.confidence_p95, 4),
            "confidence_max": round(self.confidence_max, 4),
        }


@dataclass(slots=True)
class FloorCandidateMasks:
    depth_valid: np.ndarray
    lower_candidates: np.ndarray
    strict_candidates: np.ndarray
    used_mask: np.ndarray


def floor_candidate_masks(frame: DepthFrame, config: GeometryConfig) -> FloorCandidateMasks:
    depth_valid = (
        frame.valid_mask
        & (frame.depth_m >= config.min_depth_m)
        & (frame.depth_m <= config.max_cluster_depth_m)
    )
    lower_band = np.zeros_like(depth_valid)
    lower_band[depth_valid.shape[0] // 3 :, :] = True
    lower_candidates = depth_valid & lower_band
    strict_candidates = lower_candidates & (frame.confidence >= config.confidence_threshold)

    used_mask = strict_candidates
    if np.count_nonzero(used_mask) < 256:
        used_mask = lower_candidates
    if np.count_nonzero(used_mask) < 256:
        used_mask = depth_valid

    return FloorCandidateMasks(
        depth_valid=depth_valid,
        lower_candidates=lower_candidates,
        strict_candidates=strict_candidates,
        used_mask=used_mask,
    )


def depth_to_point_cloud(
    depth_m: np.ndarray,
    intrinsics: CameraIntrinsics,
    valid_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    v_idx, u_idx = np.nonzero(valid_mask)
    if v_idx.size == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 2), dtype=np.int32)
    z = depth_m[v_idx, u_idx]
    points = intrinsics.back_project(u_idx.astype(np.float32), v_idx.astype(np.float32), z.astype(np.float32))
    pixels = np.stack((v_idx, u_idx), axis=-1).astype(np.int32)
    return points.astype(np.float32), pixels


def estimate_floor_plane(
    points_xyz: np.ndarray,
    iterations: int = 128,
    distance_threshold_m: float = 0.03,
    rng_seed: int = 42,
) -> FloorPlane | None:
    if len(points_xyz) < 32:
        return None

    rng = np.random.default_rng(rng_seed)
    best_support = 0
    best_plane: FloorPlane | None = None

    for _ in range(iterations):
        sample_ids = rng.choice(len(points_xyz), size=3, replace=False)
        p0, p1, p2 = points_xyz[sample_ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if normal[1] > 0.0:
            normal = -normal
        offset = -float(np.dot(normal, p0))
        distances = np.abs(points_xyz @ normal + offset)
        support = int(np.count_nonzero(distances < distance_threshold_m))
        if support > best_support:
            best_support = support
            best_plane = FloorPlane(normal=normal.astype(np.float32), offset=offset, support=support)

    return best_plane


class RunningBackgroundModel:
    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.depth: np.ndarray | None = None
        self.valid: np.ndarray | None = None
        self.frames_seen = 0

    def update(self, depth_m: np.ndarray, valid_mask: np.ndarray) -> None:
        if self.depth is None or self.valid is None:
            self.depth = depth_m.copy()
            self.valid = valid_mask.copy()
            self.frames_seen = 1
            return

        same_support = self.valid & valid_mask
        self.depth[same_support] = (
            self.alpha * depth_m[same_support] + (1.0 - self.alpha) * self.depth[same_support]
        )
        new_support = valid_mask & ~self.valid
        self.depth[new_support] = depth_m[new_support]
        self.valid |= valid_mask
        self.frames_seen += 1

    def foreground_mask(
        self,
        depth_m: np.ndarray,
        valid_mask: np.ndarray,
        threshold_m: float,
    ) -> np.ndarray:
        if self.depth is None or self.valid is None:
            return valid_mask.copy()
        delta = np.abs(depth_m - self.depth)
        return valid_mask & (~self.valid | (delta > threshold_m))


def _binary_dilate(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        views = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        result = np.logical_or.reduce(views)
    return result


def _binary_erode(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    result = mask.copy()
    for _ in range(iterations):
        padded = np.pad(result, 1, constant_values=False)
        views = [
            padded[0:-2, 0:-2],
            padded[0:-2, 1:-1],
            padded[0:-2, 2:],
            padded[1:-1, 0:-2],
            padded[1:-1, 1:-1],
            padded[1:-1, 2:],
            padded[2:, 0:-2],
            padded[2:, 1:-1],
            padded[2:, 2:],
        ]
        result = np.logical_and.reduce(views)
    return result


def clean_mask(mask: np.ndarray) -> np.ndarray:
    opened = _binary_dilate(_binary_erode(mask, iterations=1), iterations=1)
    return _binary_erode(_binary_dilate(opened, iterations=1), iterations=1)


def connected_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for y in range(height):
        for x in range(width):
            if not mask[y, x] or visited[y, x]:
                continue
            stack = [(y, x)]
            pixels: list[tuple[int, int]] = []
            visited[y, x] = True
            while stack:
                cy, cx = stack.pop()
                pixels.append((cy, cx))
                y0 = max(cy - 1, 0)
                y1 = min(cy + 2, height)
                x0 = max(cx - 1, 0)
                x1 = min(cx + 2, width)
                for ny in range(y0, y1):
                    for nx in range(x0, x1):
                        if mask[ny, nx] and not visited[ny, nx]:
                            visited[ny, nx] = True
                            stack.append((ny, nx))
            component = np.zeros_like(mask, dtype=bool)
            ys, xs = zip(*pixels)
            component[np.asarray(ys), np.asarray(xs)] = True
            components.append(component)
    return components


def component_roi(mask: np.ndarray, margin_px: int, frame_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    y0 = max(int(ys.min()) - margin_px, 0)
    x0 = max(int(xs.min()) - margin_px, 0)
    y1 = min(int(ys.max()) + margin_px + 1, frame_shape[0])
    x1 = min(int(xs.max()) + margin_px + 1, frame_shape[1])

    h = y1 - y0
    w = x1 - x0
    side = max(h, w)
    cy = (y0 + y1) // 2
    cx = (x0 + x1) // 2
    half = side // 2
    y0 = max(cy - half, 0)
    x0 = max(cx - half, 0)
    y1 = min(y0 + side, frame_shape[0])
    x1 = min(x0 + side, frame_shape[1])
    y0 = max(y1 - side, 0)
    x0 = max(x1 - side, 0)
    return (x0, y0, x1, y1)


def _smooth_roi(
    current: tuple[int, int, int, int],
    previous: tuple[int, int, int, int] | None,
    alpha: float,
) -> tuple[int, int, int, int]:
    if previous is None:
        return current
    cur = np.asarray(current, dtype=np.float32)
    prev = np.asarray(previous, dtype=np.float32)
    blended = alpha * cur + (1.0 - alpha) * prev
    return tuple(int(round(v)) for v in blended)


def _roi_center_size(roi: tuple[int, int, int, int]) -> tuple[np.ndarray, np.ndarray]:
    x0, y0, x1, y1 = roi
    center = np.asarray([(x0 + x1) / 2.0, (y0 + y1) / 2.0], dtype=np.float32)
    size = np.asarray([max(x1 - x0, 1), max(y1 - y0, 1)], dtype=np.float32)
    return center, size


def _roi_area_fraction(roi: tuple[int, int, int, int], frame_shape: tuple[int, int]) -> float:
    x0, y0, x1, y1 = roi
    area = max(x1 - x0, 0) * max(y1 - y0, 0)
    frame_area = max(frame_shape[0] * frame_shape[1], 1)
    return float(area / frame_area)


def _roi_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    inter = max(ix1 - ix0, 0) * max(iy1 - iy0, 0)
    area_a = max(ax1 - ax0, 0) * max(ay1 - ay0, 0)
    area_b = max(bx1 - bx0, 0) * max(by1 - by0, 0)
    union = area_a + area_b - inter
    return float(inter / max(union, 1))


def _bump_reject(reject_counts: dict[str, int], reason: str) -> None:
    reject_counts[reason] = reject_counts.get(reason, 0) + 1


def _roi_from_center_size(
    center: np.ndarray,
    size: np.ndarray,
    frame_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    h, w = frame_shape
    size = np.clip(size, np.asarray([4.0, 4.0], dtype=np.float32), np.asarray([w, h], dtype=np.float32))
    x0 = int(round(float(center[0] - size[0] / 2.0)))
    y0 = int(round(float(center[1] - size[1] / 2.0)))
    x1 = int(round(float(center[0] + size[0] / 2.0)))
    y1 = int(round(float(center[1] + size[1] / 2.0)))
    x0 = min(max(x0, 0), max(w - 1, 0))
    y0 = min(max(y0, 0), max(h - 1, 0))
    x1 = min(max(x1, x0 + 1), w)
    y1 = min(max(y1, y0 + 1), h)
    if x1 - x0 < int(size[0]) and x0 == 0:
        x1 = min(int(round(size[0])), w)
    if y1 - y0 < int(size[1]) and y0 == 0:
        y1 = min(int(round(size[1])), h)
    if x1 - x0 < int(size[0]) and x1 == w:
        x0 = max(w - int(round(size[0])), 0)
    if y1 - y0 < int(size[1]) and y1 == h:
        y0 = max(h - int(round(size[1])), 0)
    return (x0, y0, x1, y1)


def _stabilize_roi(
    current: tuple[int, int, int, int],
    previous: tuple[int, int, int, int] | None,
    alpha: float,
    frame_shape: tuple[int, int],
    max_center_step_px: float,
    max_size_step_fraction: float,
) -> tuple[int, int, int, int]:
    if previous is None:
        return current

    smoothed = _smooth_roi(current, previous, alpha)
    prev_center, prev_size = _roi_center_size(previous)
    cur_center, cur_size = _roi_center_size(smoothed)

    center_delta = cur_center - prev_center
    center_step = float(np.linalg.norm(center_delta))
    if center_step > max_center_step_px:
        cur_center = prev_center + center_delta * (max_center_step_px / max(center_step, 1e-6))

    size_limit = max(float(max_size_step_fraction), 0.0)
    min_size = prev_size * (1.0 - size_limit)
    max_size = prev_size * (1.0 + size_limit)
    cur_size = np.clip(cur_size, min_size, max_size)
    return _roi_from_center_size(cur_center, cur_size, frame_shape)


def _component_stats(
    component_mask: np.ndarray,
    frame: DepthFrame,
) -> tuple[np.ndarray, np.ndarray]:
    points, _ = depth_to_point_cloud(frame.depth_m, frame.intrinsics, component_mask)
    if len(points) == 0:
        return np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    centroid = points.mean(axis=0)
    extent = points.max(axis=0) - points.min(axis=0)
    return centroid.astype(np.float32), extent.astype(np.float32)


class GeometricPersonTracker:
    def __init__(self, config: GeometryConfig | None = None) -> None:
        self.config = config or GeometryConfig()
        self.floor_plane: FloorPlane | None = None
        self.background = RunningBackgroundModel()
        self.last_floor_calibration: FloorCalibrationDiagnostics | None = None
        self.last_diagnostics = TrackingDiagnostics()
        self._previous_roi: tuple[int, int, int, int] | None = None
        self._previous_centroid: np.ndarray | None = None
        self._previous_extent: np.ndarray | None = None
        self._previous_mask: np.ndarray | None = None
        self._missed_tracks = 0
        self._cluster_counter = 0

    def _collect_floor_points(
        self,
        frames: Iterable[DepthFrame],
    ) -> tuple[list[np.ndarray], FloorCalibrationDiagnostics]:
        stacked_points: list[np.ndarray] = []
        diagnostics = FloorCalibrationDiagnostics()
        conf_p95_values: list[float] = []
        conf_max_values: list[float] = []
        for frame in frames:
            diagnostics.frames_total += 1
            masks = floor_candidate_masks(frame, self.config)
            depth_valid = masks.depth_valid

            conf_values = frame.confidence[depth_valid]
            if conf_values.size:
                conf_p95_values.append(float(np.percentile(conf_values, 95)))
                conf_max_values.append(float(np.max(conf_values)))

            diagnostics.lower_band_points_total += int(np.count_nonzero(masks.lower_candidates))
            diagnostics.strict_points_total += int(np.count_nonzero(masks.strict_candidates))

            used_mask = masks.used_mask
            if np.array_equal(used_mask, masks.lower_candidates) and np.count_nonzero(used_mask):
                diagnostics.frames_relaxed_confidence += 1

            if np.array_equal(used_mask, depth_valid) and np.count_nonzero(used_mask):
                diagnostics.frames_full_frame_fallback += 1

            points, _ = depth_to_point_cloud(frame.depth_m, frame.intrinsics, used_mask)
            if len(points):
                stacked_points.append(points)
                diagnostics.frames_with_points += 1
                diagnostics.used_points_total += int(len(points))
            self.background.update(frame.depth_m, depth_valid)

        if conf_p95_values:
            diagnostics.confidence_p95 = float(np.median(conf_p95_values))
        if conf_max_values:
            diagnostics.confidence_max = float(np.max(conf_max_values))
        return stacked_points, diagnostics

    def calibrate_floor(
        self,
        frames: Iterable[DepthFrame],
        with_diagnostics: bool = False,
    ) -> FloorPlane | tuple[FloorPlane | None, FloorCalibrationDiagnostics] | None:
        stacked_points, diagnostics = self._collect_floor_points(frames)
        self.last_floor_calibration = diagnostics
        if not stacked_points:
            return (None, diagnostics) if with_diagnostics else None
        cloud = np.concatenate(stacked_points, axis=0)
        plane = estimate_floor_plane(
            cloud,
            distance_threshold_m=self.config.floor_distance_threshold_m,
        )
        self.floor_plane = plane
        return (plane, diagnostics) if with_diagnostics else plane

    def _foreground_mask(self, frame: DepthFrame) -> np.ndarray:
        valid = (
            frame.valid_mask
            & (frame.depth_m >= self.config.min_depth_m)
            & (frame.depth_m <= self.config.max_depth_m)
            & (frame.confidence >= self.config.confidence_threshold)
        )
        foreground = self.background.foreground_mask(
            frame.depth_m,
            valid,
            threshold_m=self.config.background_threshold_m,
        )
        if self.floor_plane is None:
            return clean_mask(foreground)

        points, pixels = depth_to_point_cloud(frame.depth_m, frame.intrinsics, foreground)
        if len(points) == 0:
            return np.zeros_like(foreground, dtype=bool)

        distances = self.floor_plane.signed_distance(points)
        keep = np.abs(distances) > self.config.floor_distance_threshold_m
        out = np.zeros_like(foreground, dtype=bool)
        kept_pixels = pixels[keep]
        if len(kept_pixels):
            out[kept_pixels[:, 0], kept_pixels[:, 1]] = True
        return clean_mask(out)

    def _tracking_gate_active(self) -> bool:
        return (
            self._previous_centroid is not None
            and self._missed_tracks < self.config.track_reacquire_after_frames
        )

    def _motion_penalty(self, centroid: np.ndarray) -> float | None:
        if not self._tracking_gate_active() or self._previous_centroid is None:
            return 0.0
        depth_delta = abs(float(centroid[2] - self._previous_centroid[2]))
        if depth_delta > self.config.track_max_depth_jump_m:
            return None
        return depth_delta * self.config.track_depth_jump_penalty

    def _candidate_roi_metrics(
        self,
        component: np.ndarray,
        frame: DepthFrame,
    ) -> tuple[tuple[int, int, int, int], float, float]:
        raw_roi = component_roi(component, self.config.roi_margin_px, frame.shape)
        roi_area_frac = _roi_area_fraction(raw_roi, frame.shape)
        roi_area = max((raw_roi[2] - raw_roi[0]) * (raw_roi[3] - raw_roi[1]), 1)
        fill_ratio = int(component.sum()) / roi_area
        return raw_roi, roi_area_frac, float(fill_ratio)

    def _candidate_reject_reason(
        self,
        raw_roi: tuple[int, int, int, int],
        roi_area_frac: float,
        fill_ratio: float,
        frame_shape: tuple[int, int],
    ) -> str | None:
        if roi_area_frac > self.config.max_candidate_roi_fraction and fill_ratio < self.config.full_height_min_fill_ratio:
            return "roi_too_large"

        height = frame_shape[0]
        roi_height_frac = (raw_roi[3] - raw_roi[1]) / max(height, 1)
        touches_full_height = raw_roi[1] <= 0 and raw_roi[3] >= height
        if (
            (touches_full_height or roi_height_frac >= self.config.full_height_roi_fraction)
            and fill_ratio < self.config.full_height_min_fill_ratio
        ):
            return "full_height_low_fill"

        if self._tracking_gate_active() and self._previous_roi is not None:
            if _roi_iou(raw_roi, self._previous_roi) < self.config.min_previous_roi_iou:
                return "low_previous_iou"
        return None

    def _finalize_track(
        self,
        component: np.ndarray,
        centroid: np.ndarray,
        extent: np.ndarray,
        score: float,
        frame: DepthFrame,
        source: str,
        raw_roi: tuple[int, int, int, int],
        candidate_count: int,
        reject_counts: dict[str, int],
    ) -> TrackedPerson:
        final_roi = _stabilize_roi(
            raw_roi,
            self._previous_roi,
            self.config.track_roi_alpha,
            frame.shape,
            self.config.track_max_center_step_px,
            self.config.track_max_size_step_fraction,
        )
        self._previous_roi = final_roi
        self._previous_centroid = centroid.copy()
        self._previous_extent = extent.copy()
        self._previous_mask = component.copy()
        self._missed_tracks = 0
        self._cluster_counter += 1
        self.last_diagnostics = TrackingDiagnostics(
            track_source=source,
            held=False,
            roi_area_frac=_roi_area_fraction(final_roi, frame.shape),
            mask_pixels=int(component.sum()),
            candidate_count=candidate_count,
            reject_counts=dict(reject_counts),
            raw_roi_px=raw_roi,
            final_roi_px=final_roi,
        )
        return TrackedPerson(
            roi_px=final_roi,
            cluster_id=self._cluster_counter,
            centroid_xyz=centroid,
            extent_xyz=extent,
            quality=max(float(score), 0.0),
            mask=component,
        )

    def _hold_previous_track(
        self,
        frame: DepthFrame,
        candidate_count: int = 0,
        reject_counts: dict[str, int] | None = None,
    ) -> TrackedPerson | None:
        reject_counts = reject_counts or {}
        if (
            self._previous_roi is None
            or self._previous_centroid is None
            or self._previous_extent is None
            or self._missed_tracks >= self.config.track_hold_frames
        ):
            self._missed_tracks += 1
            self.last_diagnostics = TrackingDiagnostics(
                track_source="none",
                held=False,
                candidate_count=candidate_count,
                reject_counts=dict(reject_counts),
            )
            return None

        self._missed_tracks += 1
        self._cluster_counter += 1
        mask = np.zeros(frame.shape, dtype=bool)
        if self._previous_mask is not None and self._previous_mask.shape == frame.shape:
            mask = self._previous_mask.copy()
        self.last_diagnostics = TrackingDiagnostics(
            track_source="held",
            held=True,
            roi_area_frac=_roi_area_fraction(self._previous_roi, frame.shape),
            mask_pixels=int(np.count_nonzero(mask)),
            candidate_count=candidate_count,
            reject_counts=dict(reject_counts),
            raw_roi_px=self._previous_roi,
            final_roi_px=self._previous_roi,
        )
        return TrackedPerson(
            roi_px=self._previous_roi,
            cluster_id=self._cluster_counter,
            centroid_xyz=self._previous_centroid.copy(),
            extent_xyz=self._previous_extent.copy(),
            quality=0.0,
            mask=mask,
        )

    def _score_component(
        self,
        component: np.ndarray,
        centroid: np.ndarray,
        extent: np.ndarray,
        frame: DepthFrame,
        reject_counts: dict[str, int],
    ) -> tuple[float, tuple[int, int, int, int] | None]:
        pixels = int(component.sum())
        if pixels < self.config.min_component_pixels:
            _bump_reject(reject_counts, "too_few_pixels")
            return -1.0, None

        height = float(abs(extent[1]))
        width = float(max(abs(extent[0]), abs(extent[2])))
        if not (self.config.min_human_height_m <= height <= self.config.max_human_height_m):
            _bump_reject(reject_counts, "bad_height")
            return -1.0, None
        if not (self.config.min_human_width_m <= width <= self.config.max_human_width_m):
            _bump_reject(reject_counts, "bad_width")
            return -1.0, None

        raw_roi, roi_area_frac, fill_ratio = self._candidate_roi_metrics(component, frame)
        reject_reason = self._candidate_reject_reason(raw_roi, roi_area_frac, fill_ratio, frame.shape)
        if reject_reason is not None:
            _bump_reject(reject_counts, reject_reason)
            return -1.0, None

        motion_penalty = self._motion_penalty(centroid)
        if motion_penalty is None:
            _bump_reject(reject_counts, "depth_jump")
            return -1.0, None

        score = float(pixels)
        score -= motion_penalty
        score -= roi_area_frac * 400.0
        if self._previous_roi is not None:
            px, py, qx, qy = self._previous_roi
            prev_center = np.asarray([(px + qx) / 2.0, (py + qy) / 2.0], dtype=np.float32)
            ys, xs = np.nonzero(component)
            comp_center = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
            score -= float(np.linalg.norm(comp_center - prev_center)) * 2.0
        score -= float(abs(centroid[2])) * 4.0
        return score, raw_roi

    def _fallback_nearest_track(self, frame: DepthFrame) -> TrackedPerson | None:
        valid = (
            frame.valid_mask
            & (frame.depth_m >= self.config.min_depth_m)
            & (frame.depth_m <= self.config.max_depth_m)
        )
        if not np.any(valid):
            self.last_diagnostics = TrackingDiagnostics(
                track_source="none",
                reject_counts={"no_valid_depth": 1},
            )
            return None

        depths = frame.depth_m[valid]
        best_component: np.ndarray | None = None
        best_score = -np.inf
        best_centroid = np.zeros(3, dtype=np.float32)
        best_extent = np.zeros(3, dtype=np.float32)
        best_raw_roi: tuple[int, int, int, int] | None = None
        candidate_count = 0
        reject_counts: dict[str, int] = {}

        h, w = frame.shape
        frame_area = float(h * w)
        base = float(self.config.fallback_near_percentile)
        percentiles = sorted(
            {
                max(1.0, base * 0.25),
                max(1.0, base * 0.5),
                base,
                min(55.0, base * 1.5),
                25.0,
                35.0,
                45.0,
            }
        )
        windows = sorted(
            {
                max(0.12, min(0.25, self.config.fallback_depth_window_m)),
                self.config.fallback_depth_window_m,
                min(0.65, self.config.fallback_depth_window_m * 1.5),
            }
        )

        for percentile in percentiles:
            anchor_depth = float(np.percentile(depths, percentile))
            for depth_window_m in windows:
                lower = max(anchor_depth - 0.05, self.config.min_depth_m)
                upper = min(anchor_depth + depth_window_m, self.config.max_depth_m)
                mask = clean_mask(valid & (frame.depth_m >= lower) & (frame.depth_m <= upper))
                if not mask.any():
                    _bump_reject(reject_counts, "empty_depth_slice")
                    continue

                for component in connected_components(mask):
                    candidate_count += 1
                    pixels = int(component.sum())
                    if pixels < self.config.min_component_pixels:
                        _bump_reject(reject_counts, "too_few_pixels")
                        continue

                    raw_roi, roi_area_frac, fill_ratio = self._candidate_roi_metrics(component, frame)
                    reject_reason = self._candidate_reject_reason(raw_roi, roi_area_frac, fill_ratio, frame.shape)
                    if reject_reason is not None:
                        _bump_reject(reject_counts, reject_reason)
                        continue

                    ys, xs = np.nonzero(component)
                    raw_roi_area = float(
                        max((raw_roi[2] - raw_roi[0]) * (raw_roi[3] - raw_roi[1]), 1)
                    )
                    component_fraction = pixels / frame_area
                    roi_fraction = raw_roi_area / frame_area
                    if component_fraction > self.config.fallback_max_component_fraction:
                        _bump_reject(reject_counts, "too_many_pixels")
                        continue
                    if roi_fraction > self.config.fallback_max_roi_fraction and fill_ratio < self.config.full_height_min_fill_ratio:
                        _bump_reject(reject_counts, "fallback_roi_too_large")
                        continue
                    if (raw_roi[2] - raw_roi[0]) >= int(w * 0.94) and (raw_roi[3] - raw_roi[1]) >= int(h * 0.94):
                        _bump_reject(reject_counts, "near_full_frame")
                        continue

                    centroid, extent = _component_stats(component, frame)
                    motion_penalty = self._motion_penalty(centroid)
                    if motion_penalty is None:
                        _bump_reject(reject_counts, "depth_jump")
                        continue
                    median_depth = float(np.median(frame.depth_m[component]))
                    compact_pixels = pixels * min(fill_ratio * 3.0, 1.0)
                    score = float(compact_pixels) - median_depth * 35.0 - roi_fraction * 250.0 - motion_penalty
                    if self._previous_roi is not None:
                        px, py, qx, qy = self._previous_roi
                        prev_center = np.asarray([(px + qx) / 2.0, (py + qy) / 2.0], dtype=np.float32)
                        comp_center = np.asarray([xs.mean(), ys.mean()], dtype=np.float32)
                        score -= float(np.linalg.norm(comp_center - prev_center)) * 1.5
                    if score > best_score:
                        best_component = component
                        best_score = score
                        best_centroid = centroid
                        best_extent = extent
                        best_raw_roi = raw_roi

        if best_component is None:
            self.last_diagnostics = TrackingDiagnostics(
                track_source="none",
                candidate_count=candidate_count,
                reject_counts=reject_counts,
            )
            return None

        assert best_raw_roi is not None
        return self._finalize_track(
            best_component,
            best_centroid,
            best_extent,
            best_score,
            frame,
            source="fallback",
            raw_roi=best_raw_roi,
            candidate_count=candidate_count,
            reject_counts=reject_counts,
        )

    def update(self, frame: DepthFrame) -> TrackedPerson | None:
        mask = self._foreground_mask(frame)
        if not mask.any():
            fallback = self._fallback_nearest_track(frame)
            if fallback is not None:
                self.background.update(frame.depth_m, frame.valid_mask)
                return fallback
            fallback_diagnostics = self.last_diagnostics
            held = self._hold_previous_track(
                frame,
                candidate_count=fallback_diagnostics.candidate_count,
                reject_counts=fallback_diagnostics.reject_counts,
            )
            if held is not None:
                self.background.update(frame.depth_m, frame.valid_mask)
                return held
            self.background.update(frame.depth_m, frame.valid_mask)
            return None

        components = connected_components(mask)
        best_component: np.ndarray | None = None
        best_centroid = np.zeros(3, dtype=np.float32)
        best_extent = np.zeros(3, dtype=np.float32)
        best_raw_roi: tuple[int, int, int, int] | None = None
        best_score = -np.inf
        candidate_count = len(components)
        reject_counts: dict[str, int] = {}

        for component in components:
            centroid, extent = _component_stats(component, frame)
            score, raw_roi = self._score_component(component, centroid, extent, frame, reject_counts)
            if score > best_score:
                best_component = component
                best_centroid = centroid
                best_extent = extent
                best_raw_roi = raw_roi
                best_score = score

        if best_component is None or best_score < 0.0:
            fallback = self._fallback_nearest_track(frame)
            if fallback is not None:
                self.background.update(frame.depth_m, frame.valid_mask)
                return fallback
            fallback_diagnostics = self.last_diagnostics
            merged_reject_counts = dict(reject_counts)
            for reason, count in fallback_diagnostics.reject_counts.items():
                merged_reject_counts[reason] = merged_reject_counts.get(reason, 0) + count
            held = self._hold_previous_track(
                frame,
                candidate_count=candidate_count + fallback_diagnostics.candidate_count,
                reject_counts=merged_reject_counts,
            )
            if held is not None:
                self.background.update(frame.depth_m, frame.valid_mask)
                return held
            self.background.update(frame.depth_m, frame.valid_mask)
            return None

        assert best_raw_roi is not None
        return self._finalize_track(
            best_component,
            best_centroid,
            best_extent,
            best_score,
            frame,
            source="foreground",
            raw_roi=best_raw_roi,
            candidate_count=candidate_count,
            reject_counts=reject_counts,
        )


def nearest_resize(image: np.ndarray, output_shape: tuple[int, int]) -> np.ndarray:
    in_h, in_w = image.shape[:2]
    out_h, out_w = output_shape
    y_idx = np.clip(np.round(np.linspace(0, in_h - 1, out_h)).astype(np.int32), 0, in_h - 1)
    x_idx = np.clip(np.round(np.linspace(0, in_w - 1, out_w)).astype(np.int32), 0, in_w - 1)
    if image.ndim == 2:
        return image[y_idx][:, x_idx]
    return image[y_idx][:, x_idx, :]
