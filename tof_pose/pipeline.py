from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .camera import ArducamCameraAdapter
from .filtering import ExponentialDepthFilter, OneEuroFilter
from .geometry import FloorCalibrationDiagnostics, GeometricPersonTracker, GeometryConfig, nearest_resize
from .inference import PoseEstimator
from .types import CameraIntrinsics, DepthFrame, FloorPlane, Pose2D, Pose3D, TrackedPerson


@dataclass(slots=True)
class PipelineConfig:
    roi_size: tuple[int, int] = (128, 128)
    joint_patch_radius: int = 3
    min_joint_confidence: float = 0.1
    depth_ema_alpha: float = 0.35
    joint_filter_min_cutoff: float = 1.2
    joint_filter_beta: float = 0.03
    geometry: GeometryConfig = field(default_factory=GeometryConfig)


class HybridToFPosePipeline:
    def __init__(
        self,
        camera: ArducamCameraAdapter | None,
        pose_estimator: PoseEstimator,
        config: PipelineConfig | None = None,
    ) -> None:
        self.camera = camera
        self.pose_estimator = pose_estimator
        self.config = config or PipelineConfig()
        self.depth_filter = ExponentialDepthFilter(alpha=self.config.depth_ema_alpha)
        self.tracker = GeometricPersonTracker(config=self.config.geometry)
        self.pose2d_filter = OneEuroFilter(
            min_cutoff=self.config.joint_filter_min_cutoff,
            beta=self.config.joint_filter_beta,
        )
        self.pose3d_filter = OneEuroFilter(
            min_cutoff=self.config.joint_filter_min_cutoff,
            beta=self.config.joint_filter_beta,
        )

    def calibrate_floor(
        self,
        frames: list[DepthFrame],
        with_diagnostics: bool = False,
    ) -> FloorPlane | tuple[FloorPlane | None, FloorCalibrationDiagnostics] | None:
        return self.tracker.calibrate_floor(frames, with_diagnostics=with_diagnostics)

    def load_floor_plane(self, path: str | Path) -> FloorPlane:
        payload = json.loads(Path(path).read_text())
        plane = FloorPlane.from_json(payload)
        self.tracker.floor_plane = plane
        return plane

    def save_floor_plane(self, path: str | Path) -> None:
        if self.tracker.floor_plane is None:
            raise RuntimeError("No floor plane is available to save")
        Path(path).write_text(json.dumps(self.tracker.floor_plane.to_json(), indent=2))

    def process_frame(self, frame: DepthFrame) -> dict[str, Any] | None:
        smoothed_depth = self.depth_filter.update(frame.depth_m, frame.valid_mask)
        smoothed_frame = DepthFrame(
            depth_m=smoothed_depth,
            amplitude=frame.amplitude,
            confidence=frame.confidence,
            valid_mask=frame.valid_mask,
            intrinsics=frame.intrinsics,
            timestamp=frame.timestamp,
        )
        track = self.tracker.update(smoothed_frame)
        if track is None:
            return None

        roi_tensor = self._build_roi_tensor(smoothed_frame, track)
        pose2d = self.pose_estimator.predict(roi_tensor, track.roi_px)
        pose2d = self._filter_pose2d(pose2d, frame.timestamp)
        pose3d = self._lift_to_3d(smoothed_frame, pose2d)
        pose3d = self._filter_pose3d(pose3d, frame.timestamp)
        return {
            "track": track,
            "pose2d": pose2d,
            "pose3d": pose3d,
            "roi_tensor": roi_tensor,
        }

    def run_once(self) -> dict[str, Any] | None:
        if self.camera is None:
            raise RuntimeError("Pipeline has no camera attached")
        return self.process_frame(self.camera.read())

    def _build_roi_tensor(self, frame: DepthFrame, track: TrackedPerson) -> np.ndarray:
        x0, y0, x1, y1 = track.roi_px
        depth_roi = frame.depth_m[y0:y1, x0:x1]
        amp_roi = frame.amplitude[y0:y1, x0:x1]
        conf_roi = frame.confidence[y0:y1, x0:x1]

        resized_depth = nearest_resize(depth_roi, self.config.roi_size).astype(np.float32)
        resized_amp = nearest_resize(amp_roi, self.config.roi_size).astype(np.float32)
        resized_conf = nearest_resize(conf_roi, self.config.roi_size).astype(np.float32)

        depth_valid = resized_depth > 0.0
        if np.any(depth_valid):
            near = np.percentile(resized_depth[depth_valid], 10)
            far = np.percentile(resized_depth[depth_valid], 90)
        else:
            near, far = 0.0, 1.0
        depth_norm = np.clip((resized_depth - near) / max(far - near, 1e-6), 0.0, 1.0)

        amp_valid = np.isfinite(resized_amp)
        if np.any(amp_valid):
            low = np.percentile(resized_amp[amp_valid], 5)
            high = np.percentile(resized_amp[amp_valid], 95)
        else:
            low, high = 0.0, 1.0
        amp_norm = np.clip((resized_amp - low) / max(high - low, 1e-6), 0.0, 1.0)
        conf_norm = np.clip(resized_conf / max(np.max(resized_conf), 1e-6), 0.0, 1.0)
        return np.stack((depth_norm, conf_norm, amp_norm), axis=-1).astype(np.float32)

    def _filter_pose2d(self, pose2d: Pose2D, timestamp: float) -> Pose2D:
        filtered = self.pose2d_filter.update(pose2d.joints_uv, timestamp)
        return Pose2D(joints_uv=filtered, scores=pose2d.scores, roi_px=pose2d.roi_px)

    def _lift_to_3d(self, frame: DepthFrame, pose2d: Pose2D) -> Pose3D:
        x0, y0, x1, y1 = pose2d.roi_px
        roi_w = max(x1 - x0, 1)
        roi_h = max(y1 - y0, 1)
        model_w, model_h = self.config.roi_size[1], self.config.roi_size[0]
        scale_x = roi_w / max(model_w, 1)
        scale_y = roi_h / max(model_h, 1)

        joints_xyz = np.zeros((pose2d.joints_uv.shape[0], 3), dtype=np.float32)
        valid = np.zeros((pose2d.joints_uv.shape[0],), dtype=bool)
        for idx, (u_model, v_model) in enumerate(pose2d.joints_uv):
            if pose2d.scores[idx] < self.config.min_joint_confidence:
                continue
            u = int(round(x0 + u_model * scale_x))
            v = int(round(y0 + v_model * scale_y))
            z = _sample_joint_depth(frame.depth_m, frame.valid_mask, frame.confidence, u, v, self.config.joint_patch_radius)
            if z is None:
                continue
            xyz = frame.intrinsics.back_project(
                np.asarray([u], dtype=np.float32),
                np.asarray([v], dtype=np.float32),
                np.asarray([z], dtype=np.float32),
            )[0]
            joints_xyz[idx] = xyz
            valid[idx] = True
        return Pose3D(joints_xyz=joints_xyz, valid=valid, scores=pose2d.scores, roi_px=pose2d.roi_px)

    def _filter_pose3d(self, pose3d: Pose3D, timestamp: float) -> Pose3D:
        filtered = self.pose3d_filter.update(pose3d.joints_xyz, timestamp)
        return Pose3D(joints_xyz=filtered, valid=pose3d.valid, scores=pose3d.scores, roi_px=pose3d.roi_px)


def _sample_joint_depth(
    depth_m: np.ndarray,
    valid_mask: np.ndarray,
    confidence: np.ndarray,
    u: int,
    v: int,
    patch_radius: int,
) -> float | None:
    h, w = depth_m.shape
    y0 = max(v - patch_radius, 0)
    y1 = min(v + patch_radius + 1, h)
    x0 = max(u - patch_radius, 0)
    x1 = min(u + patch_radius + 1, w)
    depth_patch = depth_m[y0:y1, x0:x1]
    valid_patch = valid_mask[y0:y1, x0:x1] & (confidence[y0:y1, x0:x1] > 0.0)
    if not np.any(valid_patch):
        return None
    return float(np.median(depth_patch[valid_patch]))


class FakeCamera:
    def __init__(self, frames: list[DepthFrame]) -> None:
        self.frames = list(frames)
        self.index = 0

    def read(self) -> DepthFrame:
        frame = self.frames[min(self.index, len(self.frames) - 1)]
        self.index += 1
        return frame


def synthetic_frame(
    depth_m: np.ndarray,
    amplitude: np.ndarray | None = None,
    confidence: np.ndarray | None = None,
    intrinsics: CameraIntrinsics | None = None,
    timestamp: float = 0.0,
) -> DepthFrame:
    amplitude = amplitude if amplitude is not None else np.ones_like(depth_m, dtype=np.float32)
    confidence = confidence if confidence is not None else np.ones_like(depth_m, dtype=np.float32) * 100.0
    intrinsics = intrinsics or CameraIntrinsics(fx=120.0, fy=120.0, cx=depth_m.shape[1] / 2.0, cy=depth_m.shape[0] / 2.0)
    valid = np.isfinite(depth_m) & (depth_m > 0.0)
    return DepthFrame(
        depth_m=depth_m.astype(np.float32),
        amplitude=amplitude.astype(np.float32),
        confidence=confidence.astype(np.float32),
        valid_mask=valid,
        intrinsics=intrinsics,
        timestamp=timestamp,
    )
