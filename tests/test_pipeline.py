from __future__ import annotations

import unittest

import numpy as np

from tof_pose.camera import _depth_to_meters, _rotate_image
from tof_pose.geometry import GeometryConfig, floor_candidate_masks
from tof_pose.pipeline import FakeCamera, HybridToFPosePipeline, PipelineConfig, synthetic_frame
from tof_pose.types import Pose2D


class _StubEstimator:
    def predict(self, roi_tensor: np.ndarray, roi_px: tuple[int, int, int, int]) -> Pose2D:
        joints = np.zeros((15, 2), dtype=np.float32)
        scores = np.zeros((15,), dtype=np.float32)
        height, width = roi_tensor.shape[:2]
        joints[:] = np.asarray([width * 0.5, height * 0.5], dtype=np.float32)
        scores[:] = 0.95
        return Pose2D(joints_uv=joints, scores=scores, roi_px=roi_px)


class PipelineTests(unittest.TestCase):
    def test_pipeline_lifts_pose_to_3d(self) -> None:
        depth = np.full((48, 64), 3.9, dtype=np.float32)
        depth[8:44, 20:44] = 2.2
        amplitude = np.linspace(0.0, 1.0, depth.size, dtype=np.float32).reshape(depth.shape)
        confidence = np.ones_like(depth, dtype=np.float32) * 90.0
        frame = synthetic_frame(depth, amplitude=amplitude, confidence=confidence, timestamp=1.0)

        camera = FakeCamera([frame])
        pipeline = HybridToFPosePipeline(
            camera=camera,
            pose_estimator=_StubEstimator(),
            config=PipelineConfig(),
        )
        pipeline.tracker.background.update(np.full_like(depth, 3.9), np.ones_like(depth, dtype=bool))

        result = pipeline.run_once()
        self.assertIsNotNone(result)
        assert result is not None
        pose3d = result["pose3d"]
        self.assertGreaterEqual(int(np.count_nonzero(pose3d.valid)), 1)
        self.assertAlmostEqual(float(pose3d.joints_xyz[0, 2]), 2.2, delta=0.15)

    def test_floor_plane_round_trip(self) -> None:
        pipeline = HybridToFPosePipeline(camera=None, pose_estimator=_StubEstimator(), config=PipelineConfig())
        depth = np.full((48, 64), 3.9, dtype=np.float32)
        frame = synthetic_frame(depth, timestamp=1.0)
        plane = pipeline.calibrate_floor([frame])
        self.assertIsNotNone(plane)

    def test_floor_plane_low_confidence_fallback(self) -> None:
        pipeline = HybridToFPosePipeline(camera=None, pose_estimator=_StubEstimator(), config=PipelineConfig())
        depth = np.full((48, 64), 3.9, dtype=np.float32)
        confidence = np.ones_like(depth, dtype=np.float32) * 0.8
        frame = synthetic_frame(depth, confidence=confidence, timestamp=1.0)
        plane, diagnostics = pipeline.calibrate_floor([frame], with_diagnostics=True)
        self.assertIsNotNone(plane)
        self.assertGreaterEqual(diagnostics.frames_relaxed_confidence, 1)

    def test_floor_candidates_accept_millimeter_depth_after_conversion(self) -> None:
        depth_mm = np.full((48, 64), 3900.0, dtype=np.float32)
        frame = synthetic_frame(_depth_to_meters(depth_mm, "mm"), timestamp=1.0)
        masks = floor_candidate_masks(frame, GeometryConfig())
        self.assertGreater(int(np.count_nonzero(masks.depth_valid)), 0)
        self.assertGreater(int(np.count_nonzero(masks.used_mask)), 0)

    def test_rotate_image_180(self) -> None:
        image = np.asarray([[1, 2], [3, 4]], dtype=np.float32)
        rotated = _rotate_image(image, 180)
        np.testing.assert_array_equal(rotated, np.asarray([[4, 3], [2, 1]], dtype=np.float32))

    def test_depth_to_meters_auto_detects_millimeters(self) -> None:
        depth = np.asarray([[1000.0, 2000.0]], dtype=np.float32)
        np.testing.assert_allclose(_depth_to_meters(depth, "auto"), np.asarray([[1.0, 2.0]], dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
