from __future__ import annotations

import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
