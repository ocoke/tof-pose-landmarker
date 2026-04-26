from __future__ import annotations

import unittest

import numpy as np

from tof_pose.inference import _movenet_to_pose15, _resize_nearest


class InferenceTests(unittest.TestCase):
    def test_resize_nearest_matches_requested_shape(self) -> None:
        image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        resized = _resize_nearest(image, (4, 4))
        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(float(resized[0, 0]), 1.0)
        self.assertEqual(float(resized[-1, -1]), 4.0)

    def test_movenet_adapter_uses_roi_tensor_coordinates(self) -> None:
        keypoints = np.zeros((17, 3), dtype=np.float32)
        keypoints[:, 0] = 0.25
        keypoints[:, 1] = 0.5
        keypoints[:, 2] = 0.9
        pose = _movenet_to_pose15(keypoints, roi_px=(10, 20, 210, 180), output_shape=(128, 128))
        np.testing.assert_allclose(pose.joints_uv[0], np.asarray([64.0, 32.0], dtype=np.float32))
        self.assertEqual(pose.roi_px, (10, 20, 210, 180))


if __name__ == "__main__":
    unittest.main()
