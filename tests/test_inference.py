from __future__ import annotations

import unittest

import numpy as np

from tof_pose.inference import _resize_nearest


class InferenceTests(unittest.TestCase):
    def test_resize_nearest_matches_requested_shape(self) -> None:
        image = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        resized = _resize_nearest(image, (4, 4))
        self.assertEqual(resized.shape, (4, 4))
        self.assertEqual(float(resized[0, 0]), 1.0)
        self.assertEqual(float(resized[-1, -1]), 4.0)


if __name__ == "__main__":
    unittest.main()
