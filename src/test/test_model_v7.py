import os
import sys
import unittest

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "..", "src"))

TORCH_IMPORT_ERROR = None
try:
    import torch
    from model_v7.dataset import apply_affine_to_keypoints, build_affine_matrix
    from model_v7.model import EdgePoseUNetV7
except ModuleNotFoundError as exc:
    TORCH_IMPORT_ERROR = exc
    torch = None
    apply_affine_to_keypoints = None
    build_affine_matrix = None
    EdgePoseUNetV7 = None


@unittest.skipIf(TORCH_IMPORT_ERROR is not None, f"torch-dependent tests skipped: {TORCH_IMPORT_ERROR}")
class ModelV7Tests(unittest.TestCase):
    def test_forward_shapes(self):
        model = EdgePoseUNetV7(in_ch=3, n_kpts=17, width_mult=1.15)
        inputs = torch.randn(2, 3, 240, 240)
        heat_logits = model(inputs)
        self.assertEqual(tuple(heat_logits.shape), (2, 17, 240, 240))

        heat_logits, vis_logits = model(inputs, return_aux=True)
        self.assertEqual(tuple(heat_logits.shape), (2, 17, 240, 240))
        self.assertEqual(tuple(vis_logits.shape), (2, 17))

    def test_affine_translation_matches_keypoints(self):
        keypoints = np.array([[10.0, 20.0], [50.0, 60.0]], dtype=np.float32)
        matrix = build_affine_matrix(width=240, height=180, angle=0.0, scale=1.0, tx=5.0, ty=-3.0)
        transformed = apply_affine_to_keypoints(keypoints, matrix)
        expected = np.array([[15.0, 17.0], [55.0, 57.0]], dtype=np.float32)
        np.testing.assert_allclose(transformed, expected, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
