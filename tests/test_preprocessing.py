from __future__ import annotations

import numpy as np

from macchiato.adapters.arducam_adapter import keypoints_to_yolo_label
from macchiato.preprocessing.depth_normalization import depth_to_rgb_uint8, normalize_depth
from macchiato.preprocessing.person_crop import padded_keypoint_box


def test_depth_normalization_is_fixed_and_nonfinite_values_become_zero() -> None:
    depth = np.array([[np.nan, -1.0, 0.0, 2000.0, 4000.0, 5000.0, np.inf]], dtype=np.float32)
    normalized = normalize_depth(depth, maximum_mm=4000.0)
    np.testing.assert_allclose(normalized, [[0.0, 0.0, 0.0, 0.5, 1.0, 1.0, 0.0]])
    image = depth_to_rgb_uint8(depth, maximum_mm=4000.0)
    assert image.shape == (1, 7, 3)
    assert image.dtype == np.uint8
    np.testing.assert_array_equal(image[..., 0], image[..., 1])
    np.testing.assert_array_equal(image[..., 1], image[..., 2])


def test_padded_box_and_yolo_pose_label_are_valid() -> None:
    keypoints = np.array([[20.0 + i, 30.0 + i] for i in range(17)], dtype=np.float32)
    keypoints[0] = [-1.0, 12.0]
    valid = np.ones(17, dtype=bool)
    valid[0] = False
    box = padded_keypoint_box(keypoints, valid, width=240, height=180, padding_fraction=0.10)
    assert 0 <= box[0] < box[2] <= 240
    assert 0 <= box[1] < box[3] <= 180

    tokens = keypoints_to_yolo_label(keypoints, width=240, height=180).split()
    assert len(tokens) == 56
    assert tokens[0] == "0"
    values = [float(value) for value in tokens[1:5]]
    assert all(0.0 <= value <= 1.0 for value in values)
    visibility = [int(tokens[7 + index * 3]) for index in range(17)]
    assert set(visibility) <= {0, 2}
    assert visibility[0] == 0
