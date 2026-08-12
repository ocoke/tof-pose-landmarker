from __future__ import annotations

import numpy as np

from macchiato.live_preview import (
    PosePrediction,
    RollingLatency,
    draw_pose_overlay,
    extract_highest_confidence_prediction,
)


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, index):
        return FakeTensor(self.value[index])

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.value


class FakeResult:
    def __init__(self):
        boxes = type(
            "Boxes",
            (),
            {
                "conf": FakeTensor([0.2, 0.9]),
                "xyxy": FakeTensor([[0.0, 0.0, 10.0, 10.0], [10.0, 20.0, 100.0, 160.0]]),
                "__len__": lambda self: 2,
            },
        )()
        poses = np.stack([np.zeros((17, 2)), np.ones((17, 2)) * [50.0, 80.0]])
        confidences = np.stack([np.zeros(17), np.linspace(0.1, 0.9, 17)])
        keypoints = type(
            "Keypoints",
            (),
            {"xy": FakeTensor(poses), "conf": FakeTensor(confidences)},
        )()
        self.boxes = boxes
        self.keypoints = keypoints


def test_extract_highest_confidence_live_prediction() -> None:
    prediction = extract_highest_confidence_prediction(FakeResult())
    assert prediction is not None
    assert prediction.box_confidence == 0.9
    np.testing.assert_array_equal(prediction.box_xyxy, [10.0, 20.0, 100.0, 160.0])
    np.testing.assert_array_equal(prediction.keypoints, np.ones((17, 2)) * [50.0, 80.0])
    np.testing.assert_allclose(prediction.keypoint_confidences, np.linspace(0.1, 0.9, 17))


def test_draw_pose_overlay_changes_depth_visualization() -> None:
    image = np.zeros((180, 240, 3), dtype=np.uint8)
    points = np.array([[40.0 + index * 2, 50.0 + index * 3] for index in range(17)], dtype=np.float32)
    prediction = PosePrediction(points, np.ones(17, dtype=np.float32), np.array([30, 30, 100, 130]), 0.8)
    rendered = draw_pose_overlay(image, prediction)
    assert rendered.shape == image.shape
    assert np.count_nonzero(rendered) > 0
    assert np.count_nonzero(image) == 0


def test_rolling_latency_uses_fixed_window() -> None:
    latency = RollingLatency(window_size=2)
    latency.update(0.010)
    latency.update(0.020)
    latency.update(0.030)
    summary = latency.summarize()
    assert summary["samples"] == 2
    assert summary["mean_ms"] == 25.0
    assert summary["p50_ms"] == 25.0
    assert summary["p95_ms"] == 29.5
    assert summary["fps"] == 40.0
