from __future__ import annotations

import numpy as np

from macchiato.evaluation.evaluate_generalization import select_highest_confidence_pose, unpad_keypoints
from macchiato.evaluation.pose_metrics import LatencyAccumulator, PoseMetricAccumulator


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
        self.boxes = type("Boxes", (), {"conf": FakeTensor([0.2, 0.9]), "__len__": lambda self: 2})()
        poses = np.stack([np.zeros((17, 2)), np.ones((17, 2)) * 7.0])
        self.keypoints = type("Keypoints", (), {"xy": FakeTensor(poses)})()


def test_metrics_count_missed_detection_as_pck_failure() -> None:
    target = np.zeros((17, 2), dtype=np.float32)
    valid = np.ones(17, dtype=bool)
    metrics = PoseMetricAccumulator()
    metrics.update(target.copy(), target, valid)
    metrics.update(None, target, valid)
    summary = metrics.summarize()
    assert summary["detection_coverage_percent"] == 50.0
    assert summary["pck5_percent"] == 50.0
    assert summary["pck10_percent"] == 50.0
    assert summary["mpjpe_detected_pixels"] == 0.0


def test_highest_confidence_pose_and_unpadding() -> None:
    selected = select_highest_confidence_pose(FakeResult())
    np.testing.assert_array_equal(selected, np.ones((17, 2)) * 7.0)
    padded = np.array([[20.0, 40.0], [30.0, 50.0]], dtype=np.float32)
    np.testing.assert_array_equal(unpad_keypoints(padded, pad_left=0, pad_top=30), [[20.0, 10.0], [30.0, 20.0]])


def test_latency_summary_reports_distribution_and_throughput() -> None:
    latency = LatencyAccumulator()
    for elapsed_seconds in (0.010, 0.020, 0.030):
        latency.update(elapsed_seconds)

    summary = latency.summarize()
    assert summary["latency_samples"] == 3
    assert summary["latency_mean_ms"] == 20.0
    assert summary["latency_p50_ms"] == 20.0
    assert summary["latency_p95_ms"] == 29.0
    assert summary["latency_min_ms"] == 10.0
    assert summary["latency_max_ms"] == 30.0
    assert summary["throughput_fps"] == 50.0
