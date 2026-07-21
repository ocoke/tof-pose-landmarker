from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from macchiato.adapters.arducam_adapter import (
    build_manifest_records,
    prepare_yolo_dataset,
    validate_yolo_dataset,
    write_manifests,
)


def _write_sample(project_root: Path, timestamp: int, valid_points: int = 17) -> None:
    raw = project_root / "data" / "raw" / "arducam"
    pose = project_root / "data" / "processed" / "arducam" / "pose_coco17"
    (raw / "depth").mkdir(parents=True, exist_ok=True)
    (raw / "confidence").mkdir(parents=True, exist_ok=True)
    pose.mkdir(parents=True, exist_ok=True)
    depth = np.arange(24, dtype=np.float32).reshape(4, 6) * 100.0
    np.save(raw / "depth" / f"{timestamp}.npy", depth)
    np.save(raw / "confidence" / f"{timestamp}.npy", np.ones_like(depth) * 100.0)
    points = [[float(index % 5), float(index % 3)] for index in range(17)]
    for index in range(valid_points, 17):
        points[index] = [-1.0, -1.0]
    (pose / f"{timestamp}.json").write_text(json.dumps({"keypoints": points}), encoding="utf-8")


def test_manifest_and_yolo_preparation_integration(tmp_path: Path) -> None:
    for timestamp in (100, 281, 462):
        _write_sample(tmp_path, timestamp)
    raw = tmp_path / "data" / "raw" / "arducam"
    pose = tmp_path / "data" / "processed" / "arducam" / "pose_coco17"
    records, diagnostics = build_manifest_records(
        raw_root=raw,
        pose_root=pose,
        project_root=tmp_path,
        expected_shape=(4, 6),
        minimum_valid_keypoints=12,
        train_scene_count=1,
        val_scene_count=1,
    )
    assert diagnostics["scenes"] == 3
    assert diagnostics["eligible_samples"] == 3
    write_manifests(records, tmp_path / "data" / "manifests")

    output = tmp_path / "data" / "processed" / "arducam" / "yolo_pose"
    counts = prepare_yolo_dataset(records, tmp_path, output)
    assert counts == {"train": 1, "val": 1, "test": 1}
    for split in counts:
        image_path = next((output / "images" / split).glob("*.png"))
        label_path = next((output / "labels" / split).glob("*.txt"))
        assert Image.open(image_path).size == (6, 4)
        assert len(label_path.read_text().split()) == 56
    dataset = yaml.safe_load((output / "dataset.yaml").read_text())
    assert dataset["kpt_shape"] == [17, 3]
    assert dataset["names"] == {0: "person"}
    assert validate_yolo_dataset(output) == counts
    with pytest.raises(FileExistsError):
        prepare_yolo_dataset(records, tmp_path, output)


def test_minimum_keypoint_rule_rejects_partial_sample(tmp_path: Path) -> None:
    for timestamp in (100, 281, 462):
        _write_sample(tmp_path, timestamp, valid_points=11 if timestamp == 100 else 17)
    records, diagnostics = build_manifest_records(
        raw_root=tmp_path / "data" / "raw" / "arducam",
        pose_root=tmp_path / "data" / "processed" / "arducam" / "pose_coco17",
        project_root=tmp_path,
        expected_shape=(4, 6),
        minimum_valid_keypoints=12,
        train_scene_count=1,
        val_scene_count=1,
    )
    assert diagnostics["rejected_samples"] == 1
    assert records[0].drop_reason == "fewer_than_12_valid_keypoints"
