"""Manifest-backed Arducam dataset for the U-Net baseline."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from macchiato.adapters.arducam_adapter import keypoint_valid_mask, load_keypoints
from macchiato.preprocessing.depth_normalization import normalize_confidence, normalize_depth
from macchiato.preprocessing.transforms import generate_heatmaps, random_basic_transform


class UnifiedPoseDataset(Dataset):
    """Load eligible samples from one split manifest."""

    def __init__(
        self,
        manifest_path: str | Path,
        project_root: str | Path,
        output_shape: tuple[int, int] = (240, 240),
        depth_maximum_mm: float = 4000.0,
        confidence_maximum: float = 350.0,
        heatmap_sigma: float = 4.0,
        augment: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.project_root = Path(project_root)
        self.output_shape = output_shape
        self.depth_maximum_mm = depth_maximum_mm
        self.confidence_maximum = confidence_maximum
        self.heatmap_sigma = heatmap_sigma
        self.augment = augment
        with self.manifest_path.open("r", newline="", encoding="utf-8") as handle:
            self.records = [row for row in csv.DictReader(handle) if row.get("eligible", "").lower() == "true"]
        if not self.records:
            raise ValueError(f"No eligible records in {self.manifest_path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        depth = np.load(self.project_root / record["depth_path"])
        confidence = np.load(self.project_root / record["confidence_path"])
        keypoints = load_keypoints(self.project_root / record["pose_path"])
        height, width = depth.shape
        valid_mask = keypoint_valid_mask(keypoints, width, height)

        normalized_depth = normalize_depth(depth, self.depth_maximum_mm)
        normalized_confidence = normalize_confidence(confidence, self.confidence_maximum)
        channels = np.stack(
            [normalized_depth, normalized_confidence, normalized_depth * normalized_confidence], axis=0
        ).astype(np.float32)

        if self.augment:
            channels, keypoints, valid_mask = random_basic_transform(channels, keypoints, valid_mask)

        output_height, output_width = self.output_shape
        pad_left = (output_width - width) // 2
        pad_right = output_width - width - pad_left
        pad_top = (output_height - height) // 2
        pad_bottom = output_height - height - pad_top
        if min(pad_left, pad_right, pad_top, pad_bottom) < 0:
            raise ValueError(f"Output shape {self.output_shape} is smaller than source {(height, width)}")
        channels = np.pad(channels, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        keypoints = keypoints.copy()
        keypoints[:, 0] += pad_left
        keypoints[:, 1] += pad_top
        valid_mask = (
            valid_mask
            & np.isfinite(keypoints).all(axis=1)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < output_width)
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < output_height)
        )
        heatmaps = generate_heatmaps(keypoints, valid_mask, self.output_shape, self.heatmap_sigma)
        metadata = {
            "timestamp": record["timestamp"],
            "scene_id": record["scene_id"],
            "split": record["split"],
            "pad_left": pad_left,
            "pad_top": pad_top,
            "source_width": width,
            "source_height": height,
        }
        return (
            torch.from_numpy(channels).float(),
            torch.from_numpy(heatmaps).float(),
            torch.from_numpy(keypoints.astype(np.float32)).float(),
            torch.from_numpy(valid_mask.astype(np.bool_)),
            metadata,
        )
