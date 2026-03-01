import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms

from .manifest import build_or_load_manifest, get_records_for_split

FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
DEFAULT_CONF_HI = 350.0


def build_affine_matrix(width: int, height: int, angle: float, scale: float, tx: float, ty: float) -> np.ndarray:
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale).astype(np.float32)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    return matrix


def apply_affine_to_keypoints(keypoints: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((keypoints.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([keypoints.astype(np.float32), ones], axis=1)
    return (matrix @ homo.T).T.astype(np.float32)


def generate_heatmaps(keypoints: np.ndarray, valid_mask: np.ndarray, output_res: Tuple[int, int], sigma: float = 4.0) -> torch.Tensor:
    height, width = output_res
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    heatmaps = np.zeros((keypoints.shape[0], height, width), dtype=np.float32)
    for idx, (point, is_valid) in enumerate(zip(keypoints, valid_mask)):
        if not is_valid:
            continue
        x, y = point
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        heatmaps[idx] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))
    return torch.from_numpy(heatmaps)


class PoseDatasetV7(Dataset):
    def __init__(
        self,
        data_dir: str,
        records: Optional[List[Dict[str, Any]]] = None,
        split: Optional[str] = None,
        manifest_path: Optional[str] = None,
        scene_map_path: Optional[str] = None,
        output_res: Tuple[int, int] = (240, 240),
        augment: bool = False,
        min_valid_keypoints: int = 12,
        conf_hi: float = DEFAULT_CONF_HI,
        rebuild_manifest: bool = False,
    ):
        self.data_dir = data_dir
        self.depth_dir = os.path.join(data_dir, "depth")
        self.confidence_dir = os.path.join(data_dir, "confidence")
        self.pose_dir = os.path.join(data_dir, "pose_coco17")
        self.output_res = output_res
        self.augment = augment
        self.conf_hi = conf_hi

        if records is None:
            records = build_or_load_manifest(
                data_dir=data_dir,
                manifest_path=manifest_path,
                scene_map_path=scene_map_path,
                rebuild=rebuild_manifest,
            )
        if split:
            records = get_records_for_split(records, split)

        if augment and min_valid_keypoints > 0:
            records = [record for record in records if int(record.get("num_valid_kpts", 0)) >= min_valid_keypoints]

        self.records = sorted(records, key=lambda item: item["sample_id"])
        self.file_list = [record["sample_id"] for record in self.records]
        if not self.records:
            raise ValueError(f"No records available for split={split!r} in {data_dir}.")

    def __len__(self) -> int:
        return len(self.records)

    def _load_sample(self, sample_id: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        depth_map = np.load(os.path.join(self.depth_dir, f"{sample_id}.npy"))
        confidence_map = np.load(os.path.join(self.confidence_dir, f"{sample_id}.npy"))
        with open(os.path.join(self.pose_dir, f"{sample_id}.json"), "r", encoding="utf-8") as handle:
            keypoints = np.array(json.load(handle)["keypoints"], dtype=np.float32)
        return depth_map, confidence_map, keypoints

    def _normalize_inputs(self, depth_map: np.ndarray, confidence_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depth_map = np.nan_to_num(depth_map, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        confidence_map = np.nan_to_num(confidence_map, nan=0.0, posinf=self.conf_hi, neginf=0.0).astype(np.float32)

        depth_map = np.clip(depth_map, 0.0, 4000.0) / 4000.0
        confidence_map = np.clip(confidence_map, 0.0, self.conf_hi)
        confidence_map = np.log1p(confidence_map) / np.log1p(self.conf_hi)
        return depth_map, confidence_map

    def _apply_intensity_aug(self, depth_map: np.ndarray, confidence_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        depth_map = np.clip(depth_map * random.uniform(0.9, 1.1) + random.uniform(-0.03, 0.03), 0.0, 1.0)
        confidence_map = np.clip(confidence_map * random.uniform(0.8, 1.2) + random.uniform(-0.05, 0.05), 0.0, 1.0)
        depth_map += np.random.normal(0.0, 0.01, size=depth_map.shape).astype(np.float32)
        confidence_map += np.random.normal(0.0, 0.015, size=confidence_map.shape).astype(np.float32)
        depth_map = np.clip(depth_map, 0.0, 1.0)
        confidence_map = np.clip(confidence_map, 0.0, 1.0)

        if random.random() < 0.3:
            attenuation = random.uniform(0.0, 0.4)
            bg_mask = confidence_map < 0.15
            depth_map = np.where(bg_mask, depth_map * attenuation, depth_map)
            confidence_map = np.where(bg_mask, confidence_map * attenuation, confidence_map)

        return depth_map.astype(np.float32), confidence_map.astype(np.float32)

    def _stack_channels(self, depth_map: np.ndarray, confidence_map: np.ndarray) -> np.ndarray:
        depth_gated = depth_map * confidence_map
        return np.stack([depth_map, confidence_map, depth_gated], axis=0).astype(np.float32)

    def _apply_person_crop(self, channels: np.ndarray, keypoints: np.ndarray, valid_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not valid_mask.any() or random.random() >= 0.5:
            return channels, keypoints

        xs = keypoints[valid_mask, 0]
        ys = keypoints[valid_mask, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())

        crop_w = max(32.0, (x_max - x_min) * 1.25 * random.uniform(0.9, 1.1))
        crop_h = max(32.0, (y_max - y_min) * 1.25 * random.uniform(0.9, 1.1))
        cx = (x_min + x_max) / 2.0 + random.uniform(-10.0, 10.0)
        cy = (y_min + y_max) / 2.0 + random.uniform(-10.0, 10.0)

        height, width = channels.shape[1:]
        left = max(0, int(round(cx - crop_w / 2.0)))
        top = max(0, int(round(cy - crop_h / 2.0)))
        right = min(width, int(round(cx + crop_w / 2.0)))
        bottom = min(height, int(round(cy + crop_h / 2.0)))
        if right - left < 8 or bottom - top < 8:
            return channels, keypoints

        channels = channels[:, top:bottom, left:right]
        keypoints = keypoints.copy()
        keypoints[:, 0] -= left
        keypoints[:, 1] -= top
        return channels, keypoints

    def _apply_spatial_aug(
        self,
        channels: np.ndarray,
        keypoints: np.ndarray,
        valid_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if random.random() > 0.5:
            channels = channels[:, :, ::-1].copy()
            img_width = channels.shape[2]
            keypoints = keypoints.copy()
            keypoints[:, 0] = img_width - 1 - keypoints[:, 0]
            keypoints = keypoints[FLIP_INDICES]
            valid_mask = valid_mask[FLIP_INDICES]

        angle = random.uniform(-15.0, 15.0)
        scale = random.uniform(0.95, 1.05)
        tx = random.uniform(-6.0, 6.0)
        ty = random.uniform(-6.0, 6.0)
        height, width = channels.shape[1:]
        matrix = build_affine_matrix(width, height, angle, scale, tx, ty)

        warped = np.empty_like(channels)
        for channel_idx in range(channels.shape[0]):
            warped[channel_idx] = cv2.warpAffine(
                channels[channel_idx],
                matrix,
                (width, height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0.0,
            )
        keypoints = apply_affine_to_keypoints(keypoints, matrix)

        erase_count = random.randint(1, 2)
        for _ in range(erase_count):
            if random.random() >= 0.5:
                continue
            rect_w = random.randint(8, min(24, width))
            rect_h = random.randint(8, min(24, height))
            x0 = random.randint(0, max(0, width - rect_w))
            y0 = random.randint(0, max(0, height - rect_h))
            warped[:, y0 : y0 + rect_h, x0 : x0 + rect_w] = 0.0

        return warped.astype(np.float32), keypoints.astype(np.float32), valid_mask

    def __getitem__(self, idx: int):
        record = self.records[idx]
        sample_id = record["sample_id"]
        depth_map, confidence_map, keypoints = self._load_sample(sample_id)
        depth_map, confidence_map = self._normalize_inputs(depth_map, confidence_map)

        initial_valid = np.array(record.get("valid_kpt_mask", [1] * keypoints.shape[0]), dtype=bool)
        if self.augment:
            depth_map, confidence_map = self._apply_intensity_aug(depth_map, confidence_map)

        channels = self._stack_channels(depth_map, confidence_map)
        if self.augment:
            channels, keypoints = self._apply_person_crop(channels, keypoints, initial_valid)
            channels, keypoints, initial_valid = self._apply_spatial_aug(channels, keypoints, initial_valid)

        input_tensor = torch.from_numpy(channels).float()
        _, height, width = input_tensor.shape
        pad_left = (self.output_res[1] - width) // 2
        pad_right = self.output_res[1] - width - pad_left
        pad_top = (self.output_res[0] - height) // 2
        pad_bottom = self.output_res[0] - height - pad_top
        input_tensor = transforms.functional.pad(input_tensor, (pad_left, pad_top, pad_right, pad_bottom))

        keypoints = keypoints.copy()
        keypoints[:, 0] += pad_left
        keypoints[:, 1] += pad_top

        final_valid = (
            initial_valid
            & np.isfinite(keypoints).all(axis=1)
            & (keypoints[:, 0] >= 0)
            & (keypoints[:, 0] < self.output_res[1])
            & (keypoints[:, 1] >= 0)
            & (keypoints[:, 1] < self.output_res[0])
        )

        target_heatmaps = generate_heatmaps(keypoints, final_valid, self.output_res, sigma=4.0).float()
        gt_kpts = torch.from_numpy(keypoints.astype(np.float32)).float()
        valid_kpt_mask = torch.from_numpy(final_valid.astype(np.bool_))
        sample_meta = {
            "sample_id": sample_id,
            "scene_id": record.get("scene_id", "scene_0"),
            "session_id": record.get("session_id", "session_0001"),
            "split": record.get("split", "train_core"),
        }
        return input_tensor, target_heatmaps, gt_kpts, valid_kpt_mask, sample_meta
