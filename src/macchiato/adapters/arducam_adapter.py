"""Arducam manifest inspection and Ultralytics pose conversion."""

from __future__ import annotations

import csv
import json
import random
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import yaml
from PIL import Image

from macchiato.adapters.base_adapter import MANIFEST_COLUMNS, ManifestRecord
from macchiato.preprocessing.depth_normalization import depth_to_rgb_uint8
from macchiato.preprocessing.person_crop import padded_keypoint_box

COCO_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
COCO_FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]


def _relative_path(path: Path, project_root: Path) -> str:
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def discover_timestamp_stems(
    depth_dir: Path,
    confidence_dir: Path,
    pose_dir: Path,
) -> tuple[list[int], dict[str, int]]:
    """Return sorted numeric stems present in all required components."""

    component_stems = {
        "depth": {path.stem for path in depth_dir.glob("*.npy")},
        "confidence": {path.stem for path in confidence_dir.glob("*.npy")},
        "pose": {path.stem for path in pose_dir.glob("*.json")},
    }
    common = set.intersection(*component_stems.values()) if component_stems else set()
    numeric = sorted(int(stem) for stem in common if stem.isdigit())
    union = set.union(*component_stems.values()) if component_stems else set()
    diagnostics = {f"{name}_files": len(stems) for name, stems in component_stems.items()}
    diagnostics["complete_samples"] = len(common)
    diagnostics["numeric_samples"] = len(numeric)
    diagnostics["non_timestamp_complete"] = len(common) - len(numeric)
    diagnostics["incomplete_stems"] = len(union - common)
    return numeric, diagnostics


def assign_scene_ids(timestamps: Sequence[int], gap_seconds: int = 180) -> list[str]:
    """Assign sequential scene IDs, splitting only when a gap is strictly larger."""

    if gap_seconds < 0:
        raise ValueError("gap_seconds cannot be negative")
    if not timestamps:
        return []
    if list(timestamps) != sorted(timestamps):
        raise ValueError("timestamps must be sorted")

    scene_number = 0
    previous: int | None = None
    scene_ids: list[str] = []
    for timestamp in timestamps:
        if previous is not None and timestamp - previous > gap_seconds:
            scene_number += 1
        scene_ids.append(f"scene_{scene_number:03d}")
        previous = timestamp
    return scene_ids


def assign_scene_splits(
    scene_ids: Iterable[str],
    seed: int = 42,
    train_scene_count: int = 6,
    val_scene_count: int = 2,
) -> dict[str, str]:
    """Shuffle unique scenes deterministically and assign train/val/test."""

    unique_scenes = sorted(set(scene_ids))
    if train_scene_count < 1 or val_scene_count < 1:
        raise ValueError("train_scene_count and val_scene_count must be positive")
    if train_scene_count + val_scene_count >= len(unique_scenes):
        raise ValueError("At least one scene must remain for the test split")

    shuffled = unique_scenes.copy()
    random.Random(seed).shuffle(shuffled)
    mapping: dict[str, str] = {}
    for scene_id in shuffled[:train_scene_count]:
        mapping[scene_id] = "train"
    for scene_id in shuffled[train_scene_count : train_scene_count + val_scene_count]:
        mapping[scene_id] = "val"
    for scene_id in shuffled[train_scene_count + val_scene_count :]:
        mapping[scene_id] = "test"
    return mapping


def load_keypoints(path: Path, expected_count: int = 17) -> np.ndarray:
    """Load and validate a COCO-style ``(K, 2)`` keypoint array."""

    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or "keypoints" not in payload:
        raise ValueError("pose JSON must contain a 'keypoints' field")
    keypoints = np.asarray(payload["keypoints"], dtype=np.float32)
    if keypoints.shape != (expected_count, 2):
        raise ValueError(f"expected keypoint shape {(expected_count, 2)}, got {keypoints.shape}")
    return keypoints


def keypoint_valid_mask(keypoints: np.ndarray, width: int, height: int) -> np.ndarray:
    """Return the finite, in-frame mask for keypoints."""

    return (
        np.isfinite(keypoints).all(axis=1)
        & (keypoints[:, 0] >= 0)
        & (keypoints[:, 0] < width)
        & (keypoints[:, 1] >= 0)
        & (keypoints[:, 1] < height)
    )


def build_manifest_records(
    raw_root: Path,
    pose_root: Path,
    project_root: Path,
    expected_shape: tuple[int, int] = (180, 240),
    expected_keypoints: int = 17,
    gap_seconds: int = 180,
    minimum_valid_keypoints: int = 12,
    seed: int = 42,
    train_scene_count: int = 6,
    val_scene_count: int = 2,
) -> tuple[list[ManifestRecord], dict[str, int]]:
    """Inspect timestamp-aligned samples and build deterministic manifest rows."""

    depth_dir = raw_root / "depth"
    confidence_dir = raw_root / "confidence"
    timestamps, diagnostics = discover_timestamp_stems(depth_dir, confidence_dir, pose_root)
    scene_ids = assign_scene_ids(timestamps, gap_seconds)
    split_by_scene = assign_scene_splits(scene_ids, seed, train_scene_count, val_scene_count)

    records: list[ManifestRecord] = []
    for timestamp, scene_id in zip(timestamps, scene_ids):
        stem = str(timestamp)
        depth_path = depth_dir / f"{stem}.npy"
        confidence_path = confidence_dir / f"{stem}.npy"
        pose_path = pose_root / f"{stem}.json"
        height: int | str = ""
        width: int | str = ""
        valid_count = 0
        drop_reason = ""
        eligible = False
        try:
            depth = np.load(depth_path, mmap_mode="r")
            confidence = np.load(confidence_path, mmap_mode="r")
            if depth.shape != expected_shape:
                raise ValueError(f"depth shape {depth.shape} does not match {expected_shape}")
            if confidence.shape != expected_shape:
                raise ValueError(f"confidence shape {confidence.shape} does not match {expected_shape}")
            height, width = expected_shape
            keypoints = load_keypoints(pose_path, expected_keypoints)
            valid_count = int(keypoint_valid_mask(keypoints, width, height).sum())
            eligible = valid_count >= minimum_valid_keypoints
            if not eligible:
                drop_reason = f"fewer_than_{minimum_valid_keypoints}_valid_keypoints"
        except Exception as exc:  # noqa: BLE001 - preserve the reason in the manifest
            drop_reason = f"invalid_sample: {exc}"

        records.append(
            ManifestRecord(
                timestamp=timestamp,
                datetime_utc=datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                scene_id=scene_id,
                split=split_by_scene[scene_id],
                eligible=eligible,
                valid_keypoints=valid_count,
                height=height,
                width=width,
                depth_path=_relative_path(depth_path, project_root),
                confidence_path=_relative_path(confidence_path, project_root),
                pose_path=_relative_path(pose_path, project_root),
                drop_reason=drop_reason,
            )
        )

    diagnostics["scenes"] = len(set(scene_ids))
    diagnostics["eligible_samples"] = sum(record.eligible for record in records)
    diagnostics["rejected_samples"] = len(records) - diagnostics["eligible_samples"]
    return records, diagnostics


def write_manifests(records: Sequence[ManifestRecord], output_dir: Path) -> None:
    """Write the complete manifest and eligible split-specific manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    destinations = {
        "all": output_dir / "all_samples.csv",
        "train": output_dir / "train.csv",
        "val": output_dir / "val.csv",
        "test": output_dir / "test.csv",
    }
    for split, path in destinations.items():
        selected = records if split == "all" else [r for r in records if r.eligible and r.split == split]
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=MANIFEST_COLUMNS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(record.to_row() for record in selected)


def read_manifest(path: Path, eligible_only: bool = True) -> list[dict[str, str]]:
    """Read a manifest CSV and optionally keep only eligible rows."""

    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if eligible_only:
        rows = [row for row in rows if row.get("eligible", "").lower() == "true"]
    return rows


def keypoints_to_yolo_label(
    keypoints: np.ndarray,
    width: int,
    height: int,
    padding_fraction: float = 0.10,
) -> str:
    """Encode one person as a YOLO pose label with COCO-17 visibility."""

    valid_mask = keypoint_valid_mask(keypoints, width, height)
    box = padded_keypoint_box(keypoints, valid_mask, width, height, padding_fraction)
    x_min, y_min, x_max, y_max = box
    box_values = [
        ((x_min + x_max) / 2.0) / width,
        ((y_min + y_max) / 2.0) / height,
        (x_max - x_min) / width,
        (y_max - y_min) / height,
    ]
    tokens = ["0", *(f"{value:.8f}" for value in box_values)]
    for (x_coord, y_coord), valid in zip(keypoints, valid_mask):
        if valid:
            tokens.extend([f"{float(x_coord) / width:.8f}", f"{float(y_coord) / height:.8f}", "2"])
        else:
            tokens.extend(["0.00000000", "0.00000000", "0"])
    if len(tokens) != 56:
        raise AssertionError(f"Expected 56 YOLO pose fields, produced {len(tokens)}")
    return " ".join(tokens)


def prepare_yolo_dataset(
    records: Sequence[ManifestRecord] | Sequence[dict[str, str]],
    project_root: Path,
    output_root: Path,
    depth_maximum_mm: float = 4000.0,
    padding_fraction: float = 0.10,
    force: bool = False,
) -> dict[str, int]:
    """Generate normalized PNGs, YOLO labels, and dataset YAML."""

    if output_root.exists() and any(output_root.iterdir()):
        if not force:
            raise FileExistsError(f"YOLO output is not empty: {output_root}. Pass --force to rebuild it.")
        shutil.rmtree(output_root)

    counts: Counter[str] = Counter()
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for record in records:
        row = record.to_row() if isinstance(record, ManifestRecord) else record
        if str(row.get("eligible", "")).lower() != "true":
            continue
        split = str(row["split"])
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split {split!r}")
        timestamp = str(row["timestamp"])
        depth_path = project_root / str(row["depth_path"])
        pose_path = project_root / str(row["pose_path"])
        height = int(str(row["height"]))
        width = int(str(row["width"]))

        depth = np.load(depth_path)
        if depth.shape != (height, width):
            raise ValueError(f"Depth shape changed for {timestamp}: {depth.shape}")
        image = depth_to_rgb_uint8(depth, depth_maximum_mm)
        Image.fromarray(image).save(output_root / "images" / split / f"{timestamp}.png", format="PNG")

        keypoints = load_keypoints(pose_path, expected_count=17)
        label = keypoints_to_yolo_label(keypoints, width, height, padding_fraction)
        (output_root / "labels" / split / f"{timestamp}.txt").write_text(label + "\n", encoding="utf-8")
        counts[split] += 1

    dataset_config = {
        # Ultralytics resolves relative dataset roots against its global
        # datasets directory, so generated local configs use an absolute root.
        "path": output_root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "kpt_shape": [17, 3],
        "flip_idx": COCO_FLIP_INDICES,
        "names": {0: "person"},
        "kpt_names": {0: COCO_KEYPOINT_NAMES},
    }
    with (output_root / "dataset.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dataset_config, handle, sort_keys=False)
    validated = validate_yolo_dataset(output_root)
    expected = {split: counts[split] for split in ("train", "val", "test")}
    if validated != expected:
        raise AssertionError(f"Generated YOLO counts changed during validation: {expected} vs {validated}")
    return validated


def validate_yolo_dataset(output_root: Path) -> dict[str, int]:
    """Exhaustively validate generated PNG/label pairs without Ultralytics."""

    counts: dict[str, int] = {}
    for split in ("train", "val", "test"):
        image_dir = output_root / "images" / split
        label_dir = output_root / "labels" / split
        image_paths = {path.stem: path for path in image_dir.glob("*.png")}
        label_paths = {path.stem: path for path in label_dir.glob("*.txt")}
        if image_paths.keys() != label_paths.keys():
            missing_labels = sorted(image_paths.keys() - label_paths.keys())[:5]
            missing_images = sorted(label_paths.keys() - image_paths.keys())[:5]
            raise ValueError(f"Mismatched {split} stems; missing labels={missing_labels}, missing images={missing_images}")
        if not image_paths:
            raise ValueError(f"No generated samples for split {split}")

        for stem, image_path in image_paths.items():
            with Image.open(image_path) as image_handle:
                image = np.asarray(image_handle.convert("RGB"))
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise ValueError(f"Invalid generated image {image_path}: {image.shape} {image.dtype}")
            if not (np.array_equal(image[..., 0], image[..., 1]) and np.array_equal(image[..., 1], image[..., 2])):
                raise ValueError(f"Generated depth channels differ: {image_path}")

            tokens = label_paths[stem].read_text(encoding="utf-8").split()
            if len(tokens) != 56:
                raise ValueError(f"Expected 56 fields in {label_paths[stem]}, got {len(tokens)}")
            if tokens[0] != "0":
                raise ValueError(f"Expected person class 0 in {label_paths[stem]}")
            values = np.asarray([float(token) for token in tokens[1:]], dtype=np.float64)
            if not np.isfinite(values).all():
                raise ValueError(f"Non-finite label value in {label_paths[stem]}")
            box = values[:4]
            if not ((box >= 0).all() and (box <= 1).all() and box[2] > 0 and box[3] > 0):
                raise ValueError(f"Invalid normalized box in {label_paths[stem]}")
            keypoints = values[4:].reshape(17, 3)
            if not set(keypoints[:, 2].astype(int)) <= {0, 2}:
                raise ValueError(f"Invalid visibility value in {label_paths[stem]}")
            if not ((keypoints[:, :2] >= 0).all() and (keypoints[:, :2] <= 1).all()):
                raise ValueError(f"Out-of-range normalized keypoint in {label_paths[stem]}")
            invisible = keypoints[:, 2] == 0
            if not np.all(keypoints[invisible, :2] == 0):
                raise ValueError(f"Invisible keypoints must be encoded as 0 0 0 in {label_paths[stem]}")
        counts[split] = len(image_paths)
    return counts


def summarize_records(records: Sequence[ManifestRecord]) -> str:
    """Return a compact deterministic scene/split summary."""

    counts: Counter[tuple[str, str]] = Counter((record.scene_id, record.split) for record in records if record.eligible)
    lines = ["Eligible samples by scene:"]
    for (scene_id, split), count in sorted(counts.items()):
        lines.append(f"  {scene_id}  {split:<5}  {count:>4}")
    return "\n".join(lines)
