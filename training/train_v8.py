#!/usr/bin/env python3
"""V8 ToF pose training (TensorFlow / Keras).

Trains the tiny pose model consumed by
`tof_pose.inference.HeatmapOffsetPoseEstimator`. Reuses the per-sample data
layout produced by the main-branch capture pipeline:

    {data_dir}/depth/{name}.npy           # H x W, depth in millimetres
    {data_dir}/confidence/{name}.npy      # H x W, ToF confidence
    {data_dir}/pose_coco17/{name}.json    # {"keypoints": [[x, y], ... 17 entries]}

Coordinates in the JSON are in ToF-native pixels (same coordinate frame as the
depth/confidence maps).

Pipeline differences vs main:
  * COCO-17 -> V8 15-joint scheme (synthesises `neck`, `mid_spine`)
  * person-ROI crop derived from keypoint bbox + jitter (matches runtime)
  * square-pad before resize so limb proportions are preserved
  * fixed depth scaling (mm/4000) to retain absolute scale across distances
  * heatmap (64x64) + sub-pixel offset head, matching V8 inference layout
  * ToF-specific aug: depth scale jitter, depth dropout, amplitude/conf jitter
  * PCK@0.05 of ROI diagonal as the primary accuracy metric
  * INT8 TFLite export with representative dataset

Run:
    python training/train_v8.py train \\
        --data-dir data \\
        --output artifacts/v8 \\
        --epochs 60

    python training/train_v8.py export \\
        --saved-model artifacts/v8/saved_model \\
        --representative artifacts/v8/representative.npz \\
        --output artifacts/v8/depth_pose_int8.tflite
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------- joints

# V8 joint order (matches tof_pose.types.JOINT_NAMES_15).
V8_JOINTS = (
    "head", "neck", "mid_spine",
    "right_shoulder", "right_elbow", "right_hand",
    "left_shoulder", "left_elbow", "left_hand",
    "right_hip", "right_knee", "right_foot",
    "left_hip", "left_knee", "left_foot",
)
NUM_V8_JOINTS = len(V8_JOINTS)

# (V8 index, COCO-17 index) for joints copied directly from COCO.
_COCO_DIRECT = (
    (0, 0),    # head <- nose
    (3, 6),    # right_shoulder
    (4, 8),    # right_elbow
    (5, 10),   # right_hand <- right_wrist
    (6, 5),    # left_shoulder
    (7, 7),    # left_elbow
    (8, 9),    # left_hand <- left_wrist
    (9, 12),   # right_hip
    (10, 14),  # right_knee
    (11, 16),  # right_foot <- right_ankle
    (12, 11),  # left_hip
    (13, 13),  # left_knee
    (14, 15),  # left_foot <- left_ankle
)

# Left/right index pairs in V8 space. Used for horizontal-flip augmentation.
V8_FLIP_PAIRS = (
    (3, 6), (4, 7), (5, 8),     # shoulders / elbows / hands
    (9, 12), (10, 13), (11, 14),  # hips / knees / feet
)


def coco17_to_v8(kp_coco17: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert a (17, 2) COCO keypoint array to (15, 2) V8 keypoints + validity.

    A joint is "valid" if all of its source COCO joints have finite, positive
    coordinates. Synthetic joints (neck, mid_spine) require all contributors.
    """
    kp = np.zeros((NUM_V8_JOINTS, 2), dtype=np.float32)
    valid = np.zeros((NUM_V8_JOINTS,), dtype=np.float32)

    def _is_valid(idx):
        x, y = kp_coco17[idx]
        return np.isfinite(x) and np.isfinite(y) and (x > 0.0 or y > 0.0)

    # Direct copies
    for v8_idx, coco_idx in _COCO_DIRECT:
        kp[v8_idx] = kp_coco17[coco_idx]
        valid[v8_idx] = float(_is_valid(coco_idx))

    # neck = midpoint of shoulders (COCO 5, 6)
    if _is_valid(5) and _is_valid(6):
        kp[1] = (kp_coco17[5] + kp_coco17[6]) / 2.0
        valid[1] = 1.0

    # mid_spine = mean of both shoulders + both hips
    if all(_is_valid(i) for i in (5, 6, 11, 12)):
        kp[2] = (kp_coco17[5] + kp_coco17[6] + kp_coco17[11] + kp_coco17[12]) / 4.0
        valid[2] = 1.0

    return kp, valid


# --------------------------------------------------------------------- config

@dataclass(slots=True)
class TrainConfig:
    data_dir: Path
    output_dir: Path
    input_size: int = 128         # model input H/W
    heatmap_size: int = 64        # heatmap & offset head H/W (stride 2)
    heatmap_sigma: float = 1.75   # in heatmap pixels
    offset_radius: int = 3        # offset loss is non-zero within this radius
    depth_scale_mm: float = 4000.0
    confidence_scale: float = 255.0
    backbone_alpha: float = 0.5
    backbone_weights: str | None = "imagenet"
    batch_size: int = 32
    epochs: int = 60
    learning_rate: float = 1e-3
    val_split: float = 0.15
    random_seed: int = 720
    augment: bool = True
    roi_margin: float = 0.20      # extra margin around keypoint bbox
    roi_jitter: float = 0.15      # uniform jitter on bbox during training
    early_stop_patience: int = 10
    plateau_patience: int = 4
    pck_alpha: float = 0.05       # PCK threshold = alpha * ROI diagonal
    num_workers: int = 4


# ---------------------------------------------------------------------- IO

def _list_samples(data_dir: Path) -> list[str]:
    depth_dir = data_dir / "depth"
    pose_dir = data_dir / "pose_coco17"
    conf_dir = data_dir / "confidence"
    if not depth_dir.is_dir() or not pose_dir.is_dir():
        raise FileNotFoundError(
            f"Expected {depth_dir} and {pose_dir} to exist (main-branch layout)."
        )
    names = []
    for npy in sorted(depth_dir.glob("*.npy")):
        stem = npy.stem
        if (pose_dir / f"{stem}.json").exists() and (conf_dir / f"{stem}.npy").exists():
            names.append(stem)
    if not names:
        raise FileNotFoundError(f"No matching samples found in {data_dir}.")
    return names


def _load_raw_sample(data_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    depth = np.load(data_dir / "depth" / f"{name}.npy").astype(np.float32)
    confidence = np.load(data_dir / "confidence" / f"{name}.npy").astype(np.float32)
    with open(data_dir / "pose_coco17" / f"{name}.json", "r") as f:
        kp_coco = np.asarray(json.load(f)["keypoints"], dtype=np.float32)
    return depth, confidence, kp_coco


# --------------------------------------------------------------- ROI cropping

def _keypoint_bbox(kp: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, float] | None:
    if not np.any(valid > 0.5):
        return None
    xs = kp[valid > 0.5, 0]
    ys = kp[valid > 0.5, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def _square_roi(
    bbox: tuple[float, float, float, float],
    img_shape: tuple[int, int],
    margin: float,
    jitter: float,
    rng: np.random.Generator,
) -> tuple[int, int, int, int]:
    """Expand keypoint bbox to a square ROI with margin + optional jitter."""
    h, w = img_shape
    x0, y0, x1, y1 = bbox
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    side = max(x1 - x0, y1 - y0, 16.0) * (1.0 + margin)
    if jitter > 0.0:
        cx += rng.uniform(-jitter, jitter) * side
        cy += rng.uniform(-jitter, jitter) * side
        side *= rng.uniform(1.0 - jitter, 1.0 + jitter)
    half = side * 0.5
    rx0 = int(round(cx - half))
    ry0 = int(round(cy - half))
    rx1 = int(round(cx + half))
    ry1 = int(round(cy + half))
    # Allow ROI to extend past image bounds; we pad with zeros below.
    return rx0, ry0, rx1, ry1


def _crop_and_pad(image: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = roi
    h, w = image.shape[:2]
    out_h = y1 - y0
    out_w = x1 - x0
    out = np.zeros((out_h, out_w) + image.shape[2:], dtype=image.dtype)
    sx0 = max(x0, 0)
    sy0 = max(y0, 0)
    sx1 = min(x1, w)
    sy1 = min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = image[sy0:sy1, sx0:sx1]
    return out


def _resize_2d(image: np.ndarray, size: int, mode: str = "bilinear") -> np.ndarray:
    """Pure-numpy resize. mode='bilinear' for depth/conf, 'nearest' for masks."""
    in_h, in_w = image.shape[:2]
    if in_h == size and in_w == size:
        return image
    ys = np.linspace(0, in_h - 1, size).astype(np.float32)
    xs = np.linspace(0, in_w - 1, size).astype(np.float32)
    if mode == "nearest":
        yi = np.clip(np.round(ys).astype(np.int32), 0, in_h - 1)
        xi = np.clip(np.round(xs).astype(np.int32), 0, in_w - 1)
        return image[yi][:, xi]
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    fy = (ys - y0)[:, None].astype(np.float32)
    fx = (xs - x0)[None, :].astype(np.float32)
    a = image[y0][:, x0].astype(np.float32)
    b = image[y0][:, x1].astype(np.float32)
    c = image[y1][:, x0].astype(np.float32)
    d = image[y1][:, x1].astype(np.float32)
    top = a * (1.0 - fx) + b * fx
    bot = c * (1.0 - fx) + d * fx
    return (top * (1.0 - fy) + bot * fy).astype(image.dtype)


# --------------------------------------------------------------- augmentation

def _augment_depth(depth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiplicative + additive depth jitter and patch dropout in metric mm."""
    scaled = depth * rng.uniform(0.92, 1.08)
    scaled = scaled + rng.uniform(-150.0, 150.0)  # +/- 15 cm shift
    scaled = np.clip(scaled, 0.0, None)

    if rng.random() < 0.4:
        h, w = scaled.shape
        n_holes = rng.integers(1, 5)
        for _ in range(int(n_holes)):
            hh = rng.integers(3, max(4, h // 8))
            ww = rng.integers(3, max(4, w // 8))
            yy = rng.integers(0, max(1, h - hh))
            xx = rng.integers(0, max(1, w - ww))
            scaled[yy:yy + hh, xx:xx + ww] = 0.0
    return scaled


def _augment_confidence(conf: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return np.clip(conf * rng.uniform(0.7, 1.3), 0.0, 255.0)


def _rotate(image: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate an HxW or HxWxC image around its centre with bilinear sampling."""
    h, w = image.shape[:2]
    cy = (h - 1) * 0.5
    cx = (w - 1) * 0.5
    ys, xs = np.meshgrid(np.arange(h, dtype=np.float32),
                          np.arange(w, dtype=np.float32), indexing="ij")
    yr = ys - cy
    xr = xs - cx
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    src_x = cos_a * xr + sin_a * yr + cx
    src_y = -sin_a * xr + cos_a * yr + cy

    in_h, in_w = h, w
    x0 = np.clip(np.floor(src_x).astype(np.int32), 0, in_w - 1)
    x1 = np.clip(x0 + 1, 0, in_w - 1)
    y0 = np.clip(np.floor(src_y).astype(np.int32), 0, in_h - 1)
    y1 = np.clip(y0 + 1, 0, in_h - 1)
    fx = (src_x - x0).astype(np.float32)
    fy = (src_y - y0).astype(np.float32)
    in_bounds = (src_x >= 0) & (src_x <= in_w - 1) & (src_y >= 0) & (src_y <= in_h - 1)

    def _bilinear(channel):
        a = channel[y0, x0].astype(np.float32)
        b = channel[y0, x1].astype(np.float32)
        c = channel[y1, x0].astype(np.float32)
        d = channel[y1, x1].astype(np.float32)
        top = a * (1.0 - fx) + b * fx
        bot = c * (1.0 - fx) + d * fx
        return np.where(in_bounds, top * (1.0 - fy) + bot * fy, 0.0)

    if image.ndim == 2:
        return _bilinear(image).astype(image.dtype)
    return np.stack([_bilinear(image[..., c]) for c in range(image.shape[2])], axis=-1).astype(image.dtype)


def _rotate_keypoints(kp: np.ndarray, angle_rad: float, size: int) -> np.ndarray:
    """Rotate keypoints around the image centre, matching the image rotation."""
    cx = cy = (size - 1) * 0.5
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    out = kp.copy()
    dx = kp[:, 0] - cx
    dy = kp[:, 1] - cy
    out[:, 0] = cos_a * dx - sin_a * dy + cx
    out[:, 1] = sin_a * dx + cos_a * dy + cy
    return out


def _flip_keypoints(kp: np.ndarray, valid: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    kp = kp.copy()
    valid = valid.copy()
    kp[:, 0] = (size - 1) - kp[:, 0]
    for a, b in V8_FLIP_PAIRS:
        kp[[a, b]] = kp[[b, a]]
        valid[[a, b]] = valid[[b, a]]
    return kp, valid


# --------------------------------------------------------------- target gen

def _make_heatmap_targets(
    kp_input: np.ndarray,
    valid: np.ndarray,
    input_size: int,
    heatmap_size: int,
    sigma: float,
    offset_radius: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (heatmap, offset, offset_mask) targets.

    heatmap: (H, W, J) Gaussian peaks in [0,1].
    offset:  (H, W, 2J) residuals so that  argmax * stride + offset = true_uv.
             Channel layout matches V8 inference: [y0, x0, y1, x1, ...].
    offset_mask: (H, W, 2J) where the offset loss is active.
    """
    stride = input_size / heatmap_size
    heatmap = np.zeros((heatmap_size, heatmap_size, NUM_V8_JOINTS), dtype=np.float32)
    offsets = np.zeros((heatmap_size, heatmap_size, NUM_V8_JOINTS * 2), dtype=np.float32)
    omask = np.zeros_like(offsets)

    ys, xs = np.meshgrid(
        np.arange(heatmap_size, dtype=np.float32),
        np.arange(heatmap_size, dtype=np.float32),
        indexing="ij",
    )

    for j in range(NUM_V8_JOINTS):
        if valid[j] < 0.5:
            continue
        u, v = kp_input[j]
        if not (0.0 <= u < input_size and 0.0 <= v < input_size):
            continue
        hx = u / stride
        hy = v / stride
        heatmap[..., j] = np.exp(-((xs - hx) ** 2 + (ys - hy) ** 2) / (2.0 * sigma * sigma))

        # Offset target around a small neighbourhood of the true peak.
        cx = int(round(hx))
        cy = int(round(hy))
        x0 = max(cx - offset_radius, 0)
        x1 = min(cx + offset_radius + 1, heatmap_size)
        y0 = max(cy - offset_radius, 0)
        y1 = min(cy + offset_radius + 1, heatmap_size)
        if x1 <= x0 or y1 <= y0:
            continue
        grid_x, grid_y = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        # off_y = v - hy_pixel * stride;  off_x = u - hx_pixel * stride
        off_y = v - grid_y.astype(np.float32) * stride
        off_x = u - grid_x.astype(np.float32) * stride
        offsets[y0:y1, x0:x1, j * 2 + 0] = off_y
        offsets[y0:y1, x0:x1, j * 2 + 1] = off_x
        omask[y0:y1, x0:x1, j * 2 + 0] = 1.0
        omask[y0:y1, x0:x1, j * 2 + 1] = 1.0

    return heatmap, offsets, omask


# --------------------------------------------------------------- sample build

def _build_sample(
    depth_raw: np.ndarray,
    conf_raw: np.ndarray,
    kp_raw: np.ndarray,
    config: TrainConfig,
    rng: np.random.Generator,
    augment: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Produce the model input + targets for a single sample."""
    kp_v8, valid = coco17_to_v8(kp_raw)
    img_h, img_w = depth_raw.shape

    # Augment in metric units before ROI crop.
    depth = depth_raw.copy()
    conf = conf_raw.copy()
    if augment:
        depth = _augment_depth(depth, rng)
        conf = _augment_confidence(conf, rng)

    # ROI from keypoint bbox.
    bbox = _keypoint_bbox(kp_v8, valid)
    if bbox is None:
        # No valid keypoints: skip with zero validity (caller will filter).
        empty_in = np.zeros((config.input_size, config.input_size, 2), dtype=np.float32)
        hm, off, mask = _make_heatmap_targets(
            np.zeros((NUM_V8_JOINTS, 2), dtype=np.float32),
            np.zeros((NUM_V8_JOINTS,), dtype=np.float32),
            config.input_size, config.heatmap_size,
            config.heatmap_sigma, config.offset_radius,
        )
        return empty_in, hm, off, mask, np.zeros((NUM_V8_JOINTS,), dtype=np.float32)

    margin = config.roi_margin
    jitter = config.roi_jitter if augment else 0.0
    roi = _square_roi(bbox, (img_h, img_w), margin, jitter, rng)
    rx0, ry0, rx1, ry1 = roi
    roi_side = max(rx1 - rx0, 1)

    depth_crop = _crop_and_pad(depth, roi)
    conf_crop = _crop_and_pad(conf, roi)

    # Map keypoints into ROI-relative coordinates.
    kp_roi = kp_v8.copy()
    kp_roi[:, 0] -= rx0
    kp_roi[:, 1] -= ry0

    # Resize ROI to model input size and rescale keypoints.
    scale = config.input_size / roi_side
    depth_in = _resize_2d(depth_crop, config.input_size, mode="bilinear")
    conf_in = _resize_2d(conf_crop, config.input_size, mode="bilinear")
    kp_in = kp_roi * scale

    # Optional rotation + horizontal flip post-resize.
    if augment:
        if rng.random() < 0.5:
            depth_in = np.ascontiguousarray(depth_in[:, ::-1])
            conf_in = np.ascontiguousarray(conf_in[:, ::-1])
            kp_in, valid = _flip_keypoints(kp_in, valid, config.input_size)
        angle = float(rng.uniform(-math.radians(15.0), math.radians(15.0)))
        if abs(angle) > 1e-3:
            depth_in = _rotate(depth_in, angle)
            conf_in = _rotate(conf_in, angle)
            kp_in = _rotate_keypoints(kp_in, angle, config.input_size)

    # Mark joints that fell outside the image as invalid.
    inside = (
        (kp_in[:, 0] >= 0) & (kp_in[:, 0] < config.input_size) &
        (kp_in[:, 1] >= 0) & (kp_in[:, 1] < config.input_size)
    )
    valid = valid * inside.astype(np.float32)

    # Fixed-scale normalisation (do NOT per-frame normalise - keeps metric depth).
    depth_norm = np.clip(depth_in / config.depth_scale_mm, 0.0, 1.0)
    conf_norm = np.clip(conf_in / config.confidence_scale, 0.0, 1.0)
    input_tensor = np.stack([depth_norm, conf_norm], axis=-1).astype(np.float32)

    heatmap, offsets, omask = _make_heatmap_targets(
        kp_in, valid, config.input_size, config.heatmap_size,
        config.heatmap_sigma, config.offset_radius,
    )
    return input_tensor, heatmap, offsets, omask, valid


# ---------------------------------------------------------------- tf.data

def _make_dataset(
    names: list[str],
    config: TrainConfig,
    augment: bool,
    shuffle: bool,
):
    import tensorflow as tf

    rng_seed = config.random_seed + (1 if augment else 0)

    def _gen():
        rng = np.random.default_rng(rng_seed)
        order = list(names)
        if shuffle:
            random.Random(rng_seed).shuffle(order)
        for name in order:
            depth, conf, kp = _load_raw_sample(config.data_dir, name)
            x, hm, off, mask, valid = _build_sample(depth, conf, kp, config, rng, augment)
            if valid.sum() < 1.0:
                continue
            yield (x, (hm, off, mask))

    output_signature = (
        tf.TensorSpec((config.input_size, config.input_size, 2), tf.float32),
        (
            tf.TensorSpec((config.heatmap_size, config.heatmap_size, NUM_V8_JOINTS), tf.float32),
            tf.TensorSpec((config.heatmap_size, config.heatmap_size, NUM_V8_JOINTS * 2), tf.float32),
            tf.TensorSpec((config.heatmap_size, config.heatmap_size, NUM_V8_JOINTS * 2), tf.float32),
        ),
    )
    ds = tf.data.Dataset.from_generator(_gen, output_signature=output_signature)
    if shuffle:
        ds = ds.shuffle(256, seed=config.random_seed, reshuffle_each_iteration=True)
    ds = ds.batch(config.batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ------------------------------------------------------------------- model

# MobileNetV2 ImageNet weights are published for these alpha values.
# (MobileNetV3-Small only publishes 0.75 and 1.0, and not for the minimalistic
# variant, which is why we switched.)
_MOBILENETV2_IMAGENET_ALPHAS = (0.35, 0.5, 0.75, 1.0, 1.3, 1.4)


def build_model(config: TrainConfig):
    import tensorflow as tf

    input_c = 2
    inputs = tf.keras.Input(shape=(config.input_size, config.input_size, input_c), name="roi")

    weights = config.backbone_weights or None
    alpha = float(config.backbone_alpha)
    if weights == "imagenet" and alpha not in _MOBILENETV2_IMAGENET_ALPHAS:
        raise ValueError(
            f"--alpha {alpha} is not available with ImageNet weights. "
            f"Pick one of {_MOBILENETV2_IMAGENET_ALPHAS}, or pass --backbone-weights '' "
            "to train from scratch with any alpha."
        )

    # MobileNetV2 ImageNet weights expect 3 channels. Lift 2 -> 3 with a tiny
    # learned stem so the rest of the backbone sees the expected input depth.
    if weights == "imagenet":
        x = tf.keras.layers.Conv2D(3, 1, padding="same", name="stem_lift")(inputs)
    else:
        x = inputs

    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(config.input_size, config.input_size, 3),
        alpha=alpha,
        include_top=False,
        weights=weights,
        input_tensor=x,
    )

    feat = backbone.output  # stride 32 -> 4x4 for 128 input
    x = tf.keras.layers.Conv2D(96, 1, padding="same", activation="relu", name="neck1")(feat)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)   # 8x8
    x = tf.keras.layers.SeparableConv2D(96, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)   # 16x16
    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)   # 32x32
    x = tf.keras.layers.SeparableConv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.UpSampling2D(size=2, interpolation="bilinear")(x)   # 64x64
    x = tf.keras.layers.SeparableConv2D(48, 3, padding="same", activation="relu")(x)

    heatmaps = tf.keras.layers.Conv2D(
        NUM_V8_JOINTS, 1, padding="same", activation="sigmoid", name="heatmaps"
    )(x)
    offsets = tf.keras.layers.Conv2D(
        NUM_V8_JOINTS * 2, 1, padding="same", activation=None, name="offsets"
    )(x)
    return tf.keras.Model(inputs=inputs, outputs=[heatmaps, offsets], name="tof_pose_v8")


# ------------------------------------------------------------------ losses

def _weighted_heatmap_loss(pos_weight: float = 100.0):
    import tensorflow as tf

    def loss(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, 1e-6, 1.0 - 1e-6)
        pos = -y_true * tf.math.log(y_pred) * pos_weight
        neg = -(1.0 - y_true) * tf.math.log(1.0 - y_pred)
        return tf.reduce_mean(pos + neg)

    loss.__name__ = "weighted_heatmap_bce"
    return loss


def _masked_offset_loss():
    import tensorflow as tf

    def loss(y_true_packed, y_pred):
        # y_true_packed concatenates [offsets, mask] along channel axis at build time.
        channels = y_pred.shape[-1]
        offsets_true = y_true_packed[..., :channels]
        mask = y_true_packed[..., channels:]
        diff = (y_pred - offsets_true) * mask
        # Huber on the masked residual.
        abs_diff = tf.abs(diff)
        delta = 1.0
        quad = tf.minimum(abs_diff, delta)
        lin = abs_diff - quad
        huber = 0.5 * quad * quad + delta * lin
        denom = tf.reduce_sum(mask) + 1e-6
        return tf.reduce_sum(huber) / denom

    loss.__name__ = "masked_offset_huber"
    return loss


# ------------------------------------------------------------------- pck

class PCKCallback:
    """Validation PCK at alpha * ROI-diagonal threshold."""

    def __init__(self, val_dataset, config: TrainConfig):
        self.val_dataset = val_dataset
        self.config = config

    def evaluate(self, model) -> dict[str, float]:
        import tensorflow as tf

        stride = self.config.input_size / self.config.heatmap_size
        thresh = self.config.pck_alpha * math.sqrt(2.0) * self.config.input_size
        correct = np.zeros(NUM_V8_JOINTS, dtype=np.float64)
        total = np.zeros(NUM_V8_JOINTS, dtype=np.float64)

        for xb, (hm_true, _, _) in self.val_dataset:
            preds = model.predict(xb, verbose=0)
            hm_pred = preds[0]
            offs_pred = preds[1]
            batch = hm_pred.shape[0]
            for b in range(batch):
                for j in range(NUM_V8_JOINTS):
                    plane = hm_pred[b, ..., j]
                    flat = int(np.argmax(plane))
                    py, px = divmod(flat, plane.shape[1])
                    off_y = float(offs_pred[b, py, px, j * 2 + 0])
                    off_x = float(offs_pred[b, py, px, j * 2 + 1])
                    pu = px * stride + off_x
                    pv = py * stride + off_y

                    plane_t = hm_true[b, ..., j]
                    if plane_t.numpy().max() < 0.5:
                        continue
                    flat_t = int(np.argmax(plane_t))
                    ty, tx = divmod(flat_t, plane_t.shape[1])
                    tu = (tx + 0.5) * stride
                    tv = (ty + 0.5) * stride
                    err = math.hypot(pu - tu, pv - tv)
                    if err <= thresh:
                        correct[j] += 1
                    total[j] += 1
        per_joint = np.where(total > 0, correct / np.maximum(total, 1), 0.0)
        return {
            "pck_overall": float(np.where(total.sum() > 0, correct.sum() / total.sum(), 0.0)),
            "pck_per_joint": per_joint.tolist(),
        }


# ------------------------------------------------------------------- train

def train(args: argparse.Namespace) -> int:
    import tensorflow as tf

    config = TrainConfig(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        backbone_alpha=args.alpha,
        backbone_weights=args.backbone_weights or None,
        augment=not args.no_augment,
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)

    names = _list_samples(config.data_dir)
    rng = np.random.default_rng(config.random_seed)
    rng.shuffle(names)
    split = int(len(names) * (1.0 - config.val_split))
    train_names = names[:split]
    val_names = names[split:]
    print(f"[data] total={len(names)} train={len(train_names)} val={len(val_names)}")

    train_ds = _make_dataset(train_names, config, augment=config.augment, shuffle=True)
    val_ds = _make_dataset(val_names, config, augment=False, shuffle=False)

    # Pack (offsets, mask) into one tensor so Keras can route a single y_true.
    def _pack(x, y):
        hm, off, mask = y
        packed = tf.concat([off, mask], axis=-1)
        return x, {"heatmaps": hm, "offsets": packed}

    train_ds = train_ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)
    val_pck_ds = val_ds  # keep unpacked for PCK eval
    val_ds_packed = val_ds.map(_pack, num_parallel_calls=tf.data.AUTOTUNE)

    model = build_model(config)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss={
            "heatmaps": _weighted_heatmap_loss(pos_weight=100.0),
            "offsets": _masked_offset_loss(),
        },
        loss_weights={"heatmaps": 1.0, "offsets": 0.3},
    )
    model.summary(line_length=110)

    pck = PCKCallback(val_pck_ds, config)
    best_pck = -1.0
    history = []
    plateau = 0
    no_improve = 0
    current_lr = config.learning_rate

    for epoch in range(config.epochs):
        train_metrics = model.fit(
            train_ds,
            validation_data=val_ds_packed,
            epochs=1,
            verbose=2,
        ).history
        pck_metrics = pck.evaluate(model)
        val_loss = float(train_metrics["val_loss"][-1])
        print(
            f"[epoch {epoch + 1}/{config.epochs}] "
            f"val_loss={val_loss:.4f}  "
            f"pck@{config.pck_alpha:.2f}={pck_metrics['pck_overall']:.3f}  "
            f"lr={current_lr:.1e}"
        )
        history.append({"epoch": epoch + 1, "val_loss": val_loss, **pck_metrics})

        improved = pck_metrics["pck_overall"] > best_pck + 1e-4
        if improved:
            best_pck = pck_metrics["pck_overall"]
            no_improve = 0
            plateau = 0
            ckpt_path = config.output_dir / "best.keras"
            model.save(ckpt_path)
            print(f"  -> new best PCK, saved {ckpt_path}")
        else:
            no_improve += 1
            plateau += 1

        if plateau >= config.plateau_patience:
            current_lr *= 0.3
            tf.keras.backend.set_value(model.optimizer.learning_rate, current_lr)
            plateau = 0
            print(f"  -> LR plateau, reducing to {current_lr:.1e}")

        if no_improve >= config.early_stop_patience:
            print(f"  -> early stopping after {epoch + 1} epochs")
            break

    saved_model_path = config.output_dir / "saved_model"
    if hasattr(model, "export"):
        model.export(str(saved_model_path))
    else:
        tf.saved_model.save(model, str(saved_model_path))
    (config.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"[done] best pck={best_pck:.3f}, saved_model={saved_model_path}")

    # Dump a small representative-dataset .npz next to the saved_model so export
    # can use it without rewalking the data dir.
    rep_inputs = []
    for xb, _ in val_ds.take(8):
        rep_inputs.append(xb.numpy())
    if rep_inputs:
        rep = np.concatenate(rep_inputs, axis=0)[:256]
        np.savez_compressed(config.output_dir / "representative.npz", images=rep)
    return 0


# ------------------------------------------------------------------ export

def export_tflite(args: argparse.Namespace) -> int:
    import tensorflow as tf

    converter = tf.lite.TFLiteConverter.from_saved_model(str(args.saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.float32

    if args.representative:
        reps = np.load(args.representative)["images"].astype(np.float32)

        def representative_dataset():
            for sample in reps[: min(len(reps), 256)]:
                yield [sample[None, ...]]

        converter.representative_dataset = representative_dataset

    tflite_model = converter.convert()
    Path(args.output).write_bytes(tflite_model)
    print(f"[export] wrote {args.output} ({len(tflite_model)/1024:.1f} KB)")
    return 0


# ------------------------------------------------------------------- cli

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python training/train_v8.py")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train")
    p_train.add_argument("--data-dir", required=True)
    p_train.add_argument("--output", required=True)
    p_train.add_argument("--epochs", type=int, default=60)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--learning-rate", type=float, default=1e-3)
    p_train.add_argument("--alpha", type=float, default=0.5)
    p_train.add_argument("--backbone-weights", default="imagenet",
                          help="'imagenet' for pretrained, '' for scratch")
    p_train.add_argument("--no-augment", action="store_true")
    p_train.set_defaults(func=train)

    p_export = sub.add_parser("export")
    p_export.add_argument("--saved-model", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument("--representative")
    p_export.set_defaults(func=export_tflite)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
