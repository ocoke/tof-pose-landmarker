# evaluate_pytorch.py
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import mediapipe as mp
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model architectures and dataset
from model.model import PoseUNet
from model.dataset import PoseDataset
from model_edge.model import EdgePoseUNet
from model_edge_v5.dataset import EdgePoseDataset
from model_edge_v5.model import EdgePoseUNetV2
from model_edge_v6.model import EdgePoseUNetV6
from model_edge_v6.dataset import EdgePoseDatasetV6
from model_slim.dataset import SlimPoseDataset
from model_slim.medium_model import MediumPoseUNet
from model_slim.model import SlimPoseUNet

# MediaPipe (33-point) -> COCO17 mapping
MP_TO_COCO = [
    0,   # nose
    2,   # left_eye
    5,   # right_eye
    7,   # left_ear
    8,   # right_ear
    11,  # left_shoulder
    12,  # right_shoulder
    13,  # left_elbow
    14,  # right_elbow
    15,  # left_wrist
    16,  # right_wrist
    23,  # left_hip
    24,  # right_hip
    25,  # left_knee
    26,  # right_knee
    27,  # left_ankle
    28,  # right_ankle
]


def get_coords_from_heatmaps(heatmaps_tensor: torch.Tensor) -> np.ndarray:
    """Extract (x,y) coordinates from a batch of heatmaps."""
    batch_size, num_keypoints, height, width = heatmaps_tensor.shape
    coords = np.zeros((batch_size, num_keypoints, 2), dtype=np.int32)
    for i in range(batch_size):
        for k in range(num_keypoints):
            heatmap = heatmaps_tensor[i, k, :, :]
            max_index = torch.argmax(heatmap)
            y = max_index // width
            x = max_index % width
            coords[i, k] = [x.item(), y.item()]
    return coords


def build_model(model_arch: str, device: torch.device) -> torch.nn.Module:
    if model_arch == "heavy":
        return PoseUNet(n_channels=2, n_keypoints=33).to(device)
    if model_arch == "slim":
        return SlimPoseUNet(in_ch=2, n_kpts=17).to(device)
    if model_arch == "medium":
        return MediumPoseUNet(in_ch=2, n_kpts=17).to(device)
    if model_arch == "edge":
        return EdgePoseUNet(in_ch=2, n_kpts=17, width_mult=1.15, p_drop=0.02).to(device)
    if model_arch == "edge_v5":
        return EdgePoseUNetV2(in_ch=2, n_kpts=17, width_mult=1.15).to(device)
    if model_arch == "edge_v6":
        return EdgePoseUNetV6(in_ch=3, n_kpts=17, width_mult=1.15).to(device)
    raise ValueError(f"Unknown model architecture '{model_arch}'.")


def build_dataset(model_arch: str, data_dir: str):
    if model_arch == "heavy":
        return PoseDataset(data_dir=data_dir, augment=False, num_keypoints=33)
    if model_arch in ("slim", "medium", "edge"):
        return SlimPoseDataset(data_dir=data_dir, augment=False, num_keypoints=17)
    if model_arch == "edge_v5":
        return EdgePoseDataset(data_dir=data_dir, augment=False, num_keypoints=17)
    if model_arch == "edge_v6":
        return EdgePoseDatasetV6(data_dir=data_dir, augment=False, num_keypoints=17)
    raise ValueError(f"Unknown model architecture '{model_arch}'.")


def load_ground_truth_keypoints(
    filenames: List[str], data_dir: str, model_arch: str, output_res: Tuple[int, int]
) -> Tuple[Dict[str, np.ndarray], Dict[str, Tuple[int, int]]]:
    """Load ground-truth keypoints (padded to match model input) for the given filenames."""
    data_root = Path(data_dir)
    gt_lookup: Dict[str, np.ndarray] = {}
    pad_lookup: Dict[str, Tuple[int, int]] = {}
    out_h, out_w = output_res

    for name in filenames:
        depth_path = data_root / "depth" / f"{name}.npy"
        if not depth_path.exists():
            raise FileNotFoundError(f"Missing depth map for sample {name}: {depth_path}")
        depth_map = np.load(depth_path)
        h, w = depth_map.shape

        pad_left = (out_w - w) // 2
        pad_top = (out_h - h) // 2
        pad_lookup[name] = (pad_left, pad_top)

        if model_arch == "heavy":
            json_path = data_root / "pose" / f"{name}.json"
            key = "transformed_points"
        else:
            json_path = data_root / "pose_coco17" / f"{name}.json"
            key = "keypoints"

        if not json_path.exists():
            raise FileNotFoundError(f"Missing annotation for sample {name}: {json_path}")

        with open(json_path, "r") as f:
            coords = np.array(json.load(f)[key], dtype=np.float32)

        coords[:, 0] += pad_left
        coords[:, 1] += pad_top
        gt_lookup[name] = coords

    return gt_lookup, pad_lookup


def compute_sample_metrics(pred_xy: np.ndarray, gt_xy: np.ndarray, pck5_thresh: float, pck10_thresh: float):
    mask = np.isfinite(pred_xy).all(axis=1) & np.isfinite(gt_xy).all(axis=1)
    if not np.any(mask):
        return None

    d = np.linalg.norm(pred_xy[mask] - gt_xy[mask], axis=1)
    return {
        "errors": d,
        "pck5": float(np.mean(d < pck5_thresh)),
        "pck10": float(np.mean(d < pck10_thresh)),
    }


def summarize_metrics(latencies_ms: List[float], errors: List[float], pck5_scores: List[float], pck10_scores: List[float]):
    def _safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    avg_latency = _safe_mean(latencies_ms)
    fps = float(1000.0 / avg_latency) if np.isfinite(avg_latency) and avg_latency > 0 else float("nan")

    return {
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "mpjpe_pixels": _safe_mean(errors),
        "pck_accuracy": _safe_mean(pck5_scores) * 100.0 if pck5_scores else float("nan"),
        "pck_10%_accuracy": _safe_mean(pck10_scores) * 100.0 if pck10_scores else float("nan"),
    }


def print_results(label: str, results_dict: Dict[str, float]):
    def _fmt(value: float) -> str:
        return f"{value:.2f}" if np.isfinite(value) else "n/a"

    print("\n" + "=" * 34)
    print(f"{label.center(34)}")
    print("=" * 34)
    print(f"{'Metric':<25} | {'Value':<8}")
    print("-" * 36)
    print(f"{'Avg. Latency (ms)':<25} | {_fmt(results_dict['avg_latency_ms']):<8}")
    print(f"{'Theoretical FPS':<25} | {_fmt(results_dict['fps']):<8}")
    print(f"{'Avg. Error (pixels)':<25} | {_fmt(results_dict['mpjpe_pixels']):<8}")
    print(f"{'PCK@5% Accuracy (%)':<25} | {_fmt(results_dict['pck_accuracy']):<8}")
    print(f"{'PCK@10% Accuracy (%)':<25} | {_fmt(results_dict['pck_10%_accuracy']):<8}")
    print("=" * 34)


def evaluate_pytorch_model(
    model_path: str,
    model_arch: str,
    val_loader: DataLoader,
    val_filenames: List[str],
    gt_lookup: Dict[str, np.ndarray],
    device: torch.device,
    output_res: Tuple[int, int],
) -> Dict[str, float]:
    model = build_model(model_arch, device)
    state = torch.load(model_path, map_location=device)
    load_res = model.load_state_dict(state, strict=False)
    if load_res.missing_keys:
        print(f"[WARN] Missing keys when loading state_dict: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        print(f"[WARN] Unexpected keys when loading state_dict: {load_res.unexpected_keys}")
    model.eval()

    pck5_thresh = 0.05 * max(output_res)
    pck10_thresh = 0.10 * max(output_res)

    latencies_ms: List[float] = []
    errors: List[float] = []
    pck5_scores: List[float] = []
    pck10_scores: List[float] = []

    idx_offset = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch[0].to(device)
            batch_size = inputs.size(0)
            batch_names = val_filenames[idx_offset : idx_offset + batch_size]
            idx_offset += batch_size

            start_time = time.perf_counter()
            logits = model(inputs)
            dt_ms = (time.perf_counter() - start_time) * 1000.0 / batch_size
            latencies_ms.append(dt_ms)

            predicted_heatmaps = torch.sigmoid(logits).cpu()
            pred_coords = get_coords_from_heatmaps(predicted_heatmaps).astype(np.float32)

            for i, name in enumerate(batch_names):
                metrics = compute_sample_metrics(pred_coords[i], gt_lookup[name], pck5_thresh, pck10_thresh)
                if metrics is None:
                    continue
                errors.extend(metrics["errors"])
                pck5_scores.append(metrics["pck5"])
                pck10_scores.append(metrics["pck10"])

    return summarize_metrics(latencies_ms, errors, pck5_scores, pck10_scores)


def evaluate_mediapipe_lite(
    val_filenames: List[str],
    data_dir: str,
    gt_lookup: Dict[str, np.ndarray],
    pad_lookup: Dict[str, Tuple[int, int]],
    output_res: Tuple[int, int],
    model_arch: str,
) -> Dict[str, float]:
    data_root = Path(data_dir)
    pck5_thresh = 0.05 * max(output_res)
    pck10_thresh = 0.10 * max(output_res)

    latencies_ms: List[float] = []
    errors: List[float] = []
    pck5_scores: List[float] = []
    pck10_scores: List[float] = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,  # Lite model
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for name in val_filenames:
            img_path = data_root / "tof" / f"{name}.png"
            frame = cv2.imread(str(img_path), cv2.IMREAD_COLOR)

            if frame is None:
                print(f"[WARN] Skipping sample {name}: no ToF image found.")
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            start_time = time.perf_counter()
            result = pose.process(rgb)
            latencies_ms.append((time.perf_counter() - start_time) * 1000.0)

            if result.pose_landmarks is None:
                continue

            mp_points = np.array([[lm.x * w, lm.y * h] for lm in result.pose_landmarks.landmark], dtype=np.float32)
            transformed = mp_points

            if model_arch == "heavy":
                kpt_xy = transformed
            else:
                kpt_xy = transformed[MP_TO_COCO]

            pad_left, pad_top = pad_lookup[name]
            kpt_xy[:, 0] += pad_left
            kpt_xy[:, 1] += pad_top

            metrics = compute_sample_metrics(kpt_xy, gt_lookup[name], pck5_thresh, pck10_thresh)
            if metrics is None:
                continue
            errors.extend(metrics["errors"])
            pck5_scores.append(metrics["pck5"])
            pck10_scores.append(metrics["pck10"])

    return summarize_metrics(latencies_ms, errors, pck5_scores, pck10_scores)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Pose Estimation Model and MediaPipe baseline.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model file.")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        choices=["heavy", "slim", "medium", "edge", "edge_v5", "edge_v6"],
        help="Specify model architecture.",
    )
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for evaluation (default 1).")
    parser.add_argument("--seed", type=int, default=720, help="Random split seed (default 720).")
    parser.add_argument(
        "--with-mediapipe",
        action="store_true",
        help="Also evaluate MediaPipe Lite baseline on the same validation split.",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset & deterministic split
    print(f"Loading data for '{args.arch}' architecture...")
    full_dataset = build_dataset(args.arch, args.data)
    output_res = getattr(full_dataset, "output_res", (240, 240))

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch, shuffle=False)

    # Recover filenames for the validation subset so we can align metrics and baselines
    val_indices = val_dataset.indices if hasattr(val_dataset, "indices") else list(range(len(val_dataset)))
    val_filenames = [full_dataset.file_list[i] for i in val_indices]

    gt_lookup, pad_lookup = load_ground_truth_keypoints(val_filenames, args.data, args.arch, output_res)

    torch_results = evaluate_pytorch_model(
        model_path=args.model,
        model_arch=args.arch,
        val_loader=val_loader,
        val_filenames=val_filenames,
        gt_lookup=gt_lookup,
        device=device,
        output_res=output_res,
    )
    print_results("PyTorch Model", torch_results)

    if args.with_mediapipe:
        mediapipe_results = evaluate_mediapipe_lite(
            val_filenames=val_filenames,
            data_dir=args.data,
            gt_lookup=gt_lookup,
            pad_lookup=pad_lookup,
            output_res=output_res,
            model_arch=args.arch,
        )
        print_results("MediaPipe Lite", mediapipe_results)
