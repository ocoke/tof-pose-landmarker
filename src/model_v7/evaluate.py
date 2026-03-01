import argparse
import os
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_v7.dataset import PoseDatasetV7
    from model_v7.manifest import DEFAULT_MANIFEST, build_or_load_manifest, get_records_for_split
    from model_v7.model import EdgePoseUNetV7
else:
    from .dataset import PoseDatasetV7
    from .manifest import DEFAULT_MANIFEST, build_or_load_manifest, get_records_for_split
    from .model import EdgePoseUNetV7


def soft_argmax_2d(logits: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    batch, num_keypoints, height, width = logits.shape
    probs = torch.softmax(logits.view(batch, num_keypoints, -1) * beta, dim=-1).view(batch, num_keypoints, height, width)
    xs = torch.linspace(0, width - 1, width, device=logits.device, dtype=torch.float32).view(1, 1, 1, width)
    ys = torch.linspace(0, height - 1, height, device=logits.device, dtype=torch.float32).view(1, 1, height, 1)
    pred_x = (probs * xs).sum(dim=(2, 3))
    pred_y = (probs * ys).sum(dim=(2, 3))
    return torch.stack([pred_x, pred_y], dim=-1)


def summarize_metrics(latencies_ms: List[float], errors: List[float], pck5_scores: List[float], pck10_scores: List[float]) -> Dict[str, float]:
    def safe_mean(values: List[float]) -> float:
        return float(np.mean(values)) if values else float("nan")

    avg_latency = safe_mean(latencies_ms)
    fps = float(1000.0 / avg_latency) if np.isfinite(avg_latency) and avg_latency > 0 else float("nan")
    return {
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "mpjpe_pixels": safe_mean(errors),
        "pck_accuracy": safe_mean(pck5_scores) * 100.0 if pck5_scores else float("nan"),
        "pck_10%_accuracy": safe_mean(pck10_scores) * 100.0 if pck10_scores else float("nan"),
    }


def evaluate_pytorch_model(model: EdgePoseUNetV7, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    latencies_ms: List[float] = []
    errors: List[float] = []
    pck5_scores: List[float] = []
    pck10_scores: List[float] = []

    with torch.no_grad():
        for inputs, _targets, gt_kpts, valid_mask, _meta in loader:
            inputs = inputs.to(device)
            gt_kpts = gt_kpts.to(device)
            valid_mask = valid_mask.to(device)

            start = time.perf_counter()
            logits = model(inputs)
            latencies_ms.append((time.perf_counter() - start) * 1000.0 / inputs.size(0))

            pred_xy = soft_argmax_2d(logits)
            dists = torch.norm(pred_xy - gt_kpts, dim=-1)
            for idx in range(inputs.size(0)):
                mask = valid_mask[idx]
                if not mask.any():
                    continue
                sample_dists = dists[idx][mask].detach().cpu().numpy()
                errors.extend(sample_dists.tolist())
                pck5_scores.append(float(np.mean(sample_dists <= 12.0)))
                pck10_scores.append(float(np.mean(sample_dists <= 24.0)))

    return summarize_metrics(latencies_ms, errors, pck5_scores, pck10_scores)


def evaluate_mediapipe_lite(dataset: PoseDatasetV7, data_dir: str) -> Optional[Dict[str, float]]:
    try:
        import cv2
        import mediapipe as mp
    except ModuleNotFoundError:
        print("[WARN] MediaPipe baseline requested but dependencies are unavailable.")
        return None

    latencies_ms: List[float] = []
    errors: List[float] = []
    pck5_scores: List[float] = []
    pck10_scores: List[float] = []

    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=0,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:
        for idx, record in enumerate(dataset.records):
            sample_id = record["sample_id"]
            img_path = os.path.join(data_dir, "tof", f"{sample_id}.png")
            frame = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            result_start = time.perf_counter()
            result = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            latencies_ms.append((time.perf_counter() - result_start) * 1000.0)
            if result.pose_landmarks is None:
                continue

            _, _, gt_kpts, valid_mask, _meta = dataset[idx]
            valid_mask_np = valid_mask.numpy().astype(bool)
            if not valid_mask_np.any():
                continue

            mp_points = np.array(
                [[lm.x * frame.shape[1], lm.y * frame.shape[0]] for lm in result.pose_landmarks.landmark],
                dtype=np.float32,
            )
            mp_to_coco = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
            pred_xy = mp_points[mp_to_coco]
            pad_left = (dataset.output_res[1] - frame.shape[1]) // 2
            pad_top = (dataset.output_res[0] - frame.shape[0]) // 2
            pred_xy[:, 0] += pad_left
            pred_xy[:, 1] += pad_top
            dists = np.linalg.norm(pred_xy[valid_mask_np] - gt_kpts.numpy()[valid_mask_np], axis=1)
            errors.extend(dists.tolist())
            pck5_scores.append(float(np.mean(dists <= 12.0)))
            pck10_scores.append(float(np.mean(dists <= 24.0)))

    return summarize_metrics(latencies_ms, errors, pck5_scores, pck10_scores)


def print_results(label: str, metrics: Dict[str, float]) -> None:
    def fmt(value: float) -> str:
        return f"{value:.2f}" if np.isfinite(value) else "n/a"

    print("\n" + "=" * 34)
    print(f"{label.center(34)}")
    print("=" * 34)
    print(f"{'Metric':<25} | {'Value':<8}")
    print("-" * 36)
    print(f"{'Avg. Latency (ms)':<25} | {fmt(metrics['avg_latency_ms']):<8}")
    print(f"{'Theoretical FPS':<25} | {fmt(metrics['fps']):<8}")
    print(f"{'Avg. Error (pixels)':<25} | {fmt(metrics['mpjpe_pixels']):<8}")
    print(f"{'PCK@5% Accuracy (%)':<25} | {fmt(metrics['pck_accuracy']):<8}")
    print(f"{'PCK@10% Accuracy (%)':<25} | {fmt(metrics['pck_10%_accuracy']):<8}")
    print("=" * 34)


def build_loader_from_split(
    data_dir: str,
    split: str,
    batch_size: int,
    manifest_path: Optional[str],
    scene_map_path: Optional[str],
    conf_hi: float,
    rebuild_manifest: bool,
) -> Tuple[DataLoader, PoseDatasetV7]:
    manifest_path = manifest_path or os.path.join(data_dir, DEFAULT_MANIFEST)
    records = build_or_load_manifest(data_dir, manifest_path=manifest_path, scene_map_path=scene_map_path, rebuild=rebuild_manifest)
    if split == "legacy_random":
        full_dataset = PoseDatasetV7(data_dir, records=records, augment=False, min_valid_keypoints=0, conf_hi=conf_hi)
        val_size = int(0.2 * len(full_dataset))
        train_size = len(full_dataset) - val_size
        _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
        subset_records = [full_dataset.records[i] for i in val_dataset.indices]
        dataset = PoseDatasetV7(data_dir, records=subset_records, augment=False, min_valid_keypoints=0, conf_hi=conf_hi)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return loader, dataset

    split_records = get_records_for_split(records, split)
    dataset = PoseDatasetV7(data_dir, records=split_records, augment=False, min_valid_keypoints=0, conf_hi=conf_hi)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return loader, dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the canonical v7 pose model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model file.")
    parser.add_argument("--arch", type=str, default="edge_v7", choices=["edge_v7"], help="Model architecture.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to the manifest file.")
    parser.add_argument("--scene-map", type=str, default=None, help="Optional JSON mapping sample_id to scene_id.")
    parser.add_argument(
        "--split",
        type=str,
        default="val_seen",
        choices=["train_core", "val_seen", "val_unseen", "test_unseen", "legacy_random"],
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--conf-hi", type=float, default=350.0, help="Confidence clipping constant.")
    parser.add_argument("--with-mediapipe", action="store_true", help="Also run the MediaPipe Lite baseline.")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Recompute the manifest instead of reusing it.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader, dataset = build_loader_from_split(
        args.data,
        args.split,
        args.batch,
        args.manifest,
        args.scene_map,
        args.conf_hi,
        args.rebuild_manifest,
    )

    model = EdgePoseUNetV7(in_ch=3, n_kpts=17, width_mult=1.15).to(device)
    state = torch.load(args.model, map_location=device)
    load_res = model.load_state_dict(state, strict=False)
    if load_res.missing_keys:
        print(f"[WARN] Missing keys when loading state_dict: {load_res.missing_keys}")
    if load_res.unexpected_keys:
        print(f"[WARN] Unexpected keys when loading state_dict: {load_res.unexpected_keys}")

    metrics = evaluate_pytorch_model(model, loader, device)
    print_results("PyTorch Model", metrics)

    if args.with_mediapipe:
        baseline = evaluate_mediapipe_lite(dataset, args.data)
        if baseline:
            print_results("MediaPipe Lite", baseline)


if __name__ == "__main__":
    main()
