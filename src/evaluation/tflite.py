import argparse
import os
import sys
import time
from typing import Dict, List

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import DataLoader, random_split

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v7.dataset import PoseDatasetV7


def get_coords_from_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    batch_size, num_keypoints, height, width = heatmaps.shape
    coords = np.zeros((batch_size, num_keypoints, 2), dtype=np.int32)
    for batch_idx in range(batch_size):
        for keypoint_idx in range(num_keypoints):
            flat_idx = np.argmax(heatmaps[batch_idx, keypoint_idx])
            coords[batch_idx, keypoint_idx] = [flat_idx % width, flat_idx // width]
    return coords


def summarize(latencies_ms: List[float], errors: List[float], pck5_scores: List[float], pck10_scores: List[float]) -> Dict[str, float]:
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else float("nan")
    fps = float(1000.0 / avg_latency) if np.isfinite(avg_latency) and avg_latency > 0 else float("nan")
    return {
        "avg_latency_ms": avg_latency,
        "fps": fps,
        "mpjpe_pixels": float(np.mean(errors)) if errors else float("nan"),
        "pck_accuracy": float(np.mean(pck5_scores) * 100.0) if pck5_scores else float("nan"),
        "pck_10%_accuracy": float(np.mean(pck10_scores) * 100.0) if pck10_scores else float("nan"),
    }


def print_results(metrics: Dict[str, float]) -> None:
    print("\n" + "=" * 34)
    print("TFLite Model".center(34))
    print("=" * 34)
    print(f"{'Metric':<25} | {'Value':<8}")
    print("-" * 36)
    print(f"{'Avg. Latency (ms)':<25} | {metrics['avg_latency_ms']:.2f}")
    print(f"{'Theoretical FPS':<25} | {metrics['fps']:.2f}")
    print(f"{'Avg. Error (pixels)':<25} | {metrics['mpjpe_pixels']:.2f}")
    print(f"{'PCK@5% Accuracy (%)':<25} | {metrics['pck_accuracy']:.2f}")
    print(f"{'PCK@10% Accuracy (%)':<25} | {metrics['pck_10%_accuracy']:.2f}")
    print("=" * 34)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a v7 TFLite pose model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the .tflite file.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size.")
    parser.add_argument("--conf-hi", type=float, default=350.0, help="Confidence clipping constant.")
    args = parser.parse_args()

    dataset = PoseDatasetV7(args.data, augment=False, min_valid_keypoints=0, conf_hi=args.conf_hi)
    val_size = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    loader = DataLoader(val_dataset, batch_size=args.batch, shuffle=False)

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    latencies_ms: List[float] = []
    errors: List[float] = []
    pck5_scores: List[float] = []
    pck10_scores: List[float] = []

    for inputs, _targets, gt_kpts, valid_mask, _meta in loader:
        input_data = inputs.numpy()
        start = time.perf_counter()
        interpreter.set_tensor(input_details["index"], input_data)
        interpreter.invoke()
        latencies_ms.append((time.perf_counter() - start) * 1000.0 / inputs.shape[0])

        predicted_heatmaps = interpreter.get_tensor(output_details["index"])
        if output_details["dtype"] in (np.int8, np.uint8):
            scale, zero_point = output_details["quantization"]
            predicted_heatmaps = (predicted_heatmaps.astype(np.float32) - zero_point) * scale

        pred_coords = get_coords_from_heatmaps(predicted_heatmaps)
        gt_coords = gt_kpts.numpy()
        valid_np = valid_mask.numpy().astype(bool)
        for batch_idx in range(inputs.shape[0]):
            mask = valid_np[batch_idx]
            if not mask.any():
                continue
            sample_dists = np.linalg.norm(pred_coords[batch_idx][mask] - gt_coords[batch_idx][mask], axis=1)
            errors.extend(sample_dists.tolist())
            pck5_scores.append(float(np.mean(sample_dists <= 12.0)))
            pck10_scores.append(float(np.mean(sample_dists <= 24.0)))

    print_results(summarize(latencies_ms, errors, pck5_scores, pck10_scores))


if __name__ == "__main__":
    main()
