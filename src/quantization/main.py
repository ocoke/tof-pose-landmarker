import argparse
import os
import sys

import ai_edge_torch
import tensorflow as tf
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model_v7.dataset import PoseDatasetV7
from model_v7.model import EdgePoseUNetV7


def representative_dataset_gen(data_dir: str, conf_hi: float):
    dataset = PoseDatasetV7(data_dir=data_dir, augment=False, min_valid_keypoints=0, conf_hi=conf_hi)
    steps = min(100, len(dataset))
    print(f"Generating representative data from {steps} samples...")
    for idx in range(steps):
        input_tensor, _targets, _gt, _mask, _meta = dataset[idx]
        yield [input_tensor.unsqueeze(0).numpy()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the v7 PyTorch model to TFLite.")
    parser.add_argument("--model", type=str, required=True, help="Path to the PyTorch checkpoint.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root for representative samples.")
    parser.add_argument("--output", type=str, default="./models/v7_float16.tflite", help="Output TFLite path.")
    parser.add_argument("--conf-hi", type=float, default=350.0, help="Confidence clipping constant.")
    args = parser.parse_args()

    model = EdgePoseUNetV7(in_ch=3, n_kpts=17, width_mult=1.15)
    model.load_state_dict(torch.load(args.model, map_location="cpu"), strict=False)
    model.eval()

    converter_flags = {
        "optimizations": [tf.lite.Optimize.DEFAULT],
        "target_spec.supported_types": [tf.float16],
    }

    edge_model = ai_edge_torch.convert(
        model,
        (torch.randn(1, 3, 240, 240),),
        _ai_edge_converter_flags=converter_flags,
    )

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    edge_model.export(args.output)
    original_size = os.path.getsize(args.model)
    converted_size = os.path.getsize(args.output)
    print(f"Saved {args.output}")
    print(f"Original PyTorch model size: {original_size / (1024 * 1024):.2f} MB")
    print(f"Converted TFLite model size: {converted_size / (1024 * 1024):.2f} MB")
    print("Representative dataset helper available via model_v7.dataset.PoseDatasetV7")


if __name__ == "__main__":
    main()
