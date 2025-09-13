# evaluate_tflite.py
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import os, sys
import argparse
import tensorflow as tf

# Make sure your dataset.py is accessible

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model_slim.dataset import PoseDataset

def get_coords_from_heatmaps(heatmaps):
    """ Extracts (x,y) coordinates from a batch of heatmaps. """
    batch_size, num_keypoints, height, width = heatmaps.shape
    coords = np.zeros((batch_size, num_keypoints, 2), dtype=np.int32)
    for i in range(batch_size):
        for k in range(num_keypoints):
            heatmap = heatmaps[i, k, :, :]
            max_index = np.argmax(heatmap)
            y = max_index // width
            x = max_index % width
            coords[i, k] = [x, y]
    return coords

def evaluate_tflite_model(model_path, val_loader, num_threads=4, img_res=(240,240), kpts=17):
    print(f"\n--- Evaluating TFLite Model: {os.path.basename(model_path)} ---")

    # Try tflite_runtime first (lighter on Pi), fall back to TF if available.
    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        from tensorflow.lite.python.interpreter import Interpreter

    interpreter = Interpreter(model_path=model_path, num_threads=num_threads)
    interpreter.allocate_tensors()
    input_details  = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Print a helpful one-liner so we know what the model expects
    print("Input:", input_details["shape"], input_details["dtype"])
    print("Output:", output_details["shape"], output_details["dtype"])

    # Warm-up (stabilize XNNPACK)
    H, W = img_res
    dummy = np.zeros((1, H, W, input_details["shape"][-1]), dtype=np.float32)
    if input_details["dtype"] == np.int8:
        s, z = input_details["quantization"]
        dummy = np.clip(np.round(dummy / s + z), -128, 127).astype(np.int8)
    elif input_details["dtype"] == np.float16:
        dummy = dummy.astype(np.float16)
    interpreter.set_tensor(input_details["index"], dummy)
    for _ in range(5):
        interpreter.invoke()

    inference_times = []
    all_errors = []
    pck_scores = []
    pck_threshold = 0.05 * max(H, W)  # = 12px for 240

    for inputs, targets in val_loader:
        # PyTorch (NCHW) -> TFLite (NHWC)
        x = inputs.numpy().transpose(0, 2, 3, 1).astype(np.float32)  # (B,H,W,C)
        B = x.shape[0]

        # Quantize inputs if needed
        if input_details['dtype'] == np.int8 or input_details['dtype'] == np.uint8:
            in_scale, in_zero = input_details['quantization']
            x = np.clip(np.round(x / in_scale + in_zero), -128, 127).astype(np.int8)
        elif input_details['dtype'] == np.float16:
            x = x.astype(np.float16)  # some FP16 models expect FP16 inputs
        else:
            x = x.astype(np.float32)  # most FP16-weight models still take FP32 input

        # The input shape is usually fixed to (1,H,W,C). Feed sample-by-sample for simplicity.
        start = time.perf_counter()
        for i in range(B):
            xi = x[i:i+1]
            interpreter.set_tensor(input_details['index'], xi)
            interpreter.invoke()
        dt_ms = (time.perf_counter() - start) * 1000.0 / B
        inference_times.append(dt_ms)

        # Read output for the last sample (or loop if you want exact per-sample logs)
        y = interpreter.get_tensor(output_details['index'])  # (1,H,W,K) or (B,H,W,K)

        # Dequantize outputs if INT8/UINT8
        if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
            out_scale, out_zero = output_details["quantization"]
            y = (y.astype(np.float32) - out_zero) * out_scale

        # Make sure it's (B,K,H,W) for your helper
        if y.ndim == 3:              # (H,W,K)
            y = y[None, ...]         # -> (1,H,W,K)
        if y.shape[-1] == kpts:
            y = y.transpose(0, 3, 1, 2)  # NHWC -> NCHW -> (B,K,H,W)

        predicted_heatmaps = y
        pred_coords = get_coords_from_heatmaps(predicted_heatmaps)

        # Targets in your Dataset are (B,K,H,W) already -> OK for your helper
        gt_coords = get_coords_from_heatmaps(targets.numpy())

        # Accuracy
        err = np.linalg.norm(pred_coords[0] - gt_coords[0], axis=1)
        all_errors.extend(err)
        pck_scores.append(np.mean(err < pck_threshold))

    print_results({
        "avg_latency_ms": np.mean(inference_times),
        "fps": 1000.0 / np.mean(inference_times),
        "mpjpe_pixels": np.mean(all_errors),
        "pck_accuracy": np.mean(pck_scores) * 100.0
    })


def print_results(results_dict):
    """ Prints a formatted table of results. """
    print("\n" + "="*30)
    print("      PERFORMANCE METRICS")
    print("="*30)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-"*43)
    print(f"{'Avg. Latency (ms)':<25} | {results_dict['avg_latency_ms']:.2f}")
    print(f"{'Theoretical FPS':<25} | {results_dict['fps']:.2f}")
    print(f"{'Avg. Error (pixels)':<25} | {results_dict['mpjpe_pixels']:.2f}")
    print(f"{'PCK Accuracy (%)':<25} | {results_dict['pck_accuracy']:.2f}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a TFLite Pose Estimation Model")
    parser.add_argument("--model", type=str, required=True, help="Path to the .tflite model file.")
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory.")
    parser.add_argument("--kpts", type=int, default=17, help="Number of keypoints the model outputs.")
    args = parser.parse_args()

    # Create the validation set
    full_dataset = PoseDataset(data_dir=args.data, augment=False, num_keypoints=args.kpts)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    
    evaluate_tflite_model(args.model, val_loader)