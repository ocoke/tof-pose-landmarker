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
from model.dataset import PoseDataset

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

def evaluate_tflite_model(model_path, val_loader):
    """ Evaluates a TFLite model for speed and accuracy. """
    print(f"\n--- Evaluating TFLite Model: {os.path.basename(model_path)} ---")
    
    # --- Load Model ---
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # --- Evaluation Loop ---
    inference_times = []
    all_errors = []
    pck_scores = []
    pck_threshold = 12 # in pixels

    for inputs, targets in val_loader:
        input_data = inputs.numpy()
        
        # Run Inference and measure time
        start_time = time.perf_counter()
        interpreter.set_tensor(input_details['index'], input_data)
        interpreter.invoke()
        inference_times.append((time.perf_counter() - start_time) * 1000)
        
        predicted_heatmaps = interpreter.get_tensor(output_details['index'])
        
        # Dequantize if the model output is INT8
        if output_details['dtype'] == np.int8 or output_details['dtype'] == np.uint8:
            output_scale, output_zero_point = output_details["quantization"]
            predicted_heatmaps = (predicted_heatmaps.astype(np.float32) - output_zero_point) * output_scale
            
        pred_coords = get_coords_from_heatmaps(predicted_heatmaps)
        gt_coords = get_coords_from_heatmaps(targets.numpy())
        
        # Calculate Accuracy
        error = np.linalg.norm(pred_coords[0] - gt_coords[0], axis=1)
        all_errors.extend(error)
        pck_scores.append(np.mean(error < pck_threshold))

    # --- Print Results ---
    print_results({
        "avg_latency_ms": np.mean(inference_times),
        "fps": 1000 / np.mean(inference_times),
        "mpjpe_pixels": np.mean(all_errors),
        "pck_accuracy": np.mean(pck_scores) * 100
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
    parser.add_argument("--kpts", type=int, default=33, help="Number of keypoints the model outputs.")
    args = parser.parse_args()

    # Create the validation set
    full_dataset = PoseDataset(data_dir=args.data, augment=False, num_keypoints=args.kpts)
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    
    evaluate_tflite_model(args.model, val_loader)