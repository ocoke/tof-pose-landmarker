# evaluate_pytorch.py
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import time
import os
import sys
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model architectures and dataset
from model.model import PoseUNet
from model_slim.model import SlimPoseUNet # Make sure both are available
from model_edge.model import EdgePoseUNet
from model.dataset import PoseDataset
from model_slim.dataset import SlimPoseDataset # Correctly import the slim dataset
from model_slim.medium_model import MediumPoseUNet

def get_coords_from_heatmaps(heatmaps_tensor):
    """ Extracts (x,y) coordinates from a batch of heatmaps. """
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

def evaluate_pytorch_model(model_path, model_arch, val_loader, device):
    """ Evaluates a PyTorch model for speed and accuracy. """
    # --- Load Model ---
    print(f"\n--- Evaluating PyTorch Model: {os.path.basename(model_path)} ---")
    if model_arch == 'heavy':
        model = PoseUNet(n_channels=2, n_keypoints=33).to(device) # Assuming 33 kpts for heavy
    elif model_arch == 'slim':
        model = SlimPoseUNet(in_ch=2, n_kpts=17).to(device) # Assuming 17 kpts for slim
    elif model_arch == 'medium':
        model = MediumPoseUNet(in_ch=2, n_kpts=17).to(device)
    elif model_arch == 'edge':
        model = EdgePoseUNet(in_ch=2, n_kpts=17, width_mult=1.0, p_drop=0.05).to(device)
    else:
        raise ValueError("Unknown model architecture specified.")
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # --- Evaluation Loop ---
    inference_times = []
    all_errors = []
    pck_scores = []
    pck_10_scores = []
    pck_threshold = 12 # in pixels
    pck_10_threshold = 24 # in pixels

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            
            # Run Inference and measure time
            start_time = time.perf_counter()
            logits = model(inputs)
            inference_times.append((time.perf_counter() - start_time) * 1000)
            
            # Post-process
            predicted_heatmaps = torch.sigmoid(logits).cpu()
            pred_coords = get_coords_from_heatmaps(predicted_heatmaps)
            gt_coords = get_coords_from_heatmaps(targets)
            
            # Calculate Accuracy
            error = np.linalg.norm(pred_coords[0] - gt_coords[0], axis=1)
            all_errors.extend(error)
            pck_scores.append(np.mean(error < pck_threshold))
            pck_10_scores.append(np.mean(error < pck_10_threshold))

    # --- Print Results ---
    print_results({
        "avg_latency_ms": np.mean(inference_times),
        "fps": 1000 / np.mean(inference_times),
        "mpjpe_pixels": np.mean(all_errors),
        "pck_accuracy": np.mean(pck_scores) * 100,
        "pck_10%_accuracy": np.mean(pck_10_scores) * 100
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
    print(f"{'PCK@10% Accuracy (%)':<25} | {results_dict['pck_10%_accuracy']:.2f}")
    print("="*30)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Pose Estimation Model")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model file.")
    parser.add_argument("--arch", type=str, required=True, choices=['heavy', 'slim', 'medium', 'edge'], help="Specify model architecture ('heavy' for UNet, 'slim' for SlimPoseUNet).")
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CORRECTED DATASET LOADING ---
    print(f"Loading data for '{args.arch}' architecture...")
    if args.arch == 'heavy':
        # The heavy model uses 33 keypoints and the original PoseDataset
        full_dataset = PoseDataset(data_dir=args.data, augment=False, num_keypoints=33)
    elif args.arch == 'slim' or args.arch == 'medium' or args.arch == 'edge':
        # The slim model uses 17 keypoints and the new SlimPoseDataset
        full_dataset = SlimPoseDataset(data_dir=args.data, augment=False, num_keypoints=17)
    
    # The rest of the data splitting logic is the same
    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False)
    
    evaluate_pytorch_model(args.model, args.arch, val_loader, device)