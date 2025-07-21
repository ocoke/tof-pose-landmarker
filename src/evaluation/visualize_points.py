import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import os, sys
import argparse
import time

# Use TensorFlow's full lite interpreter
import tensorflow as tf
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import both of your model and dataset architectures
from model.model import PoseUNet
from model_slim.model import SlimPoseUNet
from model.dataset import PoseDataset
from model_slim.dataset import SlimPoseDataset

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

def main(args):
    # --- Data Loading (Corrected Logic) ---
    print(f"Loading data for a model with {args.kpts} keypoints...")
    if args.kpts == 33:
        full_dataset = PoseDataset(data_dir=args.data, augment=False, num_keypoints=33)
    elif args.kpts == 17:
        full_dataset = SlimPoseDataset(data_dir=args.data, augment=False, num_keypoints=17)
    else:
        raise ValueError(f"Invalid number of keypoints specified: {args.kpts}. Must be 17 or 33.")

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=True)
    print(f"Loaded {len(val_dataset)} samples for visualization.")

    # --- Model Loading ---
    if args.type == 'pytorch':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.arch == 'heavy':
            model = PoseUNet(n_channels=2, n_keypoints=args.kpts).to(device)
        else: # slim
            model = SlimPoseUNet(in_ch=2, n_kpts=args.kpts).to(device)
        model.load_state_dict(torch.load(args.model, map_location=device))
        model.eval()
        print(f"✅ PyTorch model '{args.arch}' loaded successfully.")
    
    elif args.type == 'tflite':
        interpreter = tf.lite.Interpreter(model_path=args.model)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()[0]
        output_details = interpreter.get_output_details()[0]
        print("✅ TFLite model loaded successfully.")

    # --- Visualization Loop ---
    num_shown = 0
    for inputs, targets in val_loader:
        if num_shown >= args.num_images:
            break
            
        input_np = inputs.numpy()
        
        # --- Run Inference ---
        if args.type == 'pytorch':
            with torch.no_grad():
                # Convert to tensor for PyTorch model
                logits = model(inputs.to(device))
                predicted_heatmaps = torch.sigmoid(logits).cpu().numpy()
        
        elif args.type == 'tflite':
            interpreter.set_tensor(input_details['index'], input_np)
            interpreter.invoke()
            predicted_heatmaps = interpreter.get_tensor(output_details['index'])
            if output_details['dtype'] in [np.int8, np.uint8]:
                scale, zero_point = output_details["quantization"]
                predicted_heatmaps = (predicted_heatmaps.astype(np.float32) - zero_point) * scale

        # --- Post-processing and Plotting ---
        pred_coords = get_coords_from_heatmaps(predicted_heatmaps)[0]
        gt_coords = get_coords_from_heatmaps(targets.numpy())[0]
        
        depth_map = input_np[0, 0, :, :]

        plt.figure(figsize=(8, 8))
        plt.imshow(depth_map, cmap='viridis')
        plt.scatter(gt_coords[:, 0], gt_coords[:, 1], s=40, c='lime', marker='o', label='Ground Truth')
        plt.scatter(pred_coords[:, 0], pred_coords[:, 1], s=40, c='red', marker='x', label='Prediction')
        
        plt.title(f"Final Prediction vs. Ground Truth\nModel: {os.path.basename(args.model)}")
        plt.legend()
        plt.axis('off')
        plt.show()
        
        num_shown += 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Visualize final keypoint predictions from a model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the model file (.pth or .tflite).")
    parser.add_argument("--type", type=str, required=True, choices=['pytorch', 'tflite'], help="Type of the model.")
    parser.add_argument("--arch", type=str, choices=['heavy', 'slim'], help="Specify architecture if model type is pytorch (required for pytorch).")
    parser.add_argument("--kpts", type=int, required=True, choices=[17, 33], help="Number of keypoints the model outputs (17 for slim, 33 for heavy).")
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory.")
    parser.add_argument("--num-images", type=int, default=5, help="Number of images to visualize.")
    args = parser.parse_args()
    
    # Argument validation
    if args.type == 'pytorch' and not args.arch:
        parser.error("--arch is required when --type is pytorch")
        
    main(args)