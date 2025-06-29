# dataset.py
import torch
from torch.utils.data import Dataset
import numpy as np
import os, json
from torchvision import transforms

# This function will be called by our Dataset to create the target heatmaps
def generate_heatmaps(keypoints, output_res, sigma=2):
    """
    Generates 2D Gaussian heatmaps for keypoints.

    Args:
        keypoints (np.array): A (num_keypoints, 2) array of (x, y) coordinates.
        output_res (tuple): The (height, width) of the output heatmaps.
        sigma (float): The standard deviation of the Gaussian kernel.

    Returns:
        torch.Tensor: A (num_keypoints, height, width) tensor of heatmaps.
    """
    heatmaps = np.zeros((keypoints.shape[0], output_res[0], output_res[1]), dtype=np.float32)
    for i, (x, y) in enumerate(keypoints):
        # Ensure keypoints are within bounds
        if x < 0 or x >= output_res[1] or y < 0 or y >= output_res[0]:
            continue

        # Create a grid of coordinates
        xx, yy = np.meshgrid(np.arange(output_res[1]), np.arange(output_res[0]))
        
        # Generate the Gaussian peak
        heatmap = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
        heatmaps[i] = heatmap
    
    return torch.from_numpy(heatmaps)


class PoseDataset(Dataset):
    def __init__(self, data_dir, num_keypoints=33, output_res=(240, 240)):
        self.data_dir = data_dir
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.confidence_dir = os.path.join(data_dir, 'confidence')
        self.pose_dir = os.path.join(data_dir, 'pose')
        
        self.file_list = [f.split('.')[0] for f in os.listdir(self.depth_dir)]
        self.num_keypoints = num_keypoints
        self.output_res = output_res

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]

        # --- Load Sensor Data ---
        depth_path = os.path.join(self.depth_dir, f"{filename}.npy")
        confidence_path = os.path.join(self.confidence_dir, f"{filename}.npy")

        depth_map = np.load(depth_path)
        confidence_map = np.load(confidence_path)

        # --- Load and Process Labels ---
        pose_path = os.path.join(self.pose_dir, f"{filename}.json")

        # NEW: Open and load the JSON data
        with open(pose_path, 'r') as f:
            pose_data = json.load(f)
        
        # NEW: Extract the 'transformed_points' list and convert to a numpy array
        # This gives us our (33, 2) array of (x, y) coordinates.
        keypoints_2d = np.array(pose_data['transformed_points'])


        # --- Pre-process Input (This part is the same) ---
        # Normalize
        depth_map = depth_map / 4000.0
        confidence_map = confidence_map / 255.0
        
        # Stack depth and confidence
        input_tensor = torch.from_numpy(np.stack([depth_map, confidence_map], axis=0)).float()

        # Pad to square
        _, h, w = input_tensor.shape
        pad_left = (self.output_res[1] - w) // 2
        pad_right = self.output_res[1] - w - pad_left
        pad_top = (self.output_res[0] - h) // 2
        pad_bottom = self.output_res[0] - h - pad_top
        padding = (pad_left, pad_top, pad_right, pad_bottom) 
        input_tensor = transforms.functional.pad(input_tensor, padding)

        # --- Prepare Labels (This part is the same logic) ---
        # Account for padding in the keypoint coordinates
        keypoints_2d[:, 0] += pad_left
        keypoints_2d[:, 1] += pad_top

        # Generate target heatmaps
        target_heatmaps = generate_heatmaps(keypoints_2d, self.output_res, sigma=2)

        return input_tensor, target_heatmaps