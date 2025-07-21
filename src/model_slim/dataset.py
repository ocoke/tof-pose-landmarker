import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import os, json
import matplotlib.pyplot as plt
import random
# FLIP_INDICES = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17, 20, 19, 22, 21, 24, 23, 26, 25, 28, 27, 30, 29, 32, 31]

# FLIP FOR COCO17
FLIP_INDICES = [
    0,   # nose ↔ nose
    2,1, # left_eye ↔ right_eye
    4,3, # left_ear ↔ right_ear
    6,5, # left_shoulder ↔ right_shoulder
    8,7, # left_elbow ↔ right_elbow
    10,9,# left_wrist ↔ right_wrist
    12,11,# left_hip ↔ right_hip
    14,13,# left_knee ↔ right_knee
    16,15 # left_ankle ↔ right_ankle
]

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
    def __init__(self, data_dir, num_keypoints=17, output_res=(240, 240), augment=False):
        self.data_dir = data_dir
        self.depth_dir = os.path.join(data_dir, 'depth')
        self.confidence_dir = os.path.join(data_dir, 'confidence')
        self.pose_dir = os.path.join(data_dir, 'pose_coco17')

        self.file_list = [f.split('.')[0] for f in os.listdir(self.depth_dir)]
        self.num_keypoints = num_keypoints
        self.output_res = output_res
        self.augment = augment

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
        # print(pose_data)
        # print(pose_data['keypoints'])
        keypoints_2d = np.array(pose_data['keypoints'])


        # --- Pre-process Input (This part is the same) ---
        # Normalize
        depth_map = depth_map / 4000.0
        confidence_map = confidence_map / 255.0

        # Stack depth and confidence
        input_tensor = torch.from_numpy(np.stack([depth_map, confidence_map], axis=0)).float()


        if self.augment:
            # 1. Random Horizontal Flip (50% chance)
            if random.random() > 0.5:
                # Flip the image tensor
                input_tensor = TF.hflip(input_tensor)
                # Flip the keypoint x-coordinates
                img_width = input_tensor.shape[2]
                keypoints_2d[:, 0] = img_width - 1 - keypoints_2d[:, 0]
                # Swap the left/right keypoint indices
                keypoints_2d = keypoints_2d[FLIP_INDICES]

            # 2. Random Rotation
            angle = (random.random() - 0.5) * 2 * 15 # Random angle between -15 and +15 degrees
            # Rotate image
            input_tensor = TF.rotate(input_tensor, angle)
            # Rotate keypoints around the image center

            # Get image center, explicitly creating a float32 tensor
            center = torch.tensor([input_tensor.shape[2] / 2, input_tensor.shape[1] / 2], dtype=torch.float32)

            # Create rotation matrix, explicitly creating a float32 tensor
            rot_mat = torch.tensor([
                [np.cos(np.radians(-angle)), -np.sin(np.radians(-angle))],
                [np.sin(np.radians(-angle)), np.cos(np.radians(-angle))]
            ], dtype=torch.float32)

            # Now all tensors in the operation below are torch.float32
            keypoints_tensor = torch.from_numpy(keypoints_2d).float() - center
            keypoints_tensor = torch.matmul(keypoints_tensor, rot_mat) + center
            keypoints_2d = keypoints_tensor.numpy()




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
        target_heatmaps = generate_heatmaps(keypoints_2d, self.output_res, sigma=5)

        return input_tensor, target_heatmaps