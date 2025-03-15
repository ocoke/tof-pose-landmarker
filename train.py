import os
import json
import math
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import torchvision.transforms.functional as TF
from torch.optim.lr_scheduler import StepLR

from model import PoseResNet, BasicBlock

# generate ground truth heatmap
def generate_gaussian_heatmap(center, output_size, sigma=2):
    """
    Generates a single gaussian heatmap.

    Args:
        center (tuple): (x,y) normalized coordinates in range [0,1] corresponding to ground truth.
        output_size (tuple): (H, W) size of the heatmap.
        sigma (float): standard deviation of the Gaussian.
    Returns:
        heatmap (np.array): generated heatmap.
    """
    H, W = output_size
    # Convert normalized coordinates to pixel coordinates in heatmap space
    x0 = center[0] * (W - 1)
    y0 = center[1] * (H - 1)
    # Create meshgrid
    xs = np.arange(0, W, dtype=np.float32)
    ys = np.arange(0, H, dtype=np.float32)
    ys = ys.reshape(-1, 1)
    
    heatmap = np.exp(- ((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * sigma ** 2))
    return heatmap

def generate_target_heatmaps(keypoints, output_size, sigma=2):
    """
    Generate multilayer heatmaps for all keypoints.

    Args:
        keypoints (Tensor): Tensor of shape (num_keypoints, 2) with normalized coordinates.
        output_size (tuple): (H, W) of the heatmap output.
        sigma (float): Gaussian sigma.
    Returns:
        heatmaps (Tensor): Tensor of shape (num_keypoints, H, W)
    """
    num_keypoints = keypoints.shape[0]
    heatmaps = np.zeros((num_keypoints, output_size[0], output_size[1]), dtype=np.float32)
    keypoints_np = keypoints.cpu().numpy() if isinstance(keypoints, torch.Tensor) else keypoints
    for i in range(num_keypoints):
        # if keypoint (x,y) are both zero, assume not visible and leave heatmap as zeros
        if keypoints_np[i, 0] == 0 and keypoints_np[i, 1] == 0:
            continue
        heatmaps[i] = generate_gaussian_heatmap(keypoints_np[i], output_size, sigma)
    return torch.tensor(heatmaps)


class PoseDataset(Dataset):
    def __init__(self, image_dir, ann_dir, transform=None, heatmap_size=(12, 16), sigma=2):
        """
        Args:
            image_dir (str): Path to directory with grayscale images.
            ann_dir (str): Path to directory with JSON annotations.
            transform (callable, optional): Optional transform to be applied on an image.
            heatmap_size (tuple): The size (H,W) to which ground truth heatmaps are generated.
            sigma (float): Standard deviation for gaussian heatmap generation.
        """
        self.image_dir = image_dir
        self.ann_dir = ann_dir
        self.transform = transform
        self.heatmap_size = heatmap_size
        self.sigma = sigma
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
        

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image and resize to 240x180 (width x height)
        filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, filename)
        image = Image.open(image_path).convert('L')
        image = image.resize((240, 180))
        image = TF.hflip(image) # if needed flip the image
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)
            
        # Load annotation and use the first 15 landmarks in order.
        ann_filename = filename.replace('.png', '.json')
        ann_path = os.path.join(self.ann_dir, ann_filename)
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
        keypoints = []
        # Use the first set of landmarks.
        landmarks = annotation.get("pose_landmarks", [])
        if len(landmarks) > 0 and len(landmarks[0]) >= 15:
            for i in range(15):
                kp = landmarks[0][i]
                keypoints.append([kp['x'], kp['y']])
        else:
            keypoints = [[0.0, 0.0] for _ in range(15)]
        keypoints = torch.tensor(keypoints, dtype=torch.float32)
        target_heatmaps = generate_target_heatmaps(keypoints, self.heatmap_size, self.sigma)
        return image, target_heatmaps, keypoints


def train():
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3
    
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print('[INFO] Model moved to accelerator:', torch.accelerator.current_accelerator())

    transform  = transforms.ToTensor()

    dataset = PoseDataset(
        image_dir="images",
        ann_dir="annotations",
        transform=transform,
        heatmap_size=(12, 16),
        sigma=1.5
    )

    # 80% imaegs for training, 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = PoseResNet(BasicBlock, [2, 2, 2, 2], num_joints=15)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        for images, target_heatmaps, _ in train_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            # print("Output shape:", outputs.shape)
            # print("Target shape:", target_heatmaps.shape)
            loss = criterion(outputs, target_heatmaps)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)
        scheduler.step()
        print(f"Epoch {epoch+1}/{num_epochs}, LR: {scheduler.get_last_lr()[0]}, Loss: {total_loss:.4f}")
        
        # validate_and_visualize(model, val_loader, device)
        if (epoch+1) % 5 == 0:
            validate_and_visualize(model, val_loader, device)

    print("Finished Training")
    torch.save(model.state_dict(), "pose_resnet.pth")

def validate_and_visualize(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, target_heatmaps, _ in val_loader:
            images = images.to(device)
            target_heatmaps = target_heatmaps.to(device)
            preds = model(images)
            loss = F.mse_loss(preds, target_heatmaps)
            total_loss += loss.item()
    total_loss /= len(val_loader)
    print(f"Validation Loss: {total_loss:.4f}")
    
    # Visualize a batch sample
    images, target_heatmaps, _ = next(iter(val_loader))
    images = images.to(device)
    preds = model(images)

    random_image = random.randint(0, images.size(0) - 1)

    image_np = images[random_image].cpu().squeeze().numpy()
    gt_heat_np = target_heatmaps[random_image].cpu().numpy()  # (15, H, W)
    pred_heat_np = preds[random_image].cpu().detach().numpy()
    
    # For visualization, sum the keypoint heatmaps
    gt_sum = np.sum(gt_heat_np, axis=0)
    pred_sum = np.sum(pred_heat_np, axis=0)
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(image_np, cmap="gray")
    plt.title("Input Image")
    plt.subplot(1, 3, 2)
    plt.imshow(gt_sum, cmap="jet")
    plt.title("GT Heatmaps Sum")
    plt.subplot(1, 3, 3)
    plt.imshow(pred_sum, cmap="jet")
    plt.title("Predicted Heatmaps Sum")
    plt.show()

if __name__ == "__main__":
    train()