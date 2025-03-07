# Pose Landmark Model for Gray-scale Images
# By Jun Yu, Feb 22, 2025

import os
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt

# Heatmap
from model import HeatmapPoseModel
from dataset import PoseDataset

def heatmaps_to_coordinates(heatmaps):
    """Convert heatmaps to keypoint coordinates using argmax."""
    batch_size = heatmaps.size(0)
    num_keypoints = heatmaps.size(1)
    height = heatmaps.size(2)
    width = heatmaps.size(3)
    
    # Reshape heatmaps to find argmax
    heatmaps_flat = heatmaps.reshape(batch_size, num_keypoints, -1)
    
    # Option 1: Hard argmax (less accurate)
    # max_val, max_idx = torch.max(heatmaps_flat, dim=2)
    # x = max_idx % width
    # y = max_idx // width
    
    # Option 2: Soft-argmax (differentiable, more accurate)
    heatmaps_softmax = F.softmax(heatmaps_flat, dim=2)
    
    # Create coordinate reference maps
    x_ref = torch.arange(0, width).float().to(heatmaps.device)
    y_ref = torch.arange(0, height).float().to(heatmaps.device)
    
    y_map, x_map = torch.meshgrid(y_ref, x_ref, indexing='ij')
    x_map = x_map.reshape(-1)  # Flatten to 1D
    y_map = y_map.reshape(-1)
    
    # Weight coordinates by probabilities
    x_coord = torch.sum(heatmaps_softmax * x_map.unsqueeze(0).unsqueeze(0), dim=2)
    y_coord = torch.sum(heatmaps_softmax * y_map.unsqueeze(0).unsqueeze(0), dim=2)
    
    # Normalize to 0-1
    x_coord = x_coord / (width - 1)
    y_coord = y_coord / (height - 1)
    
    # Stack coordinates
    coords = torch.stack((x_coord, y_coord), dim=2)
    
    return coords

def calculate_pck(pred_coords, gt_coords, threshold=0.2):
    """
    Calculate PCK (Percentage of Correct Keypoints).
    
    Args:
        pred_coords: Predicted coordinates, shape (batch_size, num_keypoints, 2)
        gt_coords: Ground truth coordinates, shape (batch_size, num_keypoints, 2)
        threshold: Distance threshold as a fraction of torso size
    
    Returns:
        PCK value
    """
    batch_size = pred_coords.shape[0]
    num_keypoints = pred_coords.shape[1]
    
    # Calculate torso size for each sample (distance between hip and shoulder)
    # For simplicity, using a fixed normalization:
    # distance between left shoulder (1) and right hip (8)
    torso_sizes = torch.sqrt(
        ((gt_coords[:, 1, :] - gt_coords[:, 8, :])**2).sum(dim=1)
    )
    
    # Calculate distances between predictions and ground truth
    distances = torch.sqrt(((pred_coords - gt_coords)**2).sum(dim=2))
    
    # Normalize by torso size
    normalized_distances = distances / torso_sizes.unsqueeze(1)
    
    # Count correct keypoints
    correct_keypoints = (normalized_distances < threshold).float().sum()
    
    # Calculate PCK
    pck = correct_keypoints / (batch_size * num_keypoints)
    
    return pck.item()

def train_model(model, train_loader, val_loader, device, num_epochs=30):
    # Setup optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=3, verbose=True
    )
    
    best_val_pck = 0.0
    
    for epoch in range(num_epochs):
        print('[INFO] Start training epoch:', epoch)
        # Training phase
        model.train()
        train_loss = 0.0
        
        for images, keypoints in train_loader:
            # Move to device
            images = images.to(device)
            print('[INFO] Load Image: ', images.shape)
            # Generate target heatmaps
            target_heatmaps = torch.zeros((images.size(0), 15, 64, 64)).to(device)
            for b in range(images.size(0)):
                for k in range(keypoints.size(1)):
                    x, y = keypoints[b, k, 0].item(), keypoints[b, k, 1].item()
                    # Skip if keypoint is not visible
                    if x == 0 and y == 0:
                        continue
                    
                    # Convert to heatmap coordinates
                    x = int(x * 64)
                    y = int(y * 64)
                    
                    # Apply gaussian
                    for i in range(64):
                        for j in range(64):
                            target_heatmaps[b, k, i, j] = torch.exp(
                                torch.tensor(-((i - y)**2 + (j - x)**2) / (2 * 2**2))
                            )
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            pred_heatmaps = model(images)
            
            # Calculate loss
            loss = criterion(pred_heatmaps, target_heatmaps)
            
            # Backward pass
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            train_loss += loss.item()


        # Calculate average training loss
        train_loss /= len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_gts = []
        
        with torch.no_grad():
            for images, keypoints in val_loader:
                # Move to device
                images = images.to(device)
                keypoints = keypoints.to(device)
                
                # Forward pass
                pred_heatmaps = model(images)
                
                # Convert heatmaps to coordinates
                pred_coords = heatmaps_to_coordinates(pred_heatmaps)
                
                # Store for PCK calculation
                all_preds.append(pred_coords)
                all_gts.append(keypoints)
                
                # Generate target heatmaps (same as training)
                target_heatmaps = torch.zeros((images.size(0), 15, 64, 64)).to(device)
                for b in range(images.size(0)):
                    for k in range(keypoints.size(1)):
                        x, y = keypoints[b, k, 0].item(), keypoints[b, k, 1].item()
                        if x == 0 and y == 0:
                            continue
                        
                        x = int(x * 64)
                        y = int(y * 64)
                        
                        for i in range(64):
                            for j in range(64):
                                target_heatmaps[b, k, i, j] = torch.exp(
                                    -((i - y)**2 + (j - x)**2) / (2 * 2**2)
                                )
                
                # Calculate loss
                loss = criterion(pred_heatmaps, target_heatmaps)
                val_loss += loss.item()
        
        # Calculate average validation loss
        val_loss /= len(val_loader)
        
        # Calculate PCK
        all_preds = torch.cat(all_preds, dim=0)
        all_gts = torch.cat(all_gts, dim=0)
        val_pck = calculate_pck(all_preds, all_gts, threshold=0.2)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val PCK: {val_pck:.4f}")
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Save best model
        if val_pck > best_val_pck:
            best_val_pck = val_pck
            torch.save(model.state_dict(), 'best_pose_model.pth')
            print(f"Saved new best model with PCK: {best_val_pck:.4f}")


def visualize_predictions(model, image_path, device):
    # Load and preprocess image
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('L')
    original_image = np.array(image)
    
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get predictions
    model.eval()
    with torch.no_grad():
        heatmaps = model(input_tensor)
        keypoints = heatmaps_to_coordinates(heatmaps).squeeze().cpu().numpy()
    
    # Visualization
    plt.figure(figsize=(10, 10))
    plt.imshow(original_image, cmap='gray')
    
    # Define skeleton connections
    connections = [
        (1, 2), (1, 3), (2, 4), (3, 5), (4, 6),
        (1, 7), (2, 8), (7, 8), (7, 9), (8, 10),
        (9, 11), (10, 12), (11, 13), (12, 14)
    ]
    
    # Plot keypoints
    h, w = original_image.shape[:2]
    for i, (x, y) in enumerate(keypoints):
        plt.scatter(x * w, y * h, c='r', s=50)
        plt.text(x * w, y * h, str(i), fontsize=12)
    
    # Plot connections
    for connection in connections:
        plt.plot(
            [keypoints[connection[0], 0] * w, keypoints[connection[1], 0] * w],
            [keypoints[connection[0], 1] * h, keypoints[connection[1], 1] * h],
            'g-', linewidth=2
        )
    
    plt.title("Pose Estimation Result")
    plt.axis('off')
    plt.show()
    
    # Show individual heatmaps
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()
    
    for i in range(15):
        hmap = heatmaps[0, i].cpu().numpy()
        axes[i].imshow(hmap, cmap='hot')
        axes[i].set_title(f"Keypoint {i}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # Set device
    device = "cpu"
    if torch.accelerator.is_available():
        device = torch.accelerator.current_accelerator()
        print('[INFO] Model moved to accelerator:', torch.accelerator.current_accelerator())
        
    # Define transforms
    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    # Create dataset
    dataset = PoseDataset('images', 'annotations', transform=transform)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Create model
    model = HeatmapPoseModel(num_keypoints=15).to(device)
    
    # Train model
    train_model(model, train_loader, val_loader, device, num_epochs=30)
    
    # Load best model and evaluate
    model.load_state_dict(torch.load('best_pose_model.pth'))
    
    # Visualize predictions on a sample image
    sample_image = 'images/sample.png'
    visualize_predictions(model, sample_image, device)

if __name__ == "__main__":
    main()