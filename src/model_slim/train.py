import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
import os, json
import matplotlib.pyplot as plt

from model import SlimPoseUNet
from dataset import PoseDataset

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 32
EPOCHS = 150
# IMPORTANT: Update this path to point to your dataset in Google Drive
DATA_DIR = "data/"
NUM_KEYPOINTS = 17
INPUT_CHANNELS = 2
IMAGE_RESOLUTION = (240, 240)
VAL_SPLIT = 0.2
RANDOM_SEED = 720

# --- New Hyperparameters for Advanced Training ---
SCHEDULER_PATIENCE = 3
SCHEDULER_FACTOR = 0.1
EARLY_STOP_PATIENCE = 7

# --- Device Configuration ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f'Using device: {device}')


# --- Model, Loss, Optimizer ---
model = SlimPoseUNet(INPUT_CHANNELS, NUM_KEYPOINTS).to(device)
# loss_function = nn.MSELoss()
pos_weight = torch.tensor([500.0]).to(device)
loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight) # NEW CODE
print(f"Using loss function: {loss_function.__class__.__name__} with pos_weight={pos_weight.item()}")

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR,
)

# --- Datasets and DataLoaders ---
full_dataset = PoseDataset(data_dir=DATA_DIR, output_res=IMAGE_RESOLUTION, augment=True)
non_aug_dataset = PoseDataset(data_dir=DATA_DIR, output_res=IMAGE_RESOLUTION, augment=False)

dataset_size = len(full_dataset)
val_size = int(dataset_size * VAL_SPLIT)
train_size = dataset_size - val_size
print(f"Total samples: {dataset_size}, Training on: {train_size}, Validating on: {val_size}")

generator = torch.Generator().manual_seed(RANDOM_SEED)
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(non_aug_dataset, val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("DataLoaders created. Training set will be augmented, validation set will not.")

# --- Training State Variables ---
best_val_loss = float('inf')
epochs_no_improve = 0
# IMPORTANT: Update this path to your Google Drive
MODEL_SAVE_PATH = "/content/drive/MyDrive/pose_unet_bce_best_model_slim_1.pth"
# FINAL_MODEL_PATH = "/content/drive/MyDrive/pose_unet_bce_model.pth"

# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_train_loss = 0
    for data, targets in train_loader:
        data, targets = data.to(device), targets.to(device)
        optimizer.zero_grad()
        predictions = model(data)
        loss = loss_function(predictions, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation Loop ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            predictions = model(data)
            val_loss = loss_function(predictions, targets)
            total_val_loss += val_loss.item()
    avg_val_loss = total_val_loss / len(val_loader)

    # print(f"Epoch {epoch+1}/{EPOCHS} -> Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"LR: {current_lr:.1e}"
    )

    # --- Scheduler and Early Stopping ---
    scheduler.step(avg_val_loss)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Validation loss improved. Model saved to {MODEL_SAVE_PATH}")
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epochs.")

    if epochs_no_improve >= EARLY_STOP_PATIENCE:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

# --- Save Final Model ---
# torch.save(model.state_dict(), FINAL_MODEL_PATH)
print(f"Finished training.")
