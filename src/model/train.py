import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from model import PoseUNet
from dataset import PoseDataset
import torch.nn as nn
import os

# --- Hyperparameters ---
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EPOCHS = 100
DATA_DIR = "./data" # The root directory for collected data
NUM_KEYPOINTS = 33 # MediaPipe's 33 landmarks
INPUT_CHANNELS = 2
IMAGE_RESOLUTION = (240, 240) # Padded square resolution
VAL_SPLIT = 0.2
RANDOM_SEED = 2506

SCHEDULER_PATIENCE = 3 # How many epochs to wait before reducing LR
SCHEDULER_FACTOR = 0.1
EARLY_STOP_PATIENCE = 7 # How many epochs to wait for improvement before stopping

device = "cpu"
if torch.accelerator.is_available():
    # new API, also works with Apple Silicon
    device = torch.accelerator.current_accelerator()
    print('[INFO] Model moved to accelerator:', torch.accelerator.current_accelerator())

# Model
model = PoseUNet(n_channels=INPUT_CHANNELS, n_keypoints=NUM_KEYPOINTS).to(device)

# Loss and Optimizer
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# the learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, 'min', patience=SCHEDULER_PATIENCE, factor=SCHEDULER_FACTOR,
)


# DataLoaders
# train_dataset = PoseDataset(data_dir=os.path.join(DATA_DIR, 'train'), output_res=IMAGE_RESOLUTION)
# train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# val_dataset = PoseDataset(data_dir=os.path.join(DATA_DIR, 'val'), output_res=IMAGE_RESOLUTION)
# val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

full_dataset = PoseDataset(data_dir=DATA_DIR, output_res=IMAGE_RESOLUTION, augment=True)
non_aug_dataset = PoseDataset(data_dir=DATA_DIR, output_res=IMAGE_RESOLUTION, augment=False)

print(f"Total number of samples: {len(full_dataset)}")
dataset_size = len(full_dataset)
val_size = int(dataset_size * VAL_SPLIT)
train_size = dataset_size - val_size
print(f"Training size: {train_size}, Validation size: {val_size}")

generator = torch.Generator().manual_seed(RANDOM_SEED)

train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size], generator=generator)
# train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
val_dataset = torch.utils.data.Subset(non_aug_dataset, val_indices)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)

best_val_loss = float('inf')  # Initialize best validation loss

print("DataLoaders created. Training set will be augmented.")

epochs_no_improve = 0


# --- Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)

        # Forward pass
        predictions = model(data)
        
        # Calculate loss
        loss = loss_function(predictions, targets)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Training Loss: {avg_loss:.6f}")

    # --- Validation (Optional but Recommended) ---
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            predictions = model(data)
            val_loss = loss_function(predictions, targets)
            total_val_loss += val_loss.item()
    
    avg_val_loss = total_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{EPOCHS}, Validation Loss: {avg_val_loss:.6f}")

    current_lr = optimizer.param_groups[0]['lr']

    print(
        f"Epoch {epoch+1}/{EPOCHS} | "
        f"Train Loss: {avg_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"LR: {current_lr:.1e}"
    )

    scheduler.step(avg_val_loss)


    if avg_loss < best_val_loss:
        best_val_loss = avg_loss
        print(f"New best validation loss: {best_val_loss:.6f}, saving model...")
        # Save the model state
        torch.save(model.state_dict(), "models/pose_unet_best_model.pth")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epochs.")
        
        # Early stopping
        if epochs_no_improve >= EARLY_STOP_PATIENCE:
            print(f"Early stopping triggered after {EARLY_STOP_PATIENCE} epochs without improvement.")
            break

# --- Save the trained model ---
torch.save(model.state_dict(), "models/pose_unet_model.pth")
print("Model saved!")