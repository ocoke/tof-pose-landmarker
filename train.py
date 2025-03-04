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

# CNN Model
from model import GrayscaleCNN

BATCH_SIZE = 32

# Define the dataset
class PoseLandmarkDataset(Dataset):
    def __init__(self, images_dir, annotations_dir, transform=None):
        """
        Args:
            images_dir (string): Directory with all the images.
            annotations_dir (string): Directory with all the JSON annotation files.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.transform = transform
        # unique image filenames matching the annotation files
        file_list = sorted(os.listdir(images_dir))
        # only takes PNG files
        self.image_filenames = list(filter(lambda f: f.endswith('.png'), file_list))
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        # Get the image filename and its corresponding annotation filename
        img_filename = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_filename)
        image = Image.open(img_path).convert('L') # ensure it's gray scale
        annotation_path = os.path.join(self.annotations_dir, os.path.splitext(img_filename)[0] + '.json')

        with open(annotation_path) as f:
            annotation = json.load(f)
        
        landmarks_list = annotation['pose_landmarks'][0]
        landmarks = np.array([[lm["x"], lm["y"]] for lm in landmarks_list], dtype=np.float32)

        sample = {'image': image, 'landmarks': landmarks}

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    




class ToTensor(object):
    """Convert a sample with image and landmarks to Tensors."""
    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # Convert image to tensor: resulting shape [C, H, W]. For grayscale, C=1.
        image = transforms.ToTensor()(image)
        # Optionally convert landmarks to tensor
        landmarks = torch.from_numpy(landmarks)
        return {'image': image, 'landmarks': landmarks}

class ApplyToImage(object):
    """Apply a transformation to the 'image' key of a sample dict."""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['landmarks']
        # Apply the transformation to the image only
        image = self.transform(image)
        return {'image': image, 'landmarks': landmarks}


data_transforms = transforms.Compose([
    # transforms.Grayscale(num_output_channels=1),
    ApplyToImage(transforms.Grayscale(num_output_channels=1)), # make sure the image is grayscale
    ToTensor()
])


# Load the dataset

dataset = PoseLandmarkDataset(images_dir='images', annotations_dir='annotations', transform=data_transforms)

# randomly split the dataset into training and validation sets
# 80% data for training, 20% for validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# the new model only contains 15 keypoints
# see README.md for detailed differences
num_classes = 15  
model = GrayscaleCNN(num_classes)

# We move our tensor to the current accelerator if available

device = "cpu"
if torch.accelerator.is_available():
    model = model.to(torch.accelerator.current_accelerator())
    device = torch.accelerator.current_accelerator()
    print('[INFO] Model moved to accelerator:', torch.accelerator.current_accelerator())

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50

sample = dataset[0]
print(type(sample['image']), sample['image'].shape)      # Should be a tensor, e.g., torch.Tensor with shape [1, 240, 180]
print(type(sample['landmarks']), sample['landmarks'].shape)  # Should be a tensor or numpy array, depending on your conversion

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        inputs = batch['image']
        labels = batch['landmarks']
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {100-epoch_loss:.2f}%")

    # Validation phase
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = batch['image'].to(device)
            labels = batch['landmarks'].to(device)
            outputs = model(inputs)
            # _, predicted = torch.max(outputs, 1)

            loss = criterion(outputs, labels)
            val_loss += loss.item()
            # total += labels.size(0)
            # correct += (predicted == labels).sum().item()
    
    # val_accuracy = correct / total
    # print(f"Validation Accuracy: {val_accuracy:.4f}")
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {100-avg_val_loss:.2f}%")


torch.save(model.state_dict(), 'model.pth')
print("Model saved to model.pth")

# Export the loss graph (optional)
plt.figure(figsize=(10, 6))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss', marker='o')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('loss_graph.png')
# plt.show()