# Grayscale Pose Landmarker

A specialized CNN-based pose landmark detection model trained specifically for grayscale images captured by ToF (Time of Flight) cameras.

## Overview

This project addresses the domain shift problem in traditional pose landmark models (like MediaPipe) when working with grayscale ToF camera images. By directly training on ToF camera data with its inherent noise characteristics, this model provides more accurate pose estimation for grayscale depth-sensing applications.

## Features

- Optimized for grayscale images from ToF cameras
- Reduced keypoint set designed for practical ToF camera applications
- Noise-tolerant architecture trained directly on ToF camera data
- MediaPipe-compatible keypoint format (with modifications)
- PyTorch-based implementation

## Keypoint Structure

This model uses a modified keypoint structure with 15 landmarks, adapted from the original MediaPipe pose landmarker:

| Index | Keypoint       | Original Index |
|-------|----------------|---------------|
| 0     | Head           | 0             |
| 1     | Left Shoulder  | 11            |
| 2     | Right Shoulder | 12            |
| 3     | Left Elbow     | 13            |
| 4     | Right Elbow    | 14            |
| 5     | Left Hand      | 19            |
| 6     | Right Hand     | 20            |
| 7     | Left Hip       | 23            |
| 8     | Right Hip      | 24            |
| 9     | Left Knee      | 25            |
| 10    | Right Knee     | 26            |
| 11    | Left Ankle     | 27            |
| 12    | Right Ankle    | 28            |
| 13    | Left Foot      | 31            |
| 14    | Right Foot     | 32            |

## Model Architecture

The model employs a CNN-based architecture:
- Input: Grayscale images (1 channel)
- Multiple convolutional layers with ReLU activations and max pooling
- Fully connected layers for keypoint regression
- Output: 15 keypoints with (x,y) coordinates

## Dataset

- ~1,000 training samples (as of March 2025)
- Directly captured from ToF cameras with inherent noise
- Annotations in a MediaPipe-like format

### Current Dataset Limitations

The current dataset (as of March 3, 2025) has some coverage limitations:
- Insufficient data for feet poses
- Limited upper angle views
- Not covering all possible human poses

## Requirements

- Suggested Python version: `3.11.9+`

```bash
pip3 install -r requirements.txt
```

## Usage

### Loading the Pre-trained Model

```python
import torch
from model import GrayscaleCNN  # Import your model definition

# Initialize model
model = GrayscaleCNN(num_classes=15)

# Load pre-trained weights
model.load_state_dict(torch.load('model.pth'))
model.eval()

# Process a grayscale image
# [Your code to load and preprocess the image]
with torch.no_grad():
    keypoints = model(image_tensor)

# keypoints shape: [1, 15, 2] where 15 is the number of keypoints and 2 is (x,y) coordinates
```

## Training

The model is trained using MSE loss on landmark positions. To train on your own dataset:

1. Organize your data:
   - Place grayscale images in an `images` directory
   - Place corresponding JSON annotations in an `annotations` directory
   - Ensure each annotation JSON file has the same base filename as its image

2. Run the training script:

```bash
python train.py
```

See the full training code in the repository for detailed implementation.

## Future Improvements

- Expand dataset with more varied poses, especially for feet and upper angles
- Implement data augmentation to improve generalization
- Explore attention mechanisms for better keypoint localization
- Add temporal consistency for video streams
