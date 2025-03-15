import os
import json
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from model import PoseResNet, BasicBlock  # Adjust the imports to match your project structure

def load_model(weights_path):
    # Create the model (adjust layers and number of joints/landmarks as needed)
    model = PoseResNet(BasicBlock, [2, 2, 2, 2], num_joints=15)
    model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
    model.eval()
    return model

def preprocess_image(image_path):
    # Load the image using PIL in grayscale (change as needed)
    img = Image.open(image_path).convert('L')
    # Optionally reflect (horizontal flip) the image if desired.
    # img = transforms.functional.hflip(img)
    
    # Resize and normalize the image (adjust size as expected by the model)
    transform = transforms.Compose([
        transforms.Resize((240, 180)),  
        transforms.ToTensor(),
        # Uncomment and adjust normalization if needed.
        # transforms.Normalize(mean=[0.485], std=[0.229])
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    return img_tensor

def get_landmarks_from_heatmaps(heatmaps):
    """
    Given heatmaps output from the model, extract landmark locations.
    Each channel represents one landmark.
    """
    landmarks = []
    # Remove batch dimension and convert to NumPy array
    heatmaps_np = heatmaps.squeeze(0).detach().cpu().numpy()
    num_landmarks, h, w = heatmaps_np.shape

    for i in range(num_landmarks):
        # Get the position of the highest response in each heatmap
        y, x = np.unravel_index(np.argmax(heatmaps_np[i]), (h, w))
        landmarks.append((x, y))
    return landmarks

def get_ground_truth_landmarks(image_path, ann_dir="annotations"):
    """
    Load the JSON annotation for the image and parse the ground truth landmarks.
    The JSON is expected to have a structure like:
    {
      "pose_landmarks": [
          [
             {"x": 120, "y": 160}, {"x": 130, "y": 165}, ... up to at least 15 landmarks
          ]
      ]
    }
    Only the first 15 landmarks of the first set are used.
    """
    filename = os.path.basename(image_path)
    ann_filename = filename.replace('.png', '.json')
    ann_path = os.path.join(ann_dir, ann_filename)
    try:
        with open(ann_path, 'r') as f:
            annotation = json.load(f)
    except Exception as e:
        print("Error loading annotation:", e)
        return []
    
    landmarks_data = annotation.get("pose_landmarks", [])
    gt_landmarks = []
    if landmarks_data and len(landmarks_data[0]) >= 15:
        for i in range(15):
            kp = landmarks_data[0][i]
            gt_landmarks.append((240 - kp['x'] * 240, kp['y'] * 180))
    else:
        gt_landmarks = [(0, 0)] * 15
    return gt_landmarks

def display_results(image_path, outputs, pred_landmarks, gt_landmarks):
    # Load image with OpenCV for visualization
    img = cv2.imread(image_path)
    img_copy_pred = img.copy()
    img_copy_gt = img.copy()

    # Get heatmap dimensions and original image dimensions
    heatmap_h, heatmap_w = outputs.shape[2], outputs.shape[3]
    img_h, img_w = img.shape[:2]

    # Draw predicted landmarks on the image copy (green circles)
    for (x, y) in pred_landmarks:
        # Map heatmap coordinates to original image coordinates
        orig_x = int(x * img_w / heatmap_w)
        orig_y = int(y * img_h / heatmap_h)
        cv2.circle(img_copy_pred, (orig_x, orig_y), 5, (0, 255, 0), -1)

    # Draw ground truth landmarks on the image copy (red circles)
    for (gt_x, gt_y) in gt_landmarks:
        cv2.circle(img_copy_gt, (int(gt_x), int(gt_y)), 5, (0, 0, 255), -1)

    # Set up matplotlib figure with four subplots
    plt.figure(figsize=(20, 5))

    # 1. Original image with predicted landmarks
    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(img_copy_pred, cv2.COLOR_BGR2RGB))
    plt.title("Predicted Landmarks (Green)")
    plt.axis('off')

    # 2. Original image with ground truth landmarks
    plt.subplot(1, 4, 2)
    plt.imshow(cv2.cvtColor(img_copy_gt, cv2.COLOR_BGR2RGB))
    plt.title("Ground Truth Landmarks (Red)")
    plt.axis('off')

    # 3. Sample heatmap from the first channel of the model output
    heatmap = outputs[0, 0].detach().cpu().numpy()
    heatmap = cv2.resize(heatmap, (img_w, img_h))
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-6)
    plt.subplot(1, 4, 3)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Sample Predicted Heatmap")
    plt.axis('off')

    # 4. Sum of all predicted heatmaps (model heatmap)
    pred_heatmap_sum = torch.sum(outputs[0], dim=0).detach().cpu().numpy()
    pred_heatmap_sum = cv2.resize(pred_heatmap_sum, (img_w, img_h))
    pred_heatmap_sum = (pred_heatmap_sum - pred_heatmap_sum.min()) / (pred_heatmap_sum.max() - pred_heatmap_sum.min() + 1e-6)
    plt.subplot(1, 4, 4)
    plt.imshow(pred_heatmap_sum, cmap='jet')
    plt.title("Sum of Predicted Heatmaps")
    plt.axis('off')

    plt.show()

import os
import random

def get_random_file(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return os.path.join(folder_path, random.choice(files)) if files else None


def main():
    # random file from folder images



    # image_path = "images/frame_20250224_143718.png"  # Provide your image path
    image_path = get_random_file("images")
    weights_path = "pose_resnet_1.pth"  # Provide your model's weights path

    # Load ground truth landmarks from the annotation JSON
    gt_landmarks = get_ground_truth_landmarks(image_path, ann_dir="annotations")
    
    # Load model and preprocess image
    model = load_model(weights_path)
    img_tensor = preprocess_image(image_path)

    # Run inference
    with torch.no_grad():
        outputs = model(img_tensor)
    
    # Extract predicted landmark positions from the heatmaps (in heatmap coordinates)
    pred_landmarks = get_landmarks_from_heatmaps(outputs)
    print("Predicted landmarks (heatmap coordinates):", pred_landmarks)
    print("Ground truth landmarks (original image coords):", gt_landmarks)

    # Display results: original image with predicted landmarks, ground truth, and sample heatmap
    display_results(image_path, outputs, pred_landmarks, gt_landmarks)

if __name__ == "__main__":
    for i in range(1, 6):
        main()
        print(f"Finished running inference for image {i}")