import os
import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class PoseDataset(Dataset):
    def __init__(self, image_dir, annotation_dir, transform=None, target_transform=None):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        self.target_transform = target_transform
        
        # Get all image filenames
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # load image
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        image = Image.open(image_path).convert('L')

        annotation_path = os.path.join(
            self.annotation_dir,
            img_name.replace('.png', '.json')
        )

        with open(annotation_path, 'r') as f:
            annotation = json.load(f)

        keypoints = []
        if 'pose_landmarks' in annotation and len(annotation['pose_landmarks']) > 0:
            landmarks = annotation['pose_landmarks'][0]
            for point in landmarks:
                # print(point)
                keypoints.append([point['x'], point['y']])

            keypoints = torch.tensor(keypoints, dtype=torch.float32)
        else:
            # empty keypoints
            keypoints = torch.zeros((15, 2), dtype=torch.float32)

        if self.transform:
            image = self.transform(image)

        return image, keypoints