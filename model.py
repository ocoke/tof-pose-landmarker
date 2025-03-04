import torch
import torch.nn as nn
import torch.nn.functional as F

class GrayscaleCNN(nn.Module):
    def __init__(self, num_classes):
        super(GrayscaleCNN, self).__init__()

        self.num_keypoints = num_classes  # e.g. 15 keypoints
        self.out_features = num_classes * 2  # each keypoint has x and y coordinates

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1), # output: 32x240x180
            nn.ReLU(),
            nn.MaxPool2d(2), # output: 32x120x90,

            nn.Conv2d(32, 64, kernel_size=3, padding=1), # output: 64x120x90
            nn.ReLU(),
            nn.MaxPool2d(2), # output: 64x60x45,

            nn.Conv2d(64, 128, kernel_size=3, padding=1), # output: 128x60x45
            nn.ReLU(),
            nn.MaxPool2d(2), # output: 128x30x22
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*30*22, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, self.out_features)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(-1, self.num_keypoints, 2)
        return x