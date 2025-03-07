import torch
import torch.nn as nn
import torch.nn.functional as F


class GenerateHeatmaps:
    def __init__(self, output_size=(64, 64), sigma=2):
        self.output_size = output_size
        self.sigma = sigma
    
    def __call__(self, keypoints):
        # Create empty heatmaps
        num_keypoints = keypoints.shape[0]
        heatmaps = torch.zeros(num_keypoints, *self.output_size)
        
        height, width = self.output_size
        
        for i in range(num_keypoints):
            # Skip if keypoint is not visible
            if keypoints[i, 0] == 0 and keypoints[i, 1] == 0:
                continue
                
            # Convert to pixel coordinates
            x, y = int(keypoints[i, 0] * width), int(keypoints[i, 1] * height)
            
            # Create gaussian
            for h in range(height):
                for w in range(width):
                    heatmaps[i, h, w] = torch.exp(
                        -((h - y)**2 + (w - x)**2) / (2 * self.sigma**2)
                    )
        
        return heatmaps

class HeatmapPoseModel(nn.Module):
    def __init__(self, num_keypoints=15):
        super(HeatmapPoseModel, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Residual block 1
            self._make_residual_block(64, 64, 2),
            
            # Residual block 2
            self._make_residual_block(64, 128, 2, stride=2),
            
            # Residual block 3
            self._make_residual_block(128, 256, 2, stride=2),
        )

        # Decoder (heatmap generation)
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(64, num_keypoints, kernel_size=1)
        )

    def _make_residual_block(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        # First layer may have stride
        layers.append(self._make_residual_unit(in_channels, out_channels, stride))
        
        # Remaining layers
        for _ in range(1, blocks):
            layers.append(self._make_residual_unit(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _make_residual_unit(self, in_channels, out_channels, stride=1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        return ResidualUnit(in_channels, out_channels, stride, downsample)
    
    def forward(self, x):
        features = self.encoder(x)
        heatmaps = self.decoder(features)
        return heatmaps
    
class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out