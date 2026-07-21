"""Canonical lightweight U-Net baseline used by Experiment B."""

from __future__ import annotations

from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return int(ceil(value / divisor) * divisor)


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class DepthwiseSeparable(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.use_residual = stride == 1 and in_channels == out_channels

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.pointwise(self.depthwise(inputs))
        return inputs + outputs if self.use_residual else outputs


class DownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        middle = max(in_channels, out_channels)
        self.block = nn.Sequential(
            DepthwiseSeparable(in_channels, middle, stride=2),
            DepthwiseSeparable(middle, out_channels),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.reduce = ConvBNReLU(in_channels, out_channels, kernel=1, padding=0)
        self.refine = DepthwiseSeparable(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        outputs = F.interpolate(inputs, scale_factor=2, mode="bilinear", align_corners=False)
        outputs = self.reduce(outputs)
        height_delta = skip.shape[2] - outputs.shape[2]
        width_delta = skip.shape[3] - outputs.shape[3]
        if height_delta or width_delta:
            outputs = F.pad(outputs, (0, width_delta, 0, height_delta))
        return self.refine(torch.cat([skip, outputs], dim=1))


class MacchiatoPoseUNet(nn.Module):
    """v7 U-Net graph, renamed for the scene-split v9 baseline."""

    def __init__(self, input_channels: int = 3, num_keypoints: int = 17, width_multiplier: float = 1.15):
        super().__init__()
        c1 = make_divisible(24 * width_multiplier)
        c2 = make_divisible(48 * width_multiplier)
        c3 = make_divisible(96 * width_multiplier)
        c4 = make_divisible(192 * width_multiplier)

        self.stem = ConvBNReLU(input_channels, c1, stride=2)
        self.encoder1 = DepthwiseSeparable(c1, c1)
        self.down1 = DownBlock(c1, c2)
        self.encoder2 = DepthwiseSeparable(c2, c2)
        self.down2 = DownBlock(c2, c3)
        self.encoder3 = DepthwiseSeparable(c3, c3)
        self.down3 = DownBlock(c3, c4)
        self.encoder4 = DepthwiseSeparable(c4, c4)

        self.up1 = UpBlock(c4, c3, c3)
        self.up2 = UpBlock(c3, c2, c2)
        self.up3 = UpBlock(c2, c1, c1)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c1, c1, kernel=1, padding=0),
        )
        self.heatmap_head = nn.Conv2d(c1, num_keypoints, kernel_size=1)
        self.visibility_head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(c1, num_keypoints))
        nn.init.zeros_(self.heatmap_head.bias)
        nn.init.zeros_(self.visibility_head[-1].bias)

    def forward(self, inputs: torch.Tensor, return_aux: bool = False):
        x1 = self.encoder1(self.stem(inputs))
        x2 = self.encoder2(self.down1(x1))
        x3 = self.encoder3(self.down2(x2))
        x4 = self.encoder4(self.down3(x3))
        outputs = self.up4(self.up3(self.up2(self.up1(x4, x3), x2), x1))
        heatmaps = self.heatmap_head(outputs)
        if not return_aux:
            return heatmaps
        return heatmaps, self.visibility_head(outputs)


# Compatibility alias for legacy checkpoint/evaluator terminology.
EdgePoseUNetV7 = MacchiatoPoseUNet
