import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision import transforms
import numpy as np
class SepConv3x3(nn.Module):
    """Depthwise-separable 3×3 convolution + BN + ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn2(x)
        return self.act(x)

class SlimDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch, mid_ch=None, p_drop=0.1):
        super().__init__()
        mid_ch = mid_ch or out_ch
        self.block1 = SepConv3x3(in_ch, mid_ch)
        self.block2 = SepConv3x3(mid_ch, out_ch)
        self.dropout = nn.Dropout2d(p_drop)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return self.dropout(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = SlimDoubleConv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(self.pool(x))

class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # halve the channels after transpose
        self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2)
        # concatenation doubles channels → use SlimDoubleConv
        self.conv = SlimDoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad if needed
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = nn.functional.pad(x1, [diffX//2, diffX - diffX//2,
                                    diffY//2, diffY - diffY//2])
        return self.conv(torch.cat([x2, x1], dim=1))

class SlimPoseUNet(nn.Module):
    def __init__(self, in_ch, n_kpts):
        super().__init__()
        # encoder
        self.inc   = SlimDoubleConv(in_ch, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)
        # decoder
        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.up4 = Up(64, 32)
        self.outc = nn.Conv2d(32, n_kpts, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x, x3)
        x  = self.up3(x, x2)
        x  = self.up4(x, x1)
        return self.outc(x)
