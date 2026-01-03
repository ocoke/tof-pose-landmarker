# ---------- EdgePoseUNetV2 (ARM/TFLite friendly) ----------
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import ceil

def make_divisible(v, divisor=8):
    return int(ceil(v / divisor) * divisor)

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU(inplace=True) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DWSeparable(nn.Module):
    """Depthwise 3x3 (+BN+ReLU) then pointwise 1x1 (+BN+ReLU)."""
    def __init__(self, in_ch, out_ch, s=1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, s, 1, groups=in_ch, bias=False)
        self.bn1= nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2= nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.use_res = (s==1 and in_ch==out_ch)
    def forward(self, x):
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        return x + y if self.use_res else y

class DownDS(nn.Module):
    """Stride-2 DSConv (faster than MaxPool+conv on ARM)."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        mid = max(in_ch, out_ch)
        self.block = nn.Sequential(
            DWSeparable(in_ch, mid, s=2),
            DWSeparable(mid, out_ch, s=1),
        )
    def forward(self, x):
        return self.block(x)

class UpLite(nn.Module):
    """Bilinear upsample → 1x1 reduce → concat skip → DSConv refine."""
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.reduce = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.refine = DWSeparable(out_ch + skip_ch, out_ch, s=1)
    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        # pad if sizes mismatch by 1
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.refine(x)

class EdgePoseUNetV2(nn.Module):
    """
    width_mult scales channels; use 1.0–1.25 on Pi5.
    Blocks: DSConv encoder (stride-2) + bilinear FPN-like decoder.
    """
    def __init__(self, in_ch=2, n_kpts=17, width_mult=1.15):
        super().__init__()
        c1 = make_divisible(24 * width_mult)
        c2 = make_divisible(48 * width_mult)
        c3 = make_divisible(96 * width_mult)
        c4 = make_divisible(192 * width_mult)

        # Stem (s=2 to cut 4x compute early)
        self.stem = ConvBNReLU(in_ch, c1, k=3, s=2, p=1)   # 240 -> 120

        # Encoder (120 -> 60 -> 30 -> 15)
        self.e1 = DWSeparable(c1, c1, s=1)                 # 120
        self.d1 = DownDS(c1, c2)                           # 60
        self.e2 = DWSeparable(c2, c2, s=1)                 # 60
        self.d2 = DownDS(c2, c3)                           # 30
        self.e3 = DWSeparable(c3, c3, s=1)                 # 30
        self.d3 = DownDS(c3, c4)                           # 15
        self.e4 = DWSeparable(c4, c4, s=1)                 # 15 (bottleneck)

        # Decoder (15 -> 30 -> 60 -> 120 -> 240)
        self.up1 = UpLite(c4, c3, c3)                      # 30
        self.up2 = UpLite(c3, c2, c2)                      # 60
        self.up3 = UpLite(c2, c1, c1)                      # 120
        self.up4x2 = nn.Sequential(                        # 120 -> 240
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c1, c1, k=1, s=1, p=0)
        )

        self.head = nn.Conv2d(c1, n_kpts, kernel_size=1, bias=True)



    def forward(self, x):
        # Encoder
        x1 = self.stem(x)          # (B, c1, 120,120)
        x1 = self.e1(x1)
        x2 = self.d1(x1)           # (B, c2, 60,60)
        x2 = self.e2(x2)
        x3 = self.d2(x2)           # (B, c3, 30,30)
        x3 = self.e3(x3)
        x4 = self.d3(x3)           # (B, c4, 15,15)
        x4 = self.e4(x4)

        # Decoder
        y3 = self.up1(x4, x3)      # (B, c3, 30,30)
        y2 = self.up2(y3, x2)      # (B, c2, 60,60)
        y1 = self.up3(y2, x1)      # (B, c1, 120,120)
        y0 = self.up4x2(y1)        # (B, c1, 240,240)

        return self.head(y0)       # logits (no sigmoid)