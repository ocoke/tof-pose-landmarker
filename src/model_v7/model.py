from math import ceil

import torch
import torch.nn as nn
import torch.nn.functional as F


def make_divisible(value: float, divisor: int = 8) -> int:
    return int(ceil(value / divisor) * divisor)


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, act: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DWSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, s: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, 3, s, 1, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.use_res = s == 1 and in_ch == out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.act(self.bn1(self.dw(x)))
        y = self.act(self.bn2(self.pw(y)))
        return x + y if self.use_res else y


class DownDS(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        mid = max(in_ch, out_ch)
        self.block = nn.Sequential(
            DWSeparable(in_ch, mid, s=2),
            DWSeparable(mid, out_ch, s=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UpLite(nn.Module):
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.reduce = ConvBNReLU(in_ch, out_ch, k=1, s=1, p=0)
        self.refine = DWSeparable(out_ch + skip_ch, out_ch, s=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.reduce(x)
        dh = skip.shape[2] - x.shape[2]
        dw = skip.shape[3] - x.shape[3]
        if dh or dw:
            x = F.pad(x, (0, dw, 0, dh))
        x = torch.cat([skip, x], dim=1)
        return self.refine(x)


class EdgePoseUNetV7(nn.Module):
    """Pi 5 safe v7 model: same deployed graph shape as the strong v6 baseline."""

    def __init__(self, in_ch: int = 3, n_kpts: int = 17, width_mult: float = 1.15):
        super().__init__()
        c1 = make_divisible(24 * width_mult)
        c2 = make_divisible(48 * width_mult)
        c3 = make_divisible(96 * width_mult)
        c4 = make_divisible(192 * width_mult)

        self.stem = ConvBNReLU(in_ch, c1, k=3, s=2, p=1)
        self.e1 = DWSeparable(c1, c1, s=1)
        self.d1 = DownDS(c1, c2)
        self.e2 = DWSeparable(c2, c2, s=1)
        self.d2 = DownDS(c2, c3)
        self.e3 = DWSeparable(c3, c3, s=1)
        self.d3 = DownDS(c3, c4)
        self.e4 = DWSeparable(c4, c4, s=1)

        self.up1 = UpLite(c4, c3, c3)
        self.up2 = UpLite(c3, c2, c2)
        self.up3 = UpLite(c2, c1, c1)
        self.up4x2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNReLU(c1, c1, k=1, s=1, p=0),
        )

        self.head = nn.Conv2d(c1, n_kpts, kernel_size=1, bias=True)
        self.visibility_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c1, n_kpts),
        )
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.visibility_head[-1].bias)

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        x1 = self.stem(x)
        x1 = self.e1(x1)
        x2 = self.d1(x1)
        x2 = self.e2(x2)
        x3 = self.d2(x2)
        x3 = self.e3(x3)
        x4 = self.d3(x3)
        x4 = self.e4(x4)

        y3 = self.up1(x4, x3)
        y2 = self.up2(y3, x2)
        y1 = self.up3(y2, x1)
        y0 = self.up4x2(y1)

        heat_logits = self.head(y0)
        if not return_aux:
            return heat_logits
        vis_logits = self.visibility_head(y0)
        return heat_logits, vis_logits
