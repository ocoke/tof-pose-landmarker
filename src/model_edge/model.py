import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- small helpers ---------
class ConvBNAct(nn.Module):
    """
    2D Conv -> BatchNorm -> ReLU6
    (quant/TF-lite friendly; BN will fold into Conv at export)
    """
    def __init__(self, in_ch, out_ch, k=1, s=1, p=0, g=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p,
                              groups=g, bias=False)
        self.bn   = nn.BatchNorm2d(out_ch)
        self.act  = nn.ReLU6(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DSConv(nn.Module):
    """
    Depthwise (3x3) + Pointwise (1x1), each with BN+ReLU6.
    """
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dw = ConvBNAct(in_ch, in_ch, k=3, s=1, p=1, g=in_ch)  # depthwise
        self.pw = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, g=1)     # pointwise
    def forward(self, x):
        return self.pw(self.dw(x))


class EdgeBlock(nn.Module):
    """
    Two DSConv layers with an optional residual when channels match.
    Keeps compute low while improving representation capacity.
    """
    def __init__(self, in_ch, out_ch, p_drop=0.05):
        super().__init__()
        self.ds1 = DSConv(in_ch, out_ch)
        self.ds2 = DSConv(out_ch, out_ch)
        self.use_res = (in_ch == out_ch)
        self.drop = nn.Dropout2d(p_drop) if p_drop > 0 else nn.Identity()

    def forward(self, x):
        out = self.ds2(self.ds1(x))
        if self.use_res:
            out = out + x
        return self.drop(out)


class Down(nn.Module):
    """
    2x downsample via MaxPool -> EdgeBlock
    """
    def __init__(self, in_ch, out_ch, p_drop=0.05):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.block = EdgeBlock(in_ch, out_ch, p_drop=p_drop)
    def forward(self, x):
        return self.block(self.pool(x))


class Up(nn.Module):
    """
    2x upsample via bilinear interpolate (cheap & artifact-free),
    then 1x1 reduce, concat with skip, then EdgeBlock.
    """
    def __init__(self, in_ch, skip_ch, out_ch, p_drop=0.05):
        """
        in_ch:  channels of the upsampled path BEFORE reduction
        skip_ch: channels coming from the encoder skip
        out_ch: output channels after fusion
        """
        super().__init__()
        self.reduce = ConvBNAct(in_ch, in_ch // 2, k=1, s=1, p=0)
        self.block  = EdgeBlock(in_ch // 2 + skip_ch, out_ch, p_drop=p_drop)

    def forward(self, x_up, x_skip):
        x_up = F.interpolate(x_up, scale_factor=2, mode='bilinear', align_corners=False)
        x_up = self.reduce(x_up)  # (B, in_ch//2, H*2, W*2)

        # pad if shapes drift (shouldn't, but safe)
        diffY = x_skip.size(2) - x_up.size(2)
        diffX = x_skip.size(3) - x_up.size(3)
        if diffY != 0 or diffX != 0:
            x_up = F.pad(x_up, [diffX//2, diffX - diffX//2,
                                diffY//2, diffY - diffY//2])

        x = torch.cat([x_skip, x_up], dim=1)
        return self.block(x)


# --------- the Edge UNet ---------
class EdgePoseUNet(nn.Module):
    """
    Lightweight UNet for heatmap regression on 240x240,
    2-channel input (depth, confidence), COCO-17 keypoints output.
    Quant/FP16/tflite-friendly: ReLU6, BN (folds), no transposed convs.
    """
    def __init__(self, in_ch: int, n_kpts: int, width_mult: float = 1.0, p_drop: float = 0.05):
        super().__init__()

        # MobileNet-ish channel plan (scales well with width_mult)
        def C(c):  # round to multiple of 8 for better kernels
            c = int(c * width_mult)
            return max(8, int((c + 7) // 8) * 8)

        c1, c2, c3, c4, c5 = C(32), C(64), C(96), C(160), C(256)

        # Encoder
        self.stem  = EdgeBlock(in_ch, c1, p_drop=p_drop)  # 240x240 -> 240x240
        self.down1 = Down(c1, c2, p_drop=p_drop)          # 120x120
        self.down2 = Down(c2, c3, p_drop=p_drop)          # 60x60
        self.down3 = Down(c3, c4, p_drop=p_drop)          # 30x30
        self.down4 = Down(c4, c5, p_drop=p_drop)          # 15x15

        # Decoder (bilinear upsample + 1x1 reduce + concat + EdgeBlock)
        self.up1 = Up(c5, c4, c4, p_drop=p_drop)          # 15->30, out c4
        self.up2 = Up(c4, c3, c3, p_drop=p_drop)          # 30->60, out c3
        self.up3 = Up(c3, c2, c2, p_drop=p_drop)          # 60->120, out c2
        self.up4 = Up(c2, c1, c1, p_drop=p_drop)          # 120->240, out c1

        # Head: 1x1 conv to K heatmaps (logits; use BCEWithLogitsLoss)
        self.outc = nn.Conv2d(c1, n_kpts, kernel_size=1, bias=True)

        # Slightly better init for stability
        nn.init.zeros_(self.outc.bias)

    def forward(self, x):
        x1 = self.stem(x)        # (B, c1, 240, 240)
        x2 = self.down1(x1)      # (B, c2, 120, 120)
        x3 = self.down2(x2)      # (B, c3,  60,  60)
        x4 = self.down3(x3)      # (B, c4,  30,  30)
        x5 = self.down4(x4)      # (B, c5,  15,  15)

        y  = self.up1(x5, x4)    # (B, c4,  30,  30)
        y  = self.up2(y,  x3)    # (B, c3,  60,  60)
        y  = self.up3(y,  x2)    # (B, c2, 120, 120)
        y  = self.up4(y,  x1)    # (B, c1, 240, 240)

        return self.outc(y)      # (B, K, 240, 240)

