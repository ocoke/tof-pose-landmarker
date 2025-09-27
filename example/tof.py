#!/usr/bin/env python3
"""
Realtime EdgePoseUNetV2 (FP32) demo on Arducam ToF stream (PyTorch).
- Grabs depth+confidence from the camera
- Pads to 240x240 and normalizes
- Runs EdgePoseUNetV2 (FP32) in PyTorch
- Extracts keypoints (argmax or soft-argmax)
- Draws a COCO-17 skeleton overlay
- Shows FPS and per-frame inference time

Usage:
  python demo_edge_v5_realtime.py --model /path/to/edge_5.0.1.pth

Optional:
  --device cpu            # force CPU (default: auto)
  --width-mult 1.15       # model width multiplier
  --range-mm 4000         # ToF range for depth normalization
  --no-rotate             # do not rotate ToF image 180°
  --conf-thresh 30        # ToF confidence mask threshold (0..255)
  --heatmap-thresh 0.30   # min peak prob to draw a keypoint
  --softmax               # use soft-argmax (beta=2.0) instead of argmax
  --threads 4             # torch.set_num_threads
  --scale 3               # window scale factor for display
"""

import os
import sys
import time
import argparse
from typing import Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# ---- Arducam ToF SDK ----
try:
    import ArducamDepthCamera as ac
except ImportError:
    print("[ERROR] ArducamDepthCamera not found. Install the Arducam ToF SDK.")
    sys.exit(1)

# ---- Import your model ----
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.model_edge_v5.model import EdgePoseUNetV2
except ImportError:
    print("[ERROR] Could not import EdgePoseUNetV2 from model_edge_v5.model")
    sys.exit(1)

# COCO-17 skeleton connections (indices: 0..16)
COCO17_EDGES = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 6),              # Shoulders
    (5, 11), (6, 12),    # Torso (shoulder->hip)
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (0, 1), (0, 2),      # Nose -> eyes
    (1, 3), (2, 4)       # Eyes -> ears
]

# ---------------------- Utils ---------------------- #
def pad_to_square(depth: np.ndarray, conf: np.ndarray, out_hw=(240, 240)) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    """Pad HxW arrays to out_hw (H,W), return (depth_padded, conf_padded, (l,t,r,b))."""
    H, W = depth.shape
    outH, outW = out_hw
    pad_left = (outW - W) // 2
    pad_right = outW - W - pad_left
    pad_top = (outH - H) // 2
    pad_bottom = outH - H - pad_top
    depth_p = np.pad(depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    conf_p  = np.pad(conf,  ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return depth_p, conf_p, (pad_left, pad_top, pad_right, pad_bottom)

def argmax_2d(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns (x,y,peak_prob) for each keypoint from logits via argmax.
    logits: (1,K,H,W) float32
    """
    with torch.no_grad():
        _, K, H, W = logits.shape
        flat = logits.view(1, K, -1)                  # (1,K,H*W)
        idx  = flat.argmax(dim=-1)                    # (1,K)
        x = (idx % W).float().squeeze(0)              # (K,)
        y = (idx // W).float().squeeze(0)             # (K,)
        # peak prob from sigmoid at argmax location
        sig = torch.sigmoid(logits)
        sig_flat = sig.view(1, K, -1)
        peak = torch.gather(sig_flat, 2, idx.unsqueeze(-1)).squeeze(0).squeeze(-1)  # (K,)
        return x, y, peak

def soft_argmax_2d_logits(logits: torch.Tensor, beta: float = 2.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Soft-argmax using softmax over H*W. Returns (x,y,peak_prob) per keypoint.
    'peak_prob' is the max softmax probability per heatmap (rough confidence proxy).
    """
    with torch.no_grad():
        _, K, H, W = logits.shape
        flat = (logits * beta).view(1, K, -1)         # (1,K,H*W)
        sm   = torch.softmax(flat, dim=-1)            # (1,K,H*W)

        # coordinate grids
        ys = torch.arange(H, device=logits.device, dtype=torch.float32).view(1,1,H,1).expand(1,K,H,W).reshape(1,K,-1)
        xs = torch.arange(W, device=logits.device, dtype=torch.float32).view(1,1,1,W).expand(1,K,H,W).reshape(1,K,-1)

        ex = (sm * xs).sum(dim=-1).squeeze(0)         # (K,)
        ey = (sm * ys).sum(dim=-1).squeeze(0)         # (K,)
        peak_prob, _ = sm.max(dim=-1)                 # (1,K)
        return ex, ey, peak_prob.squeeze(0)

def make_display(depth_p: np.ndarray, conf_p: np.ndarray, range_mm: float, conf_thresh: int) -> np.ndarray:
    """Make a 3-channel display image from depth, masking low-confidence pixels."""
    depth_norm = np.clip(depth_p / range_mm, 0, 1)
    img = (depth_norm * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # mask low-conf
    mask = conf_p >= conf_thresh
    img[~mask] = (0, 0, 0)
    return img

def draw_skeleton(img: np.ndarray, xy: np.ndarray, conf: np.ndarray, heatmap_thresh: float = 0.3) -> np.ndarray:
    """
    Draw COCO-17 keypoints & lines. xy: (K,2) in (x,y), conf: (K,) in [0,1].
    Draw only if conf >= heatmap_thresh (faint otherwise).
    """
    K = xy.shape[0]
    H, W = img.shape[:2]

    # draw edges
    for a,b in COCO17_EDGES:
        if a < K and b < K:
            xa, ya = xy[a]
            xb, yb = xy[b]
            ca, cb = conf[a], conf[b]
            if 0 <= xa < W and 0 <= ya < H and 0 <= xb < W and 0 <= yb < H:
                avg_c = float((ca + cb) / 2.0)
                if avg_c >= heatmap_thresh:
                    color, th = (0, 255, 255), 2  # yellow
                else:
                    color, th = (80, 80, 80), 1   # faint gray
                cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, th)

    # draw keypoints
    for i in range(K):
        x, y = xy[i]
        c = float(conf[i])
        if 0 <= x < W and 0 <= y < H:
            if c >= heatmap_thresh:
                col, rad, border = (0, 255, 0), 4, (255,255,255)  # green
            else:
                col, rad, border = (128,128,128), 2, (200,200,200)
            cv2.circle(img, (int(x), int(y)), rad, col, -1)
            cv2.circle(img, (int(x), int(y)), rad+2, border, 1)
    return img

# ---------------------- Camera wrapper ---------------------- #
class ToFStream:
    def __init__(self, range_mm: float = 4000.0, rotate_180: bool = True):
        self.range_mm = range_mm
        self.rotate = rotate_180
        self.cam = ac.ArducamCamera()

    def open(self) -> bool:
        ret = self.cam.open(ac.Connection.CSI, 0)
        if ret != 0:
            print(f"[ERROR] Failed to open ToF camera (code {ret})")
            return False
        self.cam.setControl(ac.Control.RANGE, int(self.range_mm))
        r = self.cam.getControl(ac.Control.RANGE)
        print(f"[OK] ToF opened. Range set to: {r} mm")
        ret = self.cam.start(ac.FrameType.DEPTH)
        if ret != 0:
            print(f"[ERROR] Failed to start depth stream (code {ret})")
            self.cam.close()
            return False
        print("[OK] Depth stream started")
        return True

    def get(self, timeout_ms: int = 200) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        frame = self.cam.requestFrame(timeout_ms)
        if frame is None or not isinstance(frame, ac.DepthData):
            if frame is not None:
                self.cam.releaseFrame(frame)
            return None, None

        depth = frame.depth_data
        conf  = frame.confidence_data
        self.cam.releaseFrame(frame)
        if depth is None or depth.size == 0 or conf is None:
            return None, None

        if self.rotate:
            depth = cv2.rotate(depth, cv2.ROTATE_180)
            conf  = cv2.rotate(conf,  cv2.ROTATE_180)
        return depth.copy(), conf.copy()

    def close(self):
        try:
            self.cam.stop()
            self.cam.close()
        except:
            pass

# ---------------------- Main Demo ---------------------- #
def main():
    ap = argparse.ArgumentParser("EdgePoseUNetV2 ToF Realtime Demo (PyTorch)")
    ap.add_argument("--model", required=True, type=str, help="Path to .pth weights")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device selection")
    ap.add_argument("--width-mult", type=float, default=1.15, help="Width multiplier for EdgePoseUNetV2")
    ap.add_argument("--range-mm", type=float, default=4000.0, help="ToF depth range used for normalization")
    ap.add_argument("--no-rotate", action="store_true", help="Do not rotate ToF images 180°")
    ap.add_argument("--conf-thresh", type=int, default=30, help="ToF confidence mask threshold (0..255)")
    ap.add_argument("--heatmap-thresh", type=float, default=0.30, help="Min peak prob to draw a keypoint")
    ap.add_argument("--softmax", action="store_true", help="Use soft-argmax (beta=2.0) instead of argmax")
    ap.add_argument("--threads", type=int, default=4, help="torch.set_num_threads")
    ap.add_argument("--scale", type=int, default=3, help="Display scale")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Torch threading & perf
    try:
        torch.set_num_threads(args.threads)
        print(f"[INFO] torch.set_num_threads({args.threads})")
    except Exception as e:
        print(f"[WARN] set_num_threads failed: {e}")

    # Load model
    model = EdgePoseUNetV2(in_ch=2, n_kpts=17, width_mult=args.width_mult).to(device)
    state = torch.load(args.model, map_location=device)
    model.load_state_dict(state)
    model.eval()
    model = model.to(memory_format=torch.channels_last)
    print("[OK] Model loaded")

    # Warm-up (optional small speedup on CPU)
    with torch.inference_mode():
        dummy = torch.zeros(1, 2, 240, 240, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
        _ = model(dummy)

    # Camera
    stream = ToFStream(range_mm=args.range_mm, rotate_180=(not args.no_rotate))
    if not stream.open():
        sys.exit(1)

    cv2.namedWindow("EdgePoseUNetV2 Realtime", cv2.WINDOW_AUTOSIZE)

    # Timing
    last_t = time.time()
    fps_ma = 0.0
    alpha = 0.05  # EMA for FPS

    try:
        while True:
            # Grab frame
            depth, conf = stream.get()
            if depth is None:
                continue

            # Normalize & pad
            depth_norm = np.clip(depth / args.range_mm, 0, 1).astype(np.float32)
            conf_norm  = np.clip(conf  / 255.0,        0, 1).astype(np.float32)
            depth_p, conf_p, pads = pad_to_square(depth_norm, conf_norm, (240, 240))

            # Prepare tensor NCHW float32
            inp = np.stack([depth_p, conf_p], axis=0)  # (2,240,240)
            tensor = torch.from_numpy(inp).unsqueeze(0).to(device)  # (1,2,240,240)
            tensor = tensor.contiguous(memory_format=torch.channels_last)

            # Inference
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)  # (1,17,240,240)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # Postprocess -> keypoints + confidence
            if args.softmax:
                x, y, c = soft_argmax_2d_logits(logits, beta=2.0)
            else:
                x, y, c = argmax_2d(logits)

            xy = torch.stack([x, y], dim=-1).cpu().numpy()       # (17,2)
            conf_kp = c.clamp(0, 1).cpu().numpy()                # (17,)

            # Make display frame (apply ToF conf mask from raw confidence)
            # Re-compute padding on *raw* (un-normalized) to keep visualization consistent
            depth_p_raw, conf_p_raw, _ = pad_to_square(depth, conf, (240, 240))
            disp = make_display(depth_p_raw, conf_p_raw, args.range_mm, args.conf_thresh)

            # Draw skeleton (only draw if peak prob >= heatmap_thresh)
            disp = draw_skeleton(disp, xy, conf_kp, heatmap_thresh=args.heatmap_thresh)

            # Timing overlays
            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ma = (1 - alpha) * fps_ma + alpha * inst_fps if fps_ma > 0 else inst_fps

            cv2.putText(disp, f"FPS: {fps_ma:5.1f}  (inf: {infer_ms:4.1f} ms)",
                        (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (50,255,50), 2)
            cv2.putText(disp, f"Drawn KPs (>{args.heatmap_thresh:.2f}): {int((conf_kp >= args.heatmap_thresh).sum())}/17",
                        (8, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(disp, f"Arg: {'softmax' if args.softmax else 'argmax'} | Range: {int(args.range_mm)}mm | ConfMask>={args.conf_thresh}",
                        (8, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)

            # Scale and show
            if args.scale != 1:
                disp = cv2.resize(disp, (disp.shape[1]*args.scale, disp.shape[0]*args.scale), interpolation=cv2.INTER_NEAREST)
            cv2.imshow("EdgePoseUNetV2 Realtime", disp)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
        cv2.destroyAllWindows()
        print("[INFO] Closed cleanly.")

if __name__ == "__main__":
    main()
