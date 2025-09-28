#!/usr/bin/env python3
"""
Realtime EdgePoseUNetV2 (FP32) demo on Arducam ToF stream (PyTorch) with:
- Person/ROI mask (ToF confidence + depth) applied BEFORE decoding (kills "door skeletons")
- Confidence gating (hide when not enough confident keypoints)
- Optional skeleton toggle (--disable-skeleton) to draw points only
- Optional median + EMA smoothing (--med-window, --ema)
- Separate thresholds for model ROI vs display mask

Usage:
  python demo_edge_v5_realtime.py --model /path/to/edge_5.0.1.pth

Notes:
- The *display* confidence mask (--conf-thresh) only affects visualization.
- The *ROI* mask (--roi-*) affects decoding (if enabled), matching the behavior we discussed.
"""

import os
import sys
import time
import argparse
from collections import deque
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

# Import model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from src.model_edge_v5.model import EdgePoseUNetV2
except ImportError:
    print("[ERROR] Could not import EdgePoseUNetV2 from model_edge_v5.model")
    sys.exit(1)

# COCO-17 skeleton edges (indices 0..16)
COCO17_EDGES = [
    (5, 7), (7, 9),      # Left arm
    (6, 8), (8, 10),     # Right arm
    (5, 6),              # Shoulders
    (5, 11), (6, 12),    # Torso
    (11, 12),            # Hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (0, 1), (0, 2), (1, 3), (2, 4)  # Face
]

# ---------------------- Utils ---------------------- #
def pad_to_square(depth: np.ndarray, conf: np.ndarray, out_hw=(240, 240)) -> Tuple[np.ndarray, np.ndarray, Tuple[int,int,int,int]]:
    H, W = depth.shape
    outH, outW = out_hw
    pad_left = (outW - W) // 2
    pad_right = outW - W - pad_left
    pad_top = (outH - H) // 2
    pad_bottom = outH - H - pad_top
    depth_p = np.pad(depth, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    conf_p  = np.pad(conf,  ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant')
    return depth_p, conf_p, (pad_left, pad_top, pad_right, pad_bottom)

def argmax_2d(logits: torch.Tensor):
    """Return (x, y, peak_prob) from logits via argmax."""
    with torch.no_grad():
        _, K, H, W = logits.shape
        flat = logits.view(1, K, -1)
        idx  = flat.argmax(dim=-1)                   # (1,K)
        x = (idx % W).float().squeeze(0)             # (K,)
        y = (idx // W).float().squeeze(0)            # (K,)
        # confidence = sigmoid at the argmax location
        sig = torch.sigmoid(logits).view(1, K, -1)
        peak = torch.gather(sig, 2, idx.unsqueeze(-1)).squeeze(0).squeeze(-1)
        return x, y, peak

def soft_argmax_2d_logits(logits: torch.Tensor, beta: float = 2.0):
    """Soft-argmax over H*W. Returns (x, y, peak_prob) where peak_prob=max softmax prob."""
    with torch.no_grad():
        _, K, H, W = logits.shape
        flat = (logits * beta).view(1, K, -1)
        sm   = torch.softmax(flat, dim=-1)           # (1,K,H*W)
        ys = torch.arange(H, device=logits.device, dtype=torch.float32).view(1,1,H,1).expand(1,K,H,W).reshape(1,K,-1)
        xs = torch.arange(W, device=logits.device, dtype=torch.float32).view(1,1,1,W).expand(1,K,H,W).reshape(1,K,-1)
        ex = (sm * xs).sum(dim=-1).squeeze(0)        # (K,)
        ey = (sm * ys).sum(dim=-1).squeeze(0)        # (K,)
        peak_prob, _ = sm.max(dim=-1)                # (1,K)
        return ex, ey, peak_prob.squeeze(0)

def make_display(depth_p: np.ndarray, conf_p: np.ndarray, range_mm: float, conf_thresh: int) -> np.ndarray:
    """Visualization frame only (does not affect model)."""
    depth_norm = np.clip(depth_p / range_mm, 0, 1)
    img = (depth_norm * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    mask = conf_p >= conf_thresh
    img[~mask] = (0, 0, 0)
    return img

def draw_pose(img: np.ndarray,
              xy: np.ndarray,
              conf: np.ndarray,
              heatmap_thresh: float = 0.3,
              draw_lines: bool = True,
              hide_low_conf_points: bool = True) -> np.ndarray:
    """Draw COCO-17 pose with per-kp confidence gating."""
    K = xy.shape[0]
    H, W = img.shape[:2]

    if draw_lines:
        for a, b in COCO17_EDGES:
            if a < K and b < K:
                xa, ya = xy[a]; xb, yb = xy[b]
                ca, cb = float(conf[a]), float(conf[b])
                if 0 <= xa < W and 0 <= ya < H and 0 <= xb < W and 0 <= yb < H:
                    avg_c = (ca + cb) * 0.5
                    if avg_c >= heatmap_thresh:
                        color, th = (0, 255, 255), 2
                        cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)), color, th)

    for i in range(K):
        x, y = xy[i]; c = float(conf[i])
        if 0 <= x < W and 0 <= y < H:
            if c >= heatmap_thresh:
                col, rad, border = (0, 255, 0), 4, (255,255,255)
            else:
                if hide_low_conf_points:
                    continue
                col, rad, border = (128,128,128), 2, (200,200,200)
            cv2.circle(img, (int(x), int(y)), rad, col, -1)
            cv2.circle(img, (int(x), int(y)), rad+2, border, 1)
    return img

def build_person_mask(depth_raw: np.ndarray,
                      conf_raw: np.ndarray,
                      out_hw=(240, 240),
                      conf_thr: int = 30,
                      depth_min: int = 300,
                      depth_max: int = 3000,
                      close_iter: int = 1,
                      dilate_iter: int = 0,
                      kernel_size: int = 5) -> Tuple[np.ndarray, int]:
    """
    Build a binary person ROI mask at model/output resolution.
    Returns (mask_0_1_float, area_pixels).
    """
    # Base boolean mask
    mask = (conf_raw >= conf_thr) & (depth_raw >= depth_min) & (depth_raw <= depth_max)
    mask = mask.astype(np.uint8) * 255

    # Morphology (optional)
    k = max(1, int(kernel_size))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    if close_iter > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    if dilate_iter > 0:
        mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

    # Pad to square (use same function, treating as "depth" & "conf" just to reuse pads)
    mask_p, _, _ = pad_to_square(mask, mask, out_hw)
    area = int((mask_p > 0).sum())

    # Normalize to 0..1 float
    mask_f = (mask_p > 0).astype(np.float32)
    return mask_f, area

def apply_roi_mask_to_logits(logits: torch.Tensor, mask_01: np.ndarray) -> torch.Tensor:
    """
    Apply ROI mask by setting logits to a large negative where mask==0.
    logits: (1,K,H,W)   mask_01: (H,W) in {0,1}
    """
    if mask_01 is None:
        return logits
    with torch.no_grad():
        m = torch.from_numpy(mask_01).to(logits.device, dtype=torch.bool)  # (H,W) True where valid
        m = m.view(1, 1, *m.shape).expand_as(logits)                        # (1,K,H,W)
        masked = logits.clone()
        masked = masked.masked_fill(~m, -1e4)  # kill non-ROI positions
        return masked

def nanmedian_xy(stack_xy: np.ndarray) -> np.ndarray:
    """
    stack_xy: (T, K, 2) with possible NaNs; returns (K,2) nanmedian.
    If all-NaN for a coordinate, result is NaN.
    """
    return np.nanmedian(stack_xy, axis=0)

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

    def get(self, timeout_ms: int = 200):
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
            self.cam.stop(); self.cam.close()
        except:
            pass

# ---------------------- Main Demo ---------------------- #
def main():
    ap = argparse.ArgumentParser("EdgePoseUNetV2 ToF Realtime Demo (PyTorch)")
    ap.add_argument("--model", required=True, type=str, help="Path to .pth weights")
    ap.add_argument("--device", default="auto", choices=["auto","cpu","cuda"], help="Device selection")
    ap.add_argument("--width-mult", type=float, default=1.15, help="Width multiplier")
    ap.add_argument("--range-mm", type=float, default=4000.0, help="ToF depth range for normalization")
    ap.add_argument("--no-rotate", action="store_true", help="Do not rotate ToF images 180°")

    # Visualization-only mask (does NOT affect model)
    ap.add_argument("--conf-thresh", type=int, default=30, help="Display confidence mask threshold (0..255)")

    # Decoding ROI/person mask (affects predictions if enabled)
    ap.add_argument("--no-roi-mask", action="store_true", help="Disable ROI/person mask on logits")
    ap.add_argument("--roi-conf", type=int, default=30, help="ROI: ToF confidence threshold (0..255)")
    ap.add_argument("--roi-depth-min", type=int, default=300, help="ROI: min depth in mm")
    ap.add_argument("--roi-depth-max", type=int, default=3000, help="ROI: max depth in mm")
    ap.add_argument("--roi-kernel", type=int, default=5, help="ROI: morphology kernel size")
    ap.add_argument("--roi-close", type=int, default=1, help="ROI: morphological CLOSE iterations")
    ap.add_argument("--roi-dilate", type=int, default=0, help="ROI: DILATE iterations")
    ap.add_argument("--min-blob-area", type=int, default=200, help="ROI: min mask pixels to accept")

    # Drawing/metrics
    ap.add_argument("--heatmap-thresh", type=float, default=0.30, help="Min per-kp confidence to draw")
    ap.add_argument("--min-kps", type=int, default=6, help="Min confident keypoints to consider person present")
    ap.add_argument("--disable-skeleton", action="store_true", help="Draw points only (no lines)")
    ap.add_argument("--softmax", action="store_true", help="Use soft-argmax (beta=2.0) instead of argmax")

    # Smoothing
    ap.add_argument("--med-window", type=int, default=1, help="Median window (frames). 1=off")
    ap.add_argument("--ema", type=float, default=0.0, help="EMA factor (0..0.99), 0=off")

    # Perf/UI
    ap.add_argument("--threads", type=int, default=4, help="torch.set_num_threads")
    ap.add_argument("--scale", type=int, default=3, help="Display scale")
    args = ap.parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[INFO] Using device: {device}")

    # Torch threading
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

    # Warm-up
    with torch.inference_mode():
        dummy = torch.zeros(1, 2, 240, 240, dtype=torch.float32, device=device).contiguous(memory_format=torch.channels_last)
        _ = model(dummy)

    # Camera
    stream = ToFStream(range_mm=args.range_mm, rotate_180=(not args.no_rotate))
    if not stream.open():
        sys.exit(1)

    cv2.namedWindow("EdgePoseUNetV2 Realtime", cv2.WINDOW_AUTOSIZE)

    last_t = time.time()
    fps_ma = 0.0
    alpha = 0.05  # EMA for FPS

    # Smoothing buffers
    prev_xy = None
    med_buf: Optional[deque] = deque(maxlen=max(1, args.med_window)) if args.med_window > 1 else None

    try:
        while True:
            depth, conf = stream.get()
            if depth is None:
                continue

            # -------------------- MODEL INPUT (unchanged) --------------------
            depth_norm = np.clip(depth / args.range_mm, 0, 1).astype(np.float32)
            conf_norm  = np.clip(conf  / 255.0,        0, 1).astype(np.float32)
            depth_p, conf_p, _ = pad_to_square(depth_norm, conf_norm, (240, 240))
            inp = np.stack([depth_p, conf_p], axis=0)  # (2,240,240)
            tensor = torch.from_numpy(inp).unsqueeze(0).to(device).contiguous(memory_format=torch.channels_last)

            # -------------------- ROI/PERSON MASK (affects decoding) --------------------
            roi_mask_01 = None
            roi_area = 0
            if not args.no_roi_mask:
                roi_mask_01, roi_area = build_person_mask(
                    depth_raw=depth,
                    conf_raw=conf,
                    out_hw=(240, 240),
                    conf_thr=args.roi_conf,
                    depth_min=args.roi_depth_min,
                    depth_max=args.roi_depth_max,
                    close_iter=args.roi_close,
                    dilate_iter=args.roi_dilate,
                    kernel_size=args.roi_kernel
                )

            # Inference
            t0 = time.perf_counter()
            with torch.inference_mode():
                logits = model(tensor)  # (1,17,240,240)

                # Apply ROI mask to logits before decoding (if enabled & area ok)
                if (roi_mask_01 is not None) and (roi_area >= args.min_blob_area):
                    logits = apply_roi_mask_to_logits(logits, roi_mask_01)
            infer_ms = (time.perf_counter() - t0) * 1000.0

            # -------------------- Decode -> keypoints + confidence --------------------
            if args.softmax:
                x, y, c = soft_argmax_2d_logits(logits, beta=2.0)
            else:
                x, y, c = argmax_2d(logits)

            xy = torch.stack([x, y], dim=-1).cpu().numpy()    # (17,2)
            conf_kp = c.clamp(0, 1).cpu().numpy()             # (17,)

            # ---- Confidence gating & presence detection ----
            valid_mask = conf_kp >= args.heatmap_thresh
            valid_count = int(valid_mask.sum())
            roi_ok = (roi_area >= args.min_blob_area) if (roi_mask_01 is not None) else True
            person_present = (valid_count >= args.min_kps) and roi_ok

            # -------------------- Median + EMA smoothing --------------------
            # 1) median across last N frames (for currently valid kps). Use NaNs for invalid to ignore.
            if med_buf is not None:
                xy_for_med = xy.copy()
                # put NaNs where keypoint is low-conf so median ignores it
                xy_for_med[~valid_mask] = np.nan
                med_buf.append(xy_for_med)
                med_xy = nanmedian_xy(np.stack(med_buf, axis=0))  # (K,2)
                # if median is NaN for some kp (all invalid), fall back to current xy
                xy = np.where(np.isnan(med_xy), xy, med_xy)

            # 2) EMA smoothing (frame-to-frame)
            if args.ema > 0.0 and args.ema < 1.0 and person_present:
                if prev_xy is None:
                    prev_xy = xy.copy()
                xy = args.ema * prev_xy + (1.0 - args.ema) * xy
                prev_xy = xy.copy()
            else:
                prev_xy = None if not person_present else xy.copy()

            # -------------------- Visualization frame (separate mask) --------------------
            depth_p_raw, conf_p_raw, _ = pad_to_square(depth, conf, (240, 240))
            disp = make_display(depth_p_raw, conf_p_raw, args.range_mm, args.conf_thresh)

            # Draw ROI contour (optional helpful for debugging)
            if (roi_mask_01 is not None):
                roi_vis = (roi_mask_01 * 255).astype(np.uint8)
                contours, _ = cv2.findContours(roi_vis, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, contours, -1, (60, 120, 255), 1)

            # Draw pose if person present; else show "No person"
            if person_present:
                draw_pose(
                    disp,
                    xy,
                    conf_kp,
                    heatmap_thresh=args.heatmap_thresh,
                    draw_lines=(not args.disable_skeleton),
                    hide_low_conf_points=True
                )
            else:
                cv2.putText(disp, "No person", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)

            # Timing overlays
            now = time.time()
            dt = now - last_t
            last_t = now
            inst_fps = (1.0 / dt) if dt > 0 else 0.0
            fps_ma = (1 - alpha) * fps_ma + alpha * inst_fps if fps_ma > 0 else inst_fps

            mode_txt = "soft-argmax" if args.softmax else "argmax"
            roi_txt = "ON" if not args.no_roi_mask else "OFF"

            cv2.putText(disp, f"FPS: {fps_ma:5.1f}  (inf: {infer_ms:4.1f} ms)",
                        (8, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50,255,50), 2)
            cv2.putText(disp, f"Drawn KPs (>={args.heatmap_thresh:.2f}): {valid_count}/17 | ema={args.ema:.2f} | medN={args.med_window}",
                        (8, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
            cv2.putText(disp, f"Mode:{mode_txt} | ROI:{roi_txt} (conf>={args.roi_conf}, {args.roi_depth_min}-{args.roi_depth_max}mm, area={roi_area}) "
                              f"| DispMask>={args.conf_thresh} | Lines:{not args.disable_skeleton}",
                        (8, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,0), 1)

            # Scale & show
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
