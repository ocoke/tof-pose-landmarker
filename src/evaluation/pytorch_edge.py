# evaluate_pytorch.py
import os
import sys
import time
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# --- Make local imports work ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Models ---
from model.model import PoseUNet
from model_slim.model import SlimPoseUNet
from model_slim.medium_model import MediumPoseUNet
from model_edge.model import EdgePoseUNet
from model_edge_v5.model import EdgePoseUNetV2

# --- Datasets ---
from model.dataset import PoseDataset
from model_slim.dataset import SlimPoseDataset
from model_edge_v5.dataset import EdgePoseDataset

# ======= Utilities =======

def print_results(results_dict):
    print("\n" + "=" * 30)
    print("      PERFORMANCE METRICS")
    print("=" * 30)
    print(f"{'Metric':<25} | {'Value':<15}")
    print("-" * 43)
    print(f"{'Avg. Latency (ms)':<25} | {results_dict['avg_latency_ms']:.2f}")
    print(f"{'Theoretical FPS':<25} | {results_dict['fps']:.2f}")
    print(f"{'Avg. Error (pixels)':<25} | {results_dict['mpjpe_pixels']:.2f}")
    print(f"{'PCK Accuracy (%)':<25} | {results_dict['pck_accuracy']:.2f}")
    print(f"{'PCK@10% Accuracy (%)':<25} | {results_dict['pck_10%_accuracy']:.2f}")
    print("=" * 30)

def argmax_2d(logits: torch.Tensor) -> torch.Tensor:
    # logits: (B,K,H,W)
    B, K, H, W = logits.shape
    flat = logits.view(B, K, -1)
    idx  = flat.argmax(dim=-1)           # (B,K)
    x    = (idx % W).float()
    y    = (idx // W).float()
    return torch.stack([x, y], dim=-1)   # (B,K,2)

def soft_argmax_2d_logits_beta(logits: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """
    Soft-argmax that matches your training loss:
    - Softmax over H*W (NOT sigmoid), with sharpening temperature `beta`.
    """
    B, K, H, W = logits.shape
    prob = torch.softmax(logits.view(B, K, -1) * beta, dim=-1).view(B, K, H, W)

    xs = torch.linspace(0, W - 1, W, device=logits.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.linspace(0, H - 1, H, device=logits.device, dtype=torch.float32).view(1, 1, H, 1)

    ex = (prob * xs).sum(dim=(2, 3))  # (B,K)
    ey = (prob * ys).sum(dim=(2, 3))  # (B,K)
    return torch.stack([ex, ey], dim=-1)  # (B,K,2)

def build_model(arch: str, device: torch.device):
    if arch == 'heavy':
        return PoseUNet(n_channels=2, n_keypoints=33).to(device)
    if arch == 'slim':
        return SlimPoseUNet(in_ch=2, n_kpts=17).to(device)
    if arch == 'medium':
        return MediumPoseUNet(in_ch=2, n_kpts=17).to(device)
    if arch == 'edge':
        return EdgePoseUNet(in_ch=2, n_kpts=17, width_mult=1.15, p_drop=0.02).to(device)
    if arch == 'edge_v5':
        return EdgePoseUNetV2(in_ch=2, n_kpts=17, width_mult=1.15).to(device)
    raise ValueError(f"Unknown architecture '{arch}'.")

def build_dataset(arch: str, data_dir: str):
    """
    Keep random split; use the dataset that matches the arch.
    Note: EdgePoseDataset sorts its file list internally, which makes
    the random split deterministic for a given seed.
    """
    if arch == 'heavy':
        return PoseDataset(data_dir=data_dir, augment=False, num_keypoints=33)
    if arch in ('slim', 'medium', 'edge'):
        return SlimPoseDataset(data_dir=data_dir, augment=False, num_keypoints=17)
    if arch == 'edge_v5':
        return EdgePoseDataset(data_dir=data_dir, augment=False, num_keypoints=17)
    raise ValueError(f"Unknown architecture '{arch}'.")


# ======= Main evaluation =======

def evaluate_pytorch_model(model_path, arch, val_loader, device, metric='soft', beta=4.0,
                           img_size=240, pck_ratio_5=0.05, pck_ratio_10=0.10):
    print(f"\n--- Evaluating PyTorch Model: {os.path.basename(model_path)} ---")

    model = build_model(arch, device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    inference_times = []
    all_errors = []
    pck_scores = []
    pck_10_scores = []

    # Compute fixed-pixel thresholds from resolution (you pad to 240x240)
    H = W = img_size
    pck_threshold = int(pck_ratio_5 * max(H, W))      # e.g., 12 px for 240
    pck_10_threshold = int(pck_ratio_10 * max(H, W))  # e.g., 24 px for 240

    with torch.no_grad():
        for inputs, _targets, gt_kpts in val_loader:
            inputs = inputs.to(device)

            t0 = time.perf_counter()
            logits = model(inputs)  # (B,K,H,W)
            inference_times.append((time.perf_counter() - t0) * 1000)

            if metric == 'soft':
                pred_xy = soft_argmax_2d_logits_beta(logits, beta=beta).cpu().numpy()
            elif metric == 'argmax':
                pred_xy = argmax_2d(logits).cpu().numpy()
            else:
                raise ValueError("metric must be 'soft' or 'argmax'.")

            gt_xy = gt_kpts.numpy().astype(np.float32)

            # MPJPE / PCK per sample (batch_size=1)
            d = np.linalg.norm(pred_xy[0] - gt_xy[0], axis=1)  # (K,)
            all_errors.extend(d)
            pck_scores.append(np.mean(d < pck_threshold))
            pck_10_scores.append(np.mean(d < pck_10_threshold))

    print_results({
        "avg_latency_ms": float(np.mean(inference_times)),
        "fps": float(1000.0 / np.mean(inference_times)),
        "mpjpe_pixels": float(np.mean(all_errors)),
        "pck_accuracy": float(np.mean(pck_scores) * 100.0),
        "pck_10%_accuracy": float(np.mean(pck_10_scores) * 100.0),
    })


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate a PyTorch Pose Estimation Model")
    parser.add_argument("--model", type=str, required=True, help="Path to the .pth model file.")
    parser.add_argument("--arch", type=str, required=True,
                        choices=['heavy', 'slim', 'medium', 'edge', 'edge_v5'],
                        help="Model architecture.")
    parser.add_argument("--data", type=str, default="./data", help="Path to the data directory.")
    parser.add_argument("--batch", type=int, default=1, help="Batch size for evaluation (default 1).")
    parser.add_argument("--seed", type=int, default=720, help="Random split seed (default 720).")
    parser.add_argument("--metric", type=str, default="soft", choices=["soft", "argmax"],
                        help="Coordinate extraction method. 'soft' matches training (default).")
    parser.add_argument("--beta", type=float, default=4.0, help="Softmax sharpening beta (default 4.0).")
    parser.add_argument("--img_size", type=int, default=240, help="Square image size after padding (default 240).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load dataset & random split (still random; deterministic via seed) ---
    print(f"Loading data for '{args.arch}' architecture...")
    full_dataset = build_dataset(args.arch, args.data)

    val_size = int(0.2 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(dataset=val_dataset, batch_size=args.batch, shuffle=False)

    evaluate_pytorch_model(
        model_path=args.model,
        arch=args.arch,
        val_loader=val_loader,
        device=device,
        metric=args.metric,
        beta=args.beta,
        img_size=args.img_size,
        pck_ratio_5=0.05,
        pck_ratio_10=0.10
    )
