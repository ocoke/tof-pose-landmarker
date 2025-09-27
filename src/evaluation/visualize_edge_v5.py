# visualize_edge_v5_points.py
import os
import sys
import time
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split

# Make local package imports work (assumes this file is in src/visualization or similar)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- Model & Dataset for edge_v5 ---
from model_edge_v5.model import EdgePoseUNetV2
from model_edge_v5.dataset import EdgePoseDataset

# ======= Coord utilities (match training/eval) =======

def argmax_2d(logits: torch.Tensor) -> torch.Tensor:
    """ (B,K,2) integer-like coords from argmax over H×W """
    B, K, H, W = logits.shape
    flat = logits.view(B, K, -1)
    idx  = flat.argmax(dim=-1)           # (B,K)
    x    = (idx % W).float()
    y    = (idx // W).float()
    return torch.stack([x, y], dim=-1)   # (B,K,2)

def soft_argmax_2d_logits_beta(logits: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    """
    Soft-argmax that matches heatmap training better:
    - Softmax over H*W (NOT sigmoid), with sharpening temperature `beta`.
    """
    B, K, H, W = logits.shape
    prob = torch.softmax(logits.view(B, K, -1) * beta, dim=-1).view(B, K, H, W)

    xs = torch.linspace(0, W - 1, W, device=logits.device, dtype=torch.float32).view(1, 1, 1, W)
    ys = torch.linspace(0, H - 1, H, device=logits.device, dtype=torch.float32).view(1, 1, H, 1)

    ex = (prob * xs).sum(dim=(2, 3))  # (B,K)
    ey = (prob * ys).sum(dim=(2, 3))  # (B,K)
    return torch.stack([ex, ey], dim=-1)  # (B,K,2)

# ======= Visualization helpers =======

def extract_coords(pred_logits: torch.Tensor, targets: torch.Tensor, metric: str, beta: float):
    """
    Returns: pred_coords (K,2), gt_coords (K,2), pred_heatmaps (K,H,W), gt_heatmaps (K,H,W)
    """
    # logits -> (B,K,H,W)
    with torch.no_grad():
        if metric == "soft":
            pred_xy = soft_argmax_2d_logits_beta(pred_logits, beta=beta)[0].cpu().numpy()  # (K,2)
        elif metric == "argmax":
            pred_xy = argmax_2d(pred_logits)[0].cpu().numpy()  # (K,2)
        else:
            raise ValueError("metric must be 'soft' or 'argmax'")

        # For heatmap visualization
        pred_heatmaps = torch.sigmoid(pred_logits)[0].cpu().numpy()  # (K,H,W)
        gt_heatmaps   = targets[0].cpu().numpy()                     # (K,H,W)

        # GT keypoints in pixel coords are available in loader tuple, but we’ll extract from gt argmax for display parity
        # If you prefer exact provided coords, pass gt_kpts from the loader directly.
        # Here we’ll compute from heatmaps (visual alignment).
        K, H, W = gt_heatmaps.shape
        gt_xy = np.zeros((K, 2), dtype=np.float32)
        for k in range(K):
            idx = np.argmax(gt_heatmaps[k])
            y, x = divmod(idx, W)
            gt_xy[k] = [x, y]

        return pred_xy, gt_xy, pred_heatmaps, gt_heatmaps

def show_sample_panels(depth_map, gt_heat_sum, pred_heat_sum, gt_xy, pred_xy,
                       sample_idx: int, save_path: str | None = None):
    """
    depth_map: (H,W)
    heatmap sums: (H,W)
    gt_xy, pred_xy: (K,2)
    """
    fig, axs = plt.subplots(1, 5, figsize=(26, 5))
    fig.suptitle(f'Validation Sample #{sample_idx}', fontsize=16)

    axs[0].imshow(depth_map, cmap='viridis', aspect='equal')
    axs[0].set_title('Input Depth Map'); axs[0].axis('off')

    axs[1].imshow(gt_heat_sum, cmap='hot', aspect='equal')
    axs[1].set_title('Ground Truth Heatmap (sum)'); axs[1].axis('off')

    axs[2].imshow(pred_heat_sum, cmap='hot', aspect='equal')
    axs[2].set_title('Prediction Heatmap (sum)'); axs[2].axis('off')

    axs[3].imshow(depth_map, cmap='viridis', aspect='equal')
    axs[3].imshow(pred_heat_sum, cmap='hot', alpha=0.6, aspect='equal')
    axs[3].set_title('Prediction Overlay'); axs[3].axis('off')

    axs[4].imshow(depth_map, cmap='viridis', aspect='equal')
    axs[4].scatter(gt_xy[:, 0],   gt_xy[:, 1],   s=40, c='lime', marker='o', label='GT')
    axs[4].scatter(pred_xy[:, 0], pred_xy[:, 1], s=40, c='red',  marker='x', label='Pred')
    axs[4].set_title('Final Points Overlay'); axs[4].axis('off')
    axs[4].legend(loc='upper right', fontsize=9, frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()

# ======= Main visualization =======

def visualize_edge_v5_points(model_path: str,
                             data_dir: str,
                             width_mult: float = 1.15,
                             batch_size: int = 4,
                             num_samples: int = 8,
                             val_split: float = 0.2,
                             seed: int = 720,
                             metric: str = "soft",
                             beta: float = 4.0,
                             save_dir: str | None = None):
    """
    Visualize edge_v5 predictions with point overlays.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Model ---
    model = EdgePoseUNetV2(in_ch=2, n_kpts=17, width_mult=width_mult).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model weights loaded successfully.")

    # --- Dataset & split (random but reproducible) ---
    print("Creating dataset and random validation split…")
    full_dataset = EdgePoseDataset(data_dir=data_dir, output_res=(240, 240), augment=False, num_keypoints=17)
    dataset_size = len(full_dataset)
    val_size     = int(dataset_size * val_split)
    train_size   = dataset_size - val_size
    print(f"Total: {dataset_size} | Train: {train_size} | Val: {val_size}")

    _, val_dataset = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    # --- Iterate and visualize ---
    shown = 0
    with torch.no_grad():
        for i, (inputs, targets, gt_kpts) in enumerate(val_loader):
            if shown >= num_samples:
                break

            inputs  = inputs.to(device)        # (B,2,H,W)
            logits  = model(inputs)            # (B,K,H,W) logits
            probs   = torch.sigmoid(logits)    # for heatmap viz only

            inputs_np  = inputs.cpu().numpy()
            targets_np = targets.cpu().numpy()
            probs_np   = probs.cpu().numpy()

            B = inputs_np.shape[0]
            for j in range(B):
                if shown >= num_samples:
                    break
                sample_idx = i * val_loader.batch_size + j

                # Depth channel for background
                depth_map = inputs_np[j, 0, :, :]

                # Heatmap sums
                gt_heat_sum   = np.sum(targets_np[j], axis=0)
                pred_heat_sum = np.sum(probs_np[j],   axis=0)

                # Coords (match training by default with 'soft')
                # NOTE: We compute coords from LOGITS (not probs) for consistency with softmax over logits.
                pred_xy, gt_xy, _, _ = extract_coords(
                    pred_logits=logits[j:j+1],   # keep batch dim = 1
                    targets=targets[j:j+1],
                    metric=metric,
                    beta=beta
                )

                # Save or show
                save_path = None
                if save_dir is not None:
                    save_path = os.path.join(save_dir, f"edge_v5_sample_{sample_idx:04d}.png")

                show_sample_panels(depth_map, gt_heat_sum, pred_heat_sum, gt_xy, pred_xy, sample_idx, save_path)
                shown += 1

    print(f"Done. Visualized {shown} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize EdgePoseUNetV2 (edge_v5) predictions with keypoint overlays.")
    parser.add_argument("--model", type=str, required=True, help="Path to .pth weights (edge_v5).")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root directory.")
    parser.add_argument("--width-mult", type=float, default=1.15, help="Width multiplier used at training (default 1.15).")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for visualization loader (default 4).")
    parser.add_argument("--num", type=int, default=8, help="Number of validation samples to visualize (default 8).")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio (default 0.2).")
    parser.add_argument("--seed", type=int, default=720, help="Random split seed (default 720).")
    parser.add_argument("--metric", type=str, default="soft", choices=["soft", "argmax"],
                        help="Coordinate method: 'soft' (softmax over H*W, beta) or 'argmax'.")
    parser.add_argument("--beta", type=float, default=4.0, help="Softmax sharpening beta for soft-argmax (default 4.0).")
    parser.add_argument("--save-dir", type=str, default=None, help="If set, save figures to this directory instead of showing.")
    args = parser.parse_args()

    visualize_edge_v5_points(
        model_path=args.model,
        data_dir=args.data,
        width_mult=args.width_mult,
        batch_size=args.batch_size,
        num_samples=args.num,
        val_split=args.val_split,
        seed=args.seed,
        metric=args.metric,
        beta=args.beta,
        save_dir=args.save_dir
    )
