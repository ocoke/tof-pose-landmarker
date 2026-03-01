import argparse
import os
import sys
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

if __package__ in (None, ""):
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model_v7.dataset import PoseDatasetV7
    from model_v7.manifest import DEFAULT_MANIFEST, build_or_load_manifest, get_records_for_split
    from model_v7.model import EdgePoseUNetV7
else:
    from .dataset import PoseDatasetV7
    from .manifest import DEFAULT_MANIFEST, build_or_load_manifest, get_records_for_split
    from .model import EdgePoseUNetV7


def soft_argmax_2d(logits: torch.Tensor, beta: float = 4.0) -> torch.Tensor:
    batch, num_keypoints, height, width = logits.shape
    probs = torch.softmax(logits.view(batch, num_keypoints, -1) * beta, dim=-1).view(batch, num_keypoints, height, width)
    xs = torch.linspace(0, width - 1, width, device=logits.device, dtype=torch.float32).view(1, 1, 1, width)
    ys = torch.linspace(0, height - 1, height, device=logits.device, dtype=torch.float32).view(1, 1, height, 1)
    pred_x = (probs * xs).sum(dim=(2, 3))
    pred_y = (probs * ys).sum(dim=(2, 3))
    return torch.stack([pred_x, pred_y], dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize v7 predictions.")
    parser.add_argument("--model", type=str, required=True, help="Path to checkpoint.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root.")
    parser.add_argument("--manifest", type=str, default=None, help="Manifest path.")
    parser.add_argument("--split", type=str, default="val_seen", help="Split to visualize.")
    parser.add_argument("--count", type=int, default=4, help="Number of samples to visualize.")
    parser.add_argument("--save-dir", type=str, default=None, help="Optional directory to save images.")
    args = parser.parse_args()

    manifest_path = args.manifest or os.path.join(args.data, DEFAULT_MANIFEST)
    records = build_or_load_manifest(args.data, manifest_path=manifest_path)
    split_records = get_records_for_split(records, args.split)
    dataset = PoseDatasetV7(args.data, records=split_records, augment=False, min_valid_keypoints=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgePoseUNetV7().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device), strict=False)
    model.eval()

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)

    shown = 0
    with torch.no_grad():
        for sample_idx, (inputs, _targets, gt_kpts, valid_mask, meta) in enumerate(loader):
            inputs = inputs.to(device)
            pred_xy = soft_argmax_2d(model(inputs)).cpu().numpy()[0]
            input_np = inputs.cpu().numpy()[0]
            gt_np = gt_kpts.numpy()[0]
            valid_np = valid_mask.numpy()[0].astype(bool)

            depth_map = input_np[0]
            confidence_map = input_np[1]
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(depth_map, cmap="viridis")
            axes[0].set_title("Depth")
            axes[1].imshow(confidence_map, cmap="magma")
            axes[1].set_title("Confidence")
            axes[2].imshow(depth_map, cmap="viridis")
            axes[2].set_title(f"Pred vs GT: {meta['sample_id'][0]}")

            for axis in axes:
                axis.axis("off")

            axes[2].scatter(gt_np[valid_np, 0], gt_np[valid_np, 1], c="lime", s=18, label="GT")
            axes[2].scatter(pred_xy[valid_np, 0], pred_xy[valid_np, 1], c="cyan", s=18, label="Pred")
            axes[2].legend(loc="lower right")
            fig.tight_layout()

            if args.save_dir:
                out_path = os.path.join(args.save_dir, f"v7_sample_{sample_idx:04d}.png")
                fig.savefig(out_path, dpi=150)
                print(f"Saved {out_path}")
                plt.close(fig)
            else:
                plt.show()

            shown += 1
            if shown >= args.count:
                break


if __name__ == "__main__":
    main()
