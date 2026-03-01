import argparse
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler

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


def masked_bce_with_logits(logits: torch.Tensor, targets: torch.Tensor, valid_mask: torch.Tensor, pos_weight: float) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none",
        pos_weight=torch.tensor([pos_weight], device=logits.device),
    )
    expanded = valid_mask.unsqueeze(-1).unsqueeze(-1).float()
    denom = expanded.sum() * logits.shape[-1] * logits.shape[-2]
    denom = torch.clamp(denom, min=1.0)
    return (loss * expanded).sum() / denom


def masked_smooth_l1(pred_xy: torch.Tensor, gt_xy: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none").mean(dim=-1)
    weights = valid_mask.float()
    denom = torch.clamp(weights.sum(), min=1.0)
    return (loss * weights).sum() / denom


def evaluate_loader(
    model: EdgePoseUNetV7,
    loader: Optional[DataLoader],
    device: torch.device,
    pos_weight: float,
    lambda_coord: float,
    normalize_coords: bool,
    beta: float = 4.0,
) -> Optional[Dict[str, float]]:
    if loader is None:
        return None

    model.eval()
    total_loss = 0.0
    total_pck5 = 0.0
    total_pck10 = 0.0
    total_samples = 0

    with torch.no_grad():
        for inputs, targets, gt_kpts, valid_mask, _meta in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            gt_kpts = gt_kpts.to(device)
            valid_mask = valid_mask.to(device)

            heat_logits, vis_logits = model(inputs, return_aux=True)
            pred_xy = soft_argmax_2d(heat_logits, beta=beta)

            coord_pred = pred_xy
            coord_gt = gt_kpts
            if normalize_coords:
                scale = torch.tensor(
                    [heat_logits.shape[-1] - 1, heat_logits.shape[-2] - 1],
                    device=device,
                    dtype=torch.float32,
                )
                coord_pred = pred_xy / scale
                coord_gt = gt_kpts / scale

            heat_loss = masked_bce_with_logits(heat_logits, targets, valid_mask, pos_weight)
            coord_loss = masked_smooth_l1(coord_pred, coord_gt, valid_mask)
            aux_loss = F.binary_cross_entropy_with_logits(vis_logits, valid_mask.float())
            loss = heat_loss + lambda_coord * coord_loss + 0.05 * aux_loss
            total_loss += loss.item()

            dists = torch.norm(pred_xy - gt_kpts, dim=-1)
            weights = valid_mask.float()
            denom = torch.clamp(weights.sum(dim=1), min=1.0)
            pck5 = ((dists <= 12.0).float() * weights).sum(dim=1) / denom
            pck10 = ((dists <= 24.0).float() * weights).sum(dim=1) / denom
            total_pck5 += pck5.sum().item()
            total_pck10 += pck10.sum().item()
            total_samples += inputs.size(0)

    num_batches = max(1, len(loader))
    num_samples = max(1, total_samples)
    return {
        "loss": total_loss / num_batches,
        "pck5": (total_pck5 / num_samples) * 100.0,
        "pck10": (total_pck10 / num_samples) * 100.0,
    }


def make_train_loader(
    dataset: PoseDatasetV7,
    batch_size: int,
    num_workers: int,
    balanced_by_scene: bool,
) -> DataLoader:
    if not balanced_by_scene:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

    scene_counts = Counter(record.get("scene_id", "scene_0") for record in dataset.records)
    sample_weights = [1.0 / scene_counts[record.get("scene_id", "scene_0")] for record in dataset.records]
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )


def split_records_random(records: List[Dict[str, object]], val_ratio: float, seed: int) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not records:
        return [], []
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(records), generator=generator).tolist()
    val_size = max(1, int(len(records) * val_ratio))
    val_indices = set(perm[:val_size])
    train_records = [record for idx, record in enumerate(records) if idx not in val_indices]
    val_records = [record for idx, record in enumerate(records) if idx in val_indices]
    return train_records, val_records


def resolve_stage_defaults(args: argparse.Namespace) -> argparse.Namespace:
    if args.stage == "a":
        args.lr = args.lr or 1e-4
        args.epochs = args.epochs or 150
        args.pos_weight = args.pos_weight or 200.0
        args.lambda_coord = args.lambda_coord or 0.10
        args.normalize_coords = False
    else:
        args.lr = args.lr or 3e-5
        args.epochs = args.epochs or 15
        args.pos_weight = args.pos_weight or 120.0
        args.lambda_coord = args.lambda_coord or 0.25
        args.normalize_coords = True
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the canonical v7 pose model.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root.")
    parser.add_argument("--manifest", type=str, default=None, help="Path to the v7 manifest file.")
    parser.add_argument("--scene-map", type=str, default=None, help="Optional JSON mapping sample_id to scene_id.")
    parser.add_argument("--stage", type=str, default="a", choices=["a", "b"], help="Training stage.")
    parser.add_argument("--resume", type=str, default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Override default epoch count.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--pos-weight", type=float, default=None, help="Override heatmap BCE positive weight.")
    parser.add_argument("--lambda-coord", type=float, default=None, help="Override coordinate loss weight.")
    parser.add_argument("--eval-unseen-every", type=int, default=1, help="Evaluate val_unseen every N epochs.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader worker count.")
    parser.add_argument("--output-dir", type=str, default="./models", help="Directory for checkpoints.")
    parser.add_argument("--conf-hi", type=float, default=350.0, help="Confidence clipping constant.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--rebuild-manifest", action="store_true", help="Recompute the manifest instead of reusing it.")
    parser.add_argument(
        "--split-mode",
        type=str,
        default="auto",
        choices=["auto", "manifest", "random"],
        help="How to form train/validation splits. 'auto' falls back to random when no scene map is available.",
    )
    args = resolve_stage_defaults(parser.parse_args())

    torch.manual_seed(args.seed)
    manifest_path = args.manifest or os.path.join(args.data, DEFAULT_MANIFEST)
    records = build_or_load_manifest(
        data_dir=args.data,
        manifest_path=manifest_path,
        scene_map_path=args.scene_map,
        rebuild=args.rebuild_manifest,
    )

    if args.stage == "b" and not args.resume:
        stage_a_checkpoint = os.path.join(args.output_dir, "v7_stage_a_best.pth")
        if os.path.exists(stage_a_checkpoint):
            args.resume = stage_a_checkpoint

    unique_scenes = sorted({record.get("scene_id", "scene_0") for record in records})
    use_random_split = args.split_mode == "random" or (
        args.split_mode == "auto" and len(unique_scenes) <= 1 and args.scene_map is None
    )

    if use_random_split:
        train_records, val_seen_records = split_records_random(records, val_ratio=0.2, seed=args.seed)
        val_unseen_records = []
    else:
        train_records = get_records_for_split(records, "train_core")
        val_seen_records = get_records_for_split(records, "val_seen")
        val_unseen_records = get_records_for_split(records, "val_unseen")
        if not val_seen_records:
            print("[WARN] Manifest produced no val_seen split. Falling back to a random 80/20 bootstrap split.")
            train_records, val_seen_records = split_records_random(records, val_ratio=0.2, seed=args.seed)
            val_unseen_records = []
            use_random_split = True

    augment_profile = "basic" if args.stage == "a" else "strong"
    train_dataset = PoseDatasetV7(
        args.data,
        records=train_records,
        augment=True,
        augment_profile=augment_profile,
        conf_hi=args.conf_hi,
    )
    val_seen_loader = None
    val_unseen_loader = None

    if val_seen_records:
        val_seen_dataset = PoseDatasetV7(
            args.data,
            records=val_seen_records,
            augment=False,
            min_valid_keypoints=0,
            conf_hi=args.conf_hi,
        )
        val_seen_loader = DataLoader(val_seen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    if val_unseen_records:
        val_unseen_dataset = PoseDatasetV7(
            args.data,
            records=val_unseen_records,
            augment=False,
            min_valid_keypoints=0,
            conf_hi=args.conf_hi,
        )
        val_unseen_loader = DataLoader(val_unseen_dataset, batch_size=args.batch_size, shuffle=False, num_workers=max(1, args.num_workers // 2), pin_memory=True)

    balanced_by_scene = args.stage == "b" and not use_random_split and len(unique_scenes) > 1
    train_loader = make_train_loader(train_dataset, args.batch_size, args.num_workers, balanced_by_scene=balanced_by_scene)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EdgePoseUNetV7(in_ch=3, n_kpts=17, width_mult=1.15).to(device)
    if args.resume and os.path.exists(args.resume):
        state = torch.load(args.resume, map_location=device)
        load_res = model.load_state_dict(state, strict=False)
        if load_res.missing_keys:
            print(f"[WARN] Missing keys when resuming: {load_res.missing_keys}")
        if load_res.unexpected_keys:
            print(f"[WARN] Unexpected keys when resuming: {load_res.unexpected_keys}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"v7_stage_{args.stage}_best.pth")
    best_val_loss = float("inf")
    epochs_no_improve = 0

    split_desc = "random_80_20_bootstrap" if use_random_split else "manifest_scene_split"
    print(f"Training samples: {len(train_dataset)} | val_seen: {len(val_seen_records)} | val_unseen: {len(val_unseen_records)}")
    print(
        f"Device: {device} | Stage: {args.stage.upper()} | Split: {split_desc} | "
        f"Aug: {augment_profile} | LR: {args.lr:.1e} | PosWeight: {args.pos_weight:.1f} | LambdaCoord: {args.lambda_coord:.2f}"
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_train_loss = 0.0

        for inputs, targets, gt_kpts, valid_mask, _meta in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            gt_kpts = gt_kpts.to(device)
            valid_mask = valid_mask.to(device)

            optimizer.zero_grad(set_to_none=True)
            heat_logits, vis_logits = model(inputs, return_aux=True)
            pred_xy = soft_argmax_2d(heat_logits)

            coord_pred = pred_xy
            coord_gt = gt_kpts
            if args.normalize_coords:
                scale = torch.tensor(
                    [heat_logits.shape[-1] - 1, heat_logits.shape[-2] - 1],
                    device=device,
                    dtype=torch.float32,
                )
                coord_pred = pred_xy / scale
                coord_gt = gt_kpts / scale

            heat_loss = masked_bce_with_logits(heat_logits, targets, valid_mask, args.pos_weight)
            coord_loss = masked_smooth_l1(coord_pred, coord_gt, valid_mask)
            aux_loss = F.binary_cross_entropy_with_logits(vis_logits, valid_mask.float())
            loss = heat_loss + args.lambda_coord * coord_loss + 0.05 * aux_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / max(1, len(train_loader))
        val_seen_metrics = evaluate_loader(
            model,
            val_seen_loader,
            device,
            pos_weight=args.pos_weight,
            lambda_coord=args.lambda_coord,
            normalize_coords=args.normalize_coords,
        ) or {"loss": avg_train_loss, "pck5": 0.0, "pck10": 0.0}

        val_unseen_metrics = None
        if val_unseen_loader and epoch % max(1, args.eval_unseen_every) == 0:
            val_unseen_metrics = evaluate_loader(
                model,
                val_unseen_loader,
                device,
                pos_weight=args.pos_weight,
                lambda_coord=args.lambda_coord,
                normalize_coords=args.normalize_coords,
            )

        scheduler.step(val_seen_metrics["loss"])
        current_lr = optimizer.param_groups[0]["lr"]
        message = (
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.6f} | "
            f"Val Seen Loss: {val_seen_metrics['loss']:.6f} | "
            f"Val Seen PCK@5: {val_seen_metrics['pck5']:.2f}% | "
            f"Val Seen PCK@10: {val_seen_metrics['pck10']:.2f}% | "
            f"LR: {current_lr:.1e}"
        )
        if val_unseen_metrics:
            message += (
                f" | Val Unseen Loss: {val_unseen_metrics['loss']:.6f}"
                f" | Val Unseen PCK@5: {val_unseen_metrics['pck5']:.2f}%"
                f" | Val Unseen PCK@10: {val_unseen_metrics['pck10']:.2f}%"
            )
        print(message)

        if val_seen_metrics["loss"] < best_val_loss:
            best_val_loss = val_seen_metrics["loss"]
            epochs_no_improve = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Validation improved. Saved checkpoint to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epochs.")

        if epochs_no_improve >= 7:
            print(f"Early stopping triggered at epoch {epoch}.")
            break


if __name__ == "__main__":
    main()
