"""Guarded training entry point for Experiment B YOLO and U-Net models."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from macchiato.adapters.arducam_adapter import validate_yolo_dataset
from macchiato.config import load_config, project_path


def validate_manifests(config: dict[str, Any], project_root: Path) -> dict[str, int]:
    """Validate split manifests and return their eligible row counts."""

    manifests_dir = project_path(project_root, config["data"]["manifests_dir"])
    counts = {}
    scenes: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = manifests_dir / f"{split}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing {split} manifest: {path}")
        with path.open("r", newline="", encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row.get("eligible", "").lower() == "true"]
        if not rows:
            raise ValueError(f"No eligible samples in {path}")
        if any(row["split"] != split for row in rows):
            raise ValueError(f"Manifest contains a row assigned to the wrong split: {path}")
        counts[split] = len(rows)
        scenes[split] = {row["scene_id"] for row in rows}
    if scenes["train"] & scenes["val"] or scenes["train"] & scenes["test"] or scenes["val"] & scenes["test"]:
        raise ValueError("Scene leakage detected between manifests")
    return counts


def resolve_yolo_run(config: dict[str, Any], project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve and validate Ultralytics training arguments without importing it."""

    section = config["yolo"]
    dataset_yaml = project_path(project_root, section["dataset_yaml"])
    if not dataset_yaml.exists():
        raise FileNotFoundError(f"Missing generated YOLO dataset config: {dataset_yaml}")
    with dataset_yaml.open("r", encoding="utf-8") as handle:
        dataset_config = yaml.safe_load(handle)
    yolo_root = project_path(project_root, dataset_config["path"])
    validate_yolo_dataset(yolo_root)

    kwargs: dict[str, Any] = {
        "data": str(dataset_yaml),
        "epochs": int(section["epochs"]),
        "imgsz": int(section["image_size"]),
        "batch": int(args.batch_size or section["batch_size"]),
        "patience": int(section["patience"]),
        "workers": int(section["workers"]),
        "seed": int(config["seed"]),
        "deterministic": bool(section["deterministic"]),
        "project": str(project_path(project_root, section["project"])),
        "name": str(section["run_name"]),
        **section["augmentations"],
    }
    device = args.device or section.get("device")
    if device and device != "auto":
        kwargs["device"] = device
    return {"model": section["model"], "train_kwargs": kwargs}


def resolve_unet_run(config: dict[str, Any], project_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the canonical U-Net stages and paths without importing torch."""

    section = config["unet"]
    manifests = project_path(project_root, config["data"]["manifests_dir"])
    checkpoint_dir = project_path(project_root, section["checkpoint_dir"])
    stages = ["a", "b"] if args.stage == "all" else [args.stage]
    return {
        "stages": stages,
        "train_manifest": str(manifests / "train.csv"),
        "val_manifest": str(manifests / "val.csv"),
        "checkpoint_dir": str(checkpoint_dir),
        "stage_a_checkpoint": str(checkpoint_dir / "stage_a_best.pth"),
        "stage_b_checkpoint": str(checkpoint_dir / "stage_b_best.pth"),
        "resume": str(project_path(project_root, args.resume)) if args.resume else None,
        "batch_size": int(args.batch_size or section["batch_size"]),
        "device": args.device or "auto",
        "stage_settings": {stage: section[f"stage_{stage}"] for stage in stages},
    }


def _select_torch_device(requested: str):
    import torch

    if requested != "auto":
        if requested.isdigit():
            requested = f"cuda:{requested}"
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_model_state(model, checkpoint_path: Path, device) -> None:
    import torch

    payload = torch.load(checkpoint_path, map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise ValueError(
            f"Checkpoint is incompatible. Missing={result.missing_keys}, unexpected={result.unexpected_keys}"
        )


def _evaluate_unet(model, loader, device, settings: dict[str, Any], beta: float) -> dict[str, float]:
    import torch
    import torch.nn.functional as functional

    from macchiato.losses.heatmap_loss import masked_coordinate_loss, masked_heatmap_bce, soft_argmax_2d

    model.eval()
    totals = {"loss": 0.0, "pck5": 0.0, "pck10": 0.0, "samples": 0}
    with torch.no_grad():
        for inputs, targets, ground_truth, valid_mask, _metadata in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            ground_truth, valid_mask = ground_truth.to(device), valid_mask.to(device)
            heatmaps, visibility = model(inputs, return_aux=True)
            predicted = soft_argmax_2d(heatmaps, beta)
            loss = (
                masked_heatmap_bce(heatmaps, targets, valid_mask, float(settings["positive_weight"]))
                + float(settings["coordinate_loss_weight"])
                * masked_coordinate_loss(predicted, ground_truth, valid_mask)
                + float(settings["visibility_loss_weight"])
                * functional.binary_cross_entropy_with_logits(visibility, valid_mask.float())
            )
            distances = torch.linalg.vector_norm(predicted - ground_truth, dim=-1)
            weights = valid_mask.float()
            denominator = torch.clamp(weights.sum(dim=1), min=1.0)
            totals["loss"] += float(loss.item())
            totals["pck5"] += float((((distances <= 12.0).float() * weights).sum(dim=1) / denominator).sum().item())
            totals["pck10"] += float((((distances <= 24.0).float() * weights).sum(dim=1) / denominator).sum().item())
            totals["samples"] += inputs.shape[0]
    return {
        "loss": totals["loss"] / max(1, len(loader)),
        "pck5": 100.0 * totals["pck5"] / max(1, totals["samples"]),
        "pck10": 100.0 * totals["pck10"] / max(1, totals["samples"]),
    }


def _run_unet_stage(
    config: dict[str, Any],
    project_root: Path,
    resolved: dict[str, Any],
    stage: str,
    resume_path: Path | None,
) -> Path:
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader

    from macchiato.datasets.unified_dataset import UnifiedPoseDataset
    from macchiato.losses.heatmap_loss import masked_coordinate_loss, masked_heatmap_bce, soft_argmax_2d
    from macchiato.models.macchiato_pose import MacchiatoPoseUNet

    data_config, model_config = config["data"], config["unet"]
    settings = model_config[f"stage_{stage}"]
    output_shape = (int(model_config["image_size"][0]), int(model_config["image_size"][1]))
    train_dataset = UnifiedPoseDataset(
        resolved["train_manifest"],
        project_root=project_root,
        output_shape=output_shape,
        depth_maximum_mm=float(data_config["depth_max_mm"]),
        confidence_maximum=float(data_config["confidence_max"]),
        heatmap_sigma=float(model_config["heatmap_sigma"]),
        augment=True,
    )
    val_dataset = UnifiedPoseDataset(
        resolved["val_manifest"],
        project_root=project_root,
        output_shape=output_shape,
        depth_maximum_mm=float(data_config["depth_max_mm"]),
        confidence_maximum=float(data_config["confidence_max"]),
        heatmap_sigma=float(model_config["heatmap_sigma"]),
        augment=False,
    )
    device = _select_torch_device(resolved["device"])
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=resolved["batch_size"],
        shuffle=True,
        num_workers=int(model_config["workers"]),
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=resolved["batch_size"],
        shuffle=False,
        num_workers=max(1, int(model_config["workers"]) // 2),
        pin_memory=pin_memory,
    )

    model = MacchiatoPoseUNet(
        input_channels=int(model_config["input_channels"]),
        num_keypoints=int(model_config["num_keypoints"]),
        width_multiplier=float(model_config["width_multiplier"]),
    ).to(device)
    if resume_path:
        if not resume_path.exists():
            raise FileNotFoundError(f"Missing U-Net resume checkpoint: {resume_path}")
        _load_model_state(model, resume_path, device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(settings["learning_rate"]),
        weight_decay=float(model_config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=int(model_config["lr_scheduler_patience"]), factor=0.1
    )
    checkpoint_dir = Path(resolved["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"stage_{stage}_best.pth"
    history_dir = project_path(project_root, model_config["experiment_dir"])
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / f"stage_{stage}_history.csv"

    best_loss = float("inf")
    stale_epochs = 0
    history_rows: list[dict[str, float | int]] = []
    beta = float(model_config["softargmax_beta"])
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        running_loss = 0.0
        for inputs, targets, ground_truth, valid_mask, _metadata in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            ground_truth, valid_mask = ground_truth.to(device), valid_mask.to(device)
            optimizer.zero_grad(set_to_none=True)
            heatmaps, visibility = model(inputs, return_aux=True)
            predicted = soft_argmax_2d(heatmaps, beta)
            loss = (
                masked_heatmap_bce(heatmaps, targets, valid_mask, float(settings["positive_weight"]))
                + float(settings["coordinate_loss_weight"])
                * masked_coordinate_loss(predicted, ground_truth, valid_mask)
                + float(settings["visibility_loss_weight"])
                * functional.binary_cross_entropy_with_logits(visibility, valid_mask.float())
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += float(loss.item())

        metrics = _evaluate_unet(model, val_loader, device, settings, beta)
        scheduler.step(metrics["loss"])
        row = {
            "epoch": epoch,
            "train_loss": running_loss / max(1, len(train_loader)),
            "val_loss": metrics["loss"],
            "val_pck5": metrics["pck5"],
            "val_pck10": metrics["pck10"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(row)
        with history_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history_rows)
        print(
            f"Stage {stage.upper()} epoch {epoch}/{settings['epochs']} | "
            f"train={row['train_loss']:.6f} val={row['val_loss']:.6f} "
            f"PCK@5={row['val_pck5']:.2f}% PCK@10={row['val_pck10']:.2f}%"
        )
        if metrics["loss"] < best_loss:
            best_loss = metrics["loss"]
            stale_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "stage": stage, "epoch": epoch, "config": config}, checkpoint_path
            )
        else:
            stale_epochs += 1
            if stale_epochs >= int(model_config["early_stopping_patience"]):
                print(f"Early stopping Stage {stage.upper()} at epoch {epoch}")
                break
    return checkpoint_path


def run_unet(config: dict[str, Any], project_root: Path, resolved: dict[str, Any]) -> None:
    """Run selected canonical U-Net stages."""

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    previous = Path(resolved["resume"]).expanduser() if resolved["resume"] else None
    for stage in resolved["stages"]:
        if stage == "b" and previous is None:
            previous = Path(resolved["stage_a_checkpoint"])
        previous = _run_unet_stage(config, project_root, resolved, stage, previous)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_type", choices=["yolo", "unet"], help="Model pipeline to validate or train.")
    parser.add_argument("--config", default="configs/macchiato_finetune.yaml", help="Experiment YAML path.")
    parser.add_argument("--run", action="store_true", help="Explicitly start training after validation.")
    parser.add_argument("--device", default=None, help="Override device, e.g. cpu, mps, 0, or cuda.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override configured batch size.")
    parser.add_argument("--stage", choices=["all", "a", "b"], default="all", help="U-Net stage selection.")
    parser.add_argument("--resume", default=None, help="Optional U-Net checkpoint used to initialize the first stage.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    counts = validate_manifests(config, project_root)
    if args.model_type == "yolo":
        resolved = resolve_yolo_run(config, project_root, args)
    else:
        resolved = resolve_unet_run(config, project_root, args)
    print(f"Validated scene-split manifests: {counts}")
    print(json.dumps(resolved, indent=2, default=str))
    if not args.run:
        print("Dry run only. Add --run to start training.")
        return

    if args.model_type == "yolo":
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError as exc:
            raise RuntimeError("Install project training dependencies before running YOLO training") from exc
        model = YOLO(resolved["model"])
        model.train(**resolved["train_kwargs"])
    else:
        run_unet(config, project_root, resolved)


if __name__ == "__main__":
    main()
