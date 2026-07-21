"""Compare retrained U-Net and YOLO checkpoints on one held-out scene split."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from macchiato.adapters.arducam_adapter import COCO_KEYPOINT_NAMES, keypoint_valid_mask, load_keypoints
from macchiato.config import load_config, project_path
from macchiato.evaluation.pose_metrics import PoseMetricAccumulator
from macchiato.preprocessing.depth_normalization import normalize_confidence, normalize_depth


def select_highest_confidence_pose(result: Any) -> np.ndarray | None:
    """Return the COCO-17 coordinates for the highest-confidence person."""

    if result is None or result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None
    confidences = result.boxes.conf.detach().cpu().numpy()
    index = int(np.argmax(confidences))
    coordinates = result.keypoints.xy[index].detach().cpu().numpy().astype(np.float32)
    return coordinates if coordinates.shape == (17, 2) else None


def unpad_keypoints(keypoints: np.ndarray, pad_left: int, pad_top: int) -> np.ndarray:
    """Map padded U-Net coordinates back to the original depth frame."""

    output = np.asarray(keypoints, dtype=np.float32).copy()
    output[:, 0] -= pad_left
    output[:, 1] -= pad_top
    return output


def evaluate_rows(
    rows: list[dict[str, str]],
    project_root: Path,
    predictor: Callable[[dict[str, str]], np.ndarray | None],
) -> dict[str, object]:
    """Evaluate a model-neutral predictor over manifest rows."""

    accumulator = PoseMetricAccumulator()
    for row in rows:
        target = load_keypoints(project_root / row["pose_path"])
        valid_mask = keypoint_valid_mask(target, int(row["width"]), int(row["height"]))
        accumulator.update(predictor(row), target, valid_mask)
    summary = accumulator.summarize()
    summary["keypoint_names"] = COCO_KEYPOINT_NAMES
    return summary


def _read_split_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("eligible", "").lower() == "true"]
    if not rows:
        raise ValueError(f"No eligible samples in {path}")
    return rows


def _torch_device(requested: str):
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


def build_unet_predictor(config: dict[str, Any], project_root: Path, checkpoint: Path, device_name: str):
    """Load the retrained U-Net and return a raw-frame-coordinate predictor."""

    import torch

    from macchiato.losses.heatmap_loss import soft_argmax_2d
    from macchiato.models.macchiato_pose import MacchiatoPoseUNet

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    section, data = config["unet"], config["data"]
    device = _torch_device(device_name)
    model = MacchiatoPoseUNet(
        input_channels=int(section["input_channels"]),
        num_keypoints=int(section["num_keypoints"]),
        width_multiplier=float(section["width_multiplier"]),
    ).to(device)
    payload = torch.load(checkpoint, map_location=device)
    state = payload.get("model_state", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state, strict=True)
    model.eval()
    output_height, output_width = (int(value) for value in section["image_size"])

    def predict(row: dict[str, str]) -> np.ndarray:
        depth = np.load(project_root / row["depth_path"])
        confidence = np.load(project_root / row["confidence_path"])
        normalized_depth = normalize_depth(depth, float(data["depth_max_mm"]))
        normalized_confidence = normalize_confidence(confidence, float(data["confidence_max"]))
        channels = np.stack(
            [normalized_depth, normalized_confidence, normalized_depth * normalized_confidence], axis=0
        ).astype(np.float32)
        height, width = depth.shape
        pad_left = (output_width - width) // 2
        pad_right = output_width - width - pad_left
        pad_top = (output_height - height) // 2
        pad_bottom = output_height - height - pad_top
        channels = np.pad(channels, ((0, 0), (pad_top, pad_bottom), (pad_left, pad_right)))
        tensor = torch.from_numpy(channels).unsqueeze(0).to(device)
        with torch.inference_mode():
            predicted = soft_argmax_2d(model(tensor), float(section["softargmax_beta"]))[0].cpu().numpy()
        return unpad_keypoints(predicted, pad_left, pad_top)

    return predict


def build_yolo_predictor(
    config: dict[str, Any], project_root: Path, checkpoint: Path, split: str, device_name: str
):
    """Load a local YOLO checkpoint and return a highest-confidence predictor."""

    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("Ultralytics is required to evaluate a YOLO checkpoint") from exc
    model = YOLO(str(checkpoint))
    image_root = project_path(project_root, config["data"]["yolo_root"]) / "images" / split
    image_size = int(config["yolo"]["image_size"])

    def predict(row: dict[str, str]) -> np.ndarray | None:
        kwargs: dict[str, Any] = {"source": str(image_root / f"{row['timestamp']}.png"), "imgsz": image_size, "verbose": False}
        if device_name != "auto":
            kwargs["device"] = device_name
        results = model.predict(**kwargs)
        return select_highest_confidence_pose(results[0] if results else None)

    return predict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/macchiato_finetune.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--unet-checkpoint", default=None)
    parser.add_argument("--yolo-checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", default="experiments/macchiato_b/evaluation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.unet_checkpoint and not args.yolo_checkpoint:
        raise SystemExit("Provide --unet-checkpoint, --yolo-checkpoint, or both")
    config, project_root = load_config(args.config)
    manifest = project_path(project_root, config["data"]["manifests_dir"]) / f"{args.split}.csv"
    rows = _read_split_manifest(manifest)
    model_results: dict[str, object] = {}
    results: dict[str, object] = {"split": args.split, "manifest": str(manifest), "models": model_results}
    if args.unet_checkpoint:
        predictor = build_unet_predictor(
            config, project_root, project_path(project_root, args.unet_checkpoint), args.device
        )
        model_results["unet"] = evaluate_rows(rows, project_root, predictor)
    if args.yolo_checkpoint:
        predictor = build_yolo_predictor(
            config, project_root, project_path(project_root, args.yolo_checkpoint), args.split, args.device
        )
        model_results["yolo"] = evaluate_rows(rows, project_root, predictor)

    output_path = project_path(project_root, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results, indent=2))
    print(f"Saved evaluation to {output_path}")


if __name__ == "__main__":
    main()
