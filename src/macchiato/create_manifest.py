"""Build deterministic Experiment B manifests and optional YOLO artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from macchiato.adapters.arducam_adapter import (
    build_manifest_records,
    prepare_yolo_dataset,
    summarize_records,
    write_manifests,
)
from macchiato.config import load_config, project_path


def build_from_config(
    config: dict[str, Any],
    project_root: Path,
    prepare_yolo: bool = False,
    force: bool = False,
) -> tuple[dict[str, int], dict[str, int] | None]:
    """Build manifests and, when requested, the generated YOLO dataset."""

    data = config["data"]
    expected_height, expected_width = (int(value) for value in data["frame_size"])
    records, diagnostics = build_manifest_records(
        raw_root=project_path(project_root, data["raw_root"]),
        pose_root=project_path(project_root, data["pose_root"]),
        project_root=project_root,
        expected_shape=(expected_height, expected_width),
        expected_keypoints=int(data["num_keypoints"]),
        gap_seconds=int(data["scene_gap_seconds"]),
        minimum_valid_keypoints=int(data["minimum_valid_keypoints"]),
        seed=int(config["seed"]),
        train_scene_count=int(data["train_scene_count"]),
        val_scene_count=int(data["val_scene_count"]),
    )
    write_manifests(records, project_path(project_root, data["manifests_dir"]))

    print(
        "Discovered: "
        f"depth={diagnostics['depth_files']} "
        f"confidence={diagnostics['confidence_files']} "
        f"pose={diagnostics['pose_files']}"
    )
    print(
        f"Complete numeric samples: {diagnostics['numeric_samples']} | "
        f"incomplete stems: {diagnostics['incomplete_stems']} | "
        f"scenes: {diagnostics['scenes']}"
    )
    print(
        f"Eligible: {diagnostics['eligible_samples']} | "
        f"rejected: {diagnostics['rejected_samples']}"
    )
    print(summarize_records(records))

    yolo_counts = None
    if prepare_yolo:
        yolo_counts = prepare_yolo_dataset(
            records=records,
            project_root=project_root,
            output_root=project_path(project_root, data["yolo_root"]),
            depth_maximum_mm=float(data["depth_max_mm"]),
            padding_fraction=0.10,
            force=force,
        )
        print(f"YOLO dataset: {yolo_counts}")
    return diagnostics, yolo_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/macchiato_finetune.yaml", help="Experiment YAML path.")
    parser.add_argument("--prepare-yolo", action="store_true", help="Also generate normalized PNGs and YOLO labels.")
    parser.add_argument("--force", action="store_true", help="Replace an existing generated YOLO dataset.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config, project_root = load_config(args.config)
    build_from_config(config, project_root, prepare_yolo=args.prepare_yolo, force=args.force)


if __name__ == "__main__":
    main()
