import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

MP_TO_COCO = [0, 2, 5, 7, 8, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
PASS_THROUGH_FIELDS = ["original_tof_frame", "scene_id", "session_id", "calibration_id"]


def convert_directory(input_dir: str, output_dir: str) -> List[str]:
    os.makedirs(output_dir, exist_ok=True)
    errors: List[str] = []
    for path in sorted(Path(input_dir).glob("*.json")):
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            errors.append(str(path))
            continue

        mp_pts = data.get("transformed_points")
        if mp_pts is None or len(mp_pts) != 33:
            print(f"Skipping {path.name}: missing transformed_points")
            continue

        coco_kpts = []
        for mp_idx in MP_TO_COCO:
            x, y = mp_pts[mp_idx]
            coco_kpts.append([x, y])

        output: Dict[str, object] = {"keypoints": coco_kpts}
        for field in PASS_THROUGH_FIELDS:
            if field in data:
                output[field] = data[field]

        out_path = Path(output_dir) / path.name
        with open(out_path, "w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"Wrote COCO17 labels for {path.name}")

    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert MediaPipe 33-point JSON files to COCO17 for v7.")
    parser.add_argument("--input-dir", type=str, default="./data/pose", help="Directory containing raw pose JSON files.")
    parser.add_argument("--output-dir", type=str, default="./data/pose_coco17", help="Output directory.")
    args = parser.parse_args()

    errors = convert_directory(args.input_dir, args.output_dir)
    print("Conversion finished.")
    if errors:
        print("Unreadable files:")
        for path in errors:
            print(path)


if __name__ == "__main__":
    main()
