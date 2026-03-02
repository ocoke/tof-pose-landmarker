import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def infer_scenes(sample_ids: List[str], gap_seconds: int) -> Tuple[Dict[str, str], List[Tuple[str, int, int, int]]]:
    numeric = []
    fallback = {}
    for sample_id in sorted(set(sample_ids)):
        try:
            numeric.append((int(sample_id), sample_id))
        except ValueError:
            fallback[sample_id] = f"scene_misc_{sample_id}"

    numeric.sort()
    scene_map: Dict[str, str] = {}
    scene_ranges: List[Tuple[str, int, int, int]] = []
    if not numeric:
        return fallback, scene_ranges

    scene_idx = 1
    start_ts, _ = numeric[0]
    prev_ts = start_ts
    count = 0
    current_scene = f"scene_{scene_idx:04d}"

    for ts, sample_id in numeric:
        if count > 0 and (ts - prev_ts) > gap_seconds:
            scene_ranges.append((current_scene, start_ts, prev_ts, count))
            scene_idx += 1
            current_scene = f"scene_{scene_idx:04d}"
            start_ts = ts
            count = 0
        scene_map[sample_id] = current_scene
        prev_ts = ts
        count += 1

    scene_ranges.append((current_scene, start_ts, prev_ts, count))
    scene_map.update(fallback)
    return scene_map, scene_ranges


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a scene_map.json by splitting timestamped samples on large time gaps.")
    parser.add_argument("--data", type=str, default="./data", help="Dataset root containing depth/*.npy")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path (defaults to <data>/scene_map.json)")
    parser.add_argument("--scene-gap-seconds", type=int, default=900, help="Start a new scene when timestamp gap exceeds this value")
    args = parser.parse_args()

    depth_dir = Path(args.data) / "depth"
    sample_ids = sorted(path.stem for path in depth_dir.glob("*.npy"))
    scene_map, scene_ranges = infer_scenes(sample_ids, gap_seconds=args.scene_gap_seconds)

    output_path = Path(args.output) if args.output else Path(args.data) / "scene_map.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(scene_map, handle, indent=2, sort_keys=True)

    print(f"Wrote {len(scene_map)} scene assignments to {output_path}")
    print(f"Gap threshold: {args.scene_gap_seconds}s")
    print(f"Inferred scenes: {len(scene_ranges)}")
    for scene_id, start_ts, end_ts, count in scene_ranges:
        print(f"{scene_id}: {start_ts} -> {end_ts} ({count} samples)")


if __name__ == "__main__":
    main()
