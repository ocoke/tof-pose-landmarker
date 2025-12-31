#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np


def _sort_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def load_confidence_paths(conf_dir: Path, limit: int | None):
    paths = sorted(conf_dir.glob("*.npy"), key=_sort_key)
    if limit is not None:
        paths = paths[:limit]
    return paths


def compute_stats(conf_paths, conf_thresh: float, saturation_value: float | None, saturation_tol: float):
    total_pixels = 0
    pixels_over_thresh = 0
    saturated_pixels = 0
    all_values = []

    for path in conf_paths:
        arr = np.load(path)
        arr = np.squeeze(arr)
        if arr.ndim != 2:
            raise ValueError(f"Expected 2D confidence map in {path}, got shape {arr.shape}")

        flat = arr.ravel()
        total_pixels += flat.size
        pixels_over_thresh += np.count_nonzero(flat >= conf_thresh)
        all_values.append(flat.astype(np.float64, copy=False))

        if saturation_value is None:
            frame_max = flat.max()
            saturated_pixels += np.count_nonzero(np.isclose(flat, frame_max, rtol=0.0, atol=saturation_tol))
        else:
            saturated_pixels += np.count_nonzero(flat >= saturation_value)

    if total_pixels == 0 or not all_values:
        raise ValueError("No confidence values found to analyze.")

    stacked = np.concatenate(all_values)
    return {
        "frames": len(conf_paths),
        "pixels": total_pixels,
        "pct_over_thresh": (pixels_over_thresh / total_pixels) * 100.0,
        "median": float(np.median(stacked)),
        "percentile_10": float(np.percentile(stacked, 10)),
        "saturation_rate": (saturated_pixels / total_pixels) * 100.0,
        "saturation_mode": "per-frame max" if saturation_value is None else f">= {saturation_value}",
    }


def main():
    parser = argparse.ArgumentParser(description="Report quick stats over ToF confidence maps.")
    parser.add_argument("--conf-dir", type=Path, default=Path("data/confidence"),
                        help="Directory containing confidence .npy files.")
    parser.add_argument("--conf-thresh", type=float, default=200.0,
                        help="Threshold used for the %% pixels >= conf_thresh metric.")
    parser.add_argument("--num-frames", type=int, default=984,
                        help="How many frames to analyze after numeric sorting. Use -1 for all.")
    parser.add_argument("--saturation-value", type=float, default=None,
                        help="Value treated as saturation. If omitted, counts pixels equal to each frame's max.")
    parser.add_argument("--saturation-tol", type=float, default=1e-3,
                        help="Tolerance for matching per-frame maxima when inferring saturation.")
    args = parser.parse_args()

    frame_limit = None if args.num_frames is None or args.num_frames < 0 else args.num_frames
    conf_paths = load_confidence_paths(args.conf_dir, frame_limit)
    if not conf_paths:
        raise SystemExit(f"No .npy files found in {args.conf_dir}")

    stats = compute_stats(conf_paths, args.conf_thresh, args.saturation_value, args.saturation_tol)

    print(f"Frames analyzed: {stats['frames']} (numeric sort, limit={frame_limit or 'all'})")
    print(f"Total pixels: {stats['pixels']:,}")
    print(f"Confidence threshold: {args.conf_thresh}")
    print(f"Pixels >= threshold: {stats['pct_over_thresh']:.2f}%")
    print(f"Median confidence: {stats['median']:.3f}")
    print(f"10th percentile confidence: {stats['percentile_10']:.3f}")
    print(f"Saturated pixel rate: {stats['saturation_rate']:.4f}% ({stats['saturation_mode']})")


if __name__ == "__main__":
    main()
