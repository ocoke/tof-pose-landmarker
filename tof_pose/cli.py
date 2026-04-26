from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TextIO

import numpy as np

from .camera import ArducamCameraAdapter, CameraConfig
from .geometry import floor_candidate_masks
from .inference import HeatmapOffsetPoseEstimator, MoveNetEstimator
from .model_store import DEFAULT_MODELS_DIR, download_model, resolve_model_path
from .pipeline import HybridToFPosePipeline, PipelineConfig
from .visualization import PreviewWindow


def _build_camera(args: argparse.Namespace) -> ArducamCameraAdapter:
    config = CameraConfig(
        connection=args.connection,
        index=args.index,
        range_mode_m=args.range,
        request_timeout_ms=args.timeout_ms,
        rotate=args.rotate,
        depth_unit=args.depth_unit,
    )
    return ArducamCameraAdapter(config=config)


def _build_pipeline_config(args: argparse.Namespace) -> PipelineConfig:
    config = PipelineConfig()
    geometry = config.geometry
    tuning_map = {
        "confidence_threshold": "confidence_threshold",
        "background_threshold_m": "background_threshold_m",
        "fallback_depth_window_m": "fallback_depth_window_m",
        "track_max_depth_jump_m": "track_max_depth_jump_m",
        "roi_margin_px": "roi_margin_px",
        "track_size_step_fraction": "track_max_size_step_fraction",
    }
    for arg_name, field_name in tuning_map.items():
        value = getattr(args, arg_name, None)
        if value is not None:
            setattr(geometry, field_name, value)
    return config


def _common_parser(name: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=f"tof-pose {name}")
    parser.add_argument("--connection", default="CSI")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--range", type=int, default=4)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--rotate", type=int, choices=(0, 180), default=0)
    parser.add_argument("--depth-unit", choices=("auto", "m", "mm"), default="auto")
    parser.add_argument("--floor-plane", type=Path)
    parser.add_argument("--frames", type=int, default=0)
    return parser


def _add_tracker_tuning_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--background-threshold-m", type=float)
    parser.add_argument("--fallback-depth-window-m", type=float)
    parser.add_argument("--track-max-depth-jump-m", type=float)
    parser.add_argument("--roi-margin-px", type=int)
    parser.add_argument("--track-size-step-fraction", type=float)


def _add_debug_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--debug-jsonl", type=Path)
    parser.add_argument("--debug-every", type=int, default=1)


def inspect_camera(args: argparse.Namespace) -> int:
    camera = _build_camera(args)
    payload = camera.inspect()
    print(json.dumps(payload, indent=2))
    return 0


def calibrate_floor(args: argparse.Namespace) -> int:
    camera = _build_camera(args)
    frames_to_collect = max(args.frames, 30)
    config = PipelineConfig()
    pipeline = HybridToFPosePipeline(camera=None, pose_estimator=_NoOpEstimator(), config=config)
    preview_window = PreviewWindow(title="ToF Floor Calibration") if args.preview else None
    frames = []
    try:
        with camera:
            for frame_index in range(1, frames_to_collect + 1):
                frame = camera.read()
                frames.append(frame)
                if preview_window is not None:
                    masks = floor_candidate_masks(frame, config.geometry)
                    if not preview_window.show_calibration(frame, masks, frame_index=frame_index):
                        break
    finally:
        if preview_window is not None:
            preview_window.close()
    plane, diagnostics = pipeline.calibrate_floor(frames, with_diagnostics=True)
    if plane is None:
        print("Floor calibration failed: not enough usable floor points for plane fitting.", file=sys.stderr)
        print(json.dumps(diagnostics.to_json(), indent=2), file=sys.stderr)
        print(
            "Likely causes: the floor is not visible, a person/object is occupying the lower field of view, "
            "or the camera confidence values are lower than the runtime threshold.",
            file=sys.stderr,
        )
        return 1
    output = args.output or Path("floor_plane.json")
    output.write_text(json.dumps(plane.to_json(), indent=2))
    print(f"Saved floor plane to {output}")
    print(json.dumps({"diagnostics": diagnostics.to_json()}, indent=2))
    return 0


def fetch_models(args: argparse.Namespace) -> int:
    aliases = args.models or ["movenet_lightning_int8"]
    for alias in aliases:
        path = download_model(alias, models_dir=args.models_dir, force=args.force)
        print(json.dumps({"model": alias, "path": str(path)}))
    return 0


def run_demo(args: argparse.Namespace) -> int:
    camera = _build_camera(args)
    model_path = args.movenet_model or resolve_model_path("movenet_lightning_int8", models_dir=args.models_dir)
    estimator = MoveNetEstimator(model_path=str(model_path), num_threads=args.threads)
    pipeline = HybridToFPosePipeline(camera=camera, pose_estimator=estimator, config=_build_pipeline_config(args))
    if args.floor_plane:
        pipeline.load_floor_plane(args.floor_plane)
    return _run_loop(
        camera,
        pipeline,
        frames=args.frames,
        preview=args.preview,
        debug_jsonl=args.debug_jsonl,
        debug_every=args.debug_every,
    )


def run_production(args: argparse.Namespace) -> int:
    camera = _build_camera(args)
    estimator = HeatmapOffsetPoseEstimator(model_path=str(args.pose_model), num_threads=args.threads)
    pipeline = HybridToFPosePipeline(camera=camera, pose_estimator=estimator, config=_build_pipeline_config(args))
    if args.floor_plane:
        pipeline.load_floor_plane(args.floor_plane)
    return _run_loop(
        camera,
        pipeline,
        frames=args.frames,
        preview=args.preview,
        debug_jsonl=args.debug_jsonl,
        debug_every=args.debug_every,
    )


def _debug_payload(frame_index: int, result: dict | None, diagnostics, fps: float) -> dict[str, object]:
    payload: dict[str, object] = {
        "frame": frame_index,
        "fps": round(fps, 3),
        "tracking_diagnostics": diagnostics.to_json() if diagnostics is not None else None,
    }
    if result is None:
        payload["track"] = None
        payload["valid_joints"] = 0
        return payload

    track = result["track"]
    pose3d = result["pose3d"]
    payload["track"] = {
        "cluster_id": track.cluster_id,
        "quality": round(track.quality, 3),
        "roi_px": list(track.roi_px),
        "centroid_xyz": np.round(track.centroid_xyz, 4).tolist(),
        "extent_xyz": np.round(track.extent_xyz, 4).tolist(),
    }
    payload["valid_joints"] = int(np.count_nonzero(pose3d.valid))
    return payload


def _write_debug_payload(handle: TextIO, payload: dict[str, object]) -> None:
    handle.write(json.dumps(payload) + "\n")
    handle.flush()


def _run_loop(
    camera: ArducamCameraAdapter,
    pipeline: HybridToFPosePipeline,
    frames: int,
    preview: bool = False,
    debug_jsonl: Path | None = None,
    debug_every: int = 1,
) -> int:
    processed = 0
    start = time.perf_counter()
    preview_window = PreviewWindow() if preview else None
    debug_file = debug_jsonl.open("w") if debug_jsonl is not None else None
    debug_every = max(int(debug_every), 1)
    try:
        with camera:
            while frames <= 0 or processed < frames:
                frame = camera.read()
                result = pipeline.process_frame(frame)
                processed += 1
                fps = processed / max(time.perf_counter() - start, 1e-6)
                payload = {"frame": processed}
                if result is not None:
                    track = result["track"]
                    pose3d = result["pose3d"]
                    payload.update(
                        {
                            "cluster_id": track.cluster_id,
                            "quality": round(track.quality, 3),
                            "roi_px": list(track.roi_px),
                            "centroid_xyz": np.round(track.centroid_xyz, 4).tolist(),
                            "valid_joints": int(np.count_nonzero(pose3d.valid)),
                        }
                    )
                print(json.dumps(payload))
                if debug_file is not None and processed % debug_every == 0:
                    _write_debug_payload(
                        debug_file,
                        _debug_payload(processed, result, pipeline.tracker.last_diagnostics, fps),
                    )
                if preview_window is not None:
                    if not preview_window.show(frame, result, fps=fps, frame_index=processed):
                        break
    finally:
        if preview_window is not None:
            preview_window.close()
        if debug_file is not None:
            debug_file.close()
    elapsed = max(time.perf_counter() - start, 1e-6)
    fps = processed / elapsed
    print(json.dumps({"processed_frames": processed, "elapsed_s": round(elapsed, 3), "fps": round(fps, 2)}))
    return 0


class _NoOpEstimator:
    def predict(self, roi_tensor: np.ndarray, roi_px: tuple[int, int, int, int]):
        raise RuntimeError("No pose estimator is attached")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tof-pose")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = _common_parser("inspect-camera")
    inspect_parser.set_defaults(func=inspect_camera)
    subparsers.add_parser("inspect-camera", parents=[inspect_parser], add_help=False)

    calibrate_parser = _common_parser("calibrate-floor")
    calibrate_parser.add_argument("--output", type=Path, default=Path("floor_plane.json"))
    calibrate_parser.add_argument("--preview", action="store_true")
    calibrate_parser.set_defaults(func=calibrate_floor)
    subparsers.add_parser("calibrate-floor", parents=[calibrate_parser], add_help=False)

    fetch_parser = argparse.ArgumentParser(prog="tof-pose fetch-models")
    fetch_parser.add_argument("--models", nargs="+", default=["movenet_lightning_int8"])
    fetch_parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    fetch_parser.add_argument("--force", action="store_true")
    fetch_parser.set_defaults(func=fetch_models)
    subparsers.add_parser("fetch-models", parents=[fetch_parser], add_help=False)

    demo_parser = _common_parser("run-demo")
    demo_parser.add_argument("--movenet-model", type=Path)
    demo_parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    demo_parser.add_argument("--threads", type=int, default=4)
    demo_parser.add_argument("--preview", action="store_true")
    _add_tracker_tuning_args(demo_parser)
    _add_debug_args(demo_parser)
    demo_parser.set_defaults(func=run_demo)
    subparsers.add_parser("run-demo", parents=[demo_parser], add_help=False)

    prod_parser = _common_parser("run-production")
    prod_parser.add_argument("--pose-model", type=Path, required=True)
    prod_parser.add_argument("--threads", type=int, default=4)
    prod_parser.add_argument("--preview", action="store_true")
    _add_tracker_tuning_args(prod_parser)
    _add_debug_args(prod_parser)
    prod_parser.set_defaults(func=run_production)
    subparsers.add_parser("run-production", parents=[prod_parser], add_help=False)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
