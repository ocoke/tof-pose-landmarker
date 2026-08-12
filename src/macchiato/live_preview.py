"""Live YOLO26 pose preview for an Arducam Time-of-Flight depth stream."""

from __future__ import annotations

import argparse
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from macchiato.config import load_config, project_path
from macchiato.preprocessing.depth_normalization import depth_to_rgb_uint8

COCO17_EDGES = (
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
)


@dataclass(frozen=True)
class PosePrediction:
    """One highest-confidence COCO-17 person prediction."""

    keypoints: np.ndarray
    keypoint_confidences: np.ndarray
    box_xyxy: np.ndarray
    box_confidence: float


@dataclass
class RollingLatency:
    """Fixed-size latency window used by the live display."""

    window_size: int = 120
    values_ms: deque[float] = field(init=False)

    def __post_init__(self) -> None:
        if self.window_size < 1:
            raise ValueError("window_size must be positive")
        self.values_ms = deque(maxlen=self.window_size)

    def update(self, elapsed_seconds: float) -> None:
        elapsed_ms = float(elapsed_seconds) * 1000.0
        if np.isfinite(elapsed_ms) and elapsed_ms >= 0:
            self.values_ms.append(elapsed_ms)

    def summarize(self) -> dict[str, float | int | None]:
        if not self.values_ms:
            return {"samples": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None, "fps": None}
        values = np.asarray(self.values_ms, dtype=np.float64)
        mean_ms = float(np.mean(values))
        return {
            "samples": len(values),
            "mean_ms": mean_ms,
            "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95)),
            "fps": 1000.0 / mean_ms if mean_ms > 0 else None,
        }


def _as_numpy(value: Any) -> np.ndarray:
    """Convert a Torch-like tensor or array to a NumPy array."""

    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def extract_highest_confidence_prediction(result: Any) -> PosePrediction | None:
    """Extract the highest-confidence person from one Ultralytics result."""

    if result is None or result.boxes is None or result.keypoints is None or len(result.boxes) == 0:
        return None

    box_confidences = _as_numpy(result.boxes.conf).reshape(-1)
    index = int(np.argmax(box_confidences))
    keypoints = _as_numpy(result.keypoints.xy[index]).astype(np.float32)
    boxes = _as_numpy(result.boxes.xyxy)
    if keypoints.shape != (17, 2) or boxes.ndim != 2 or boxes.shape[1] != 4:
        return None

    raw_keypoint_confidences = getattr(result.keypoints, "conf", None)
    if raw_keypoint_confidences is None:
        keypoint_confidences = np.ones(17, dtype=np.float32)
    else:
        keypoint_confidences = _as_numpy(raw_keypoint_confidences[index]).astype(np.float32).reshape(-1)
        if keypoint_confidences.shape != (17,):
            return None

    return PosePrediction(
        keypoints=keypoints,
        keypoint_confidences=keypoint_confidences,
        box_xyxy=boxes[index].astype(np.float32),
        box_confidence=float(box_confidences[index]),
    )


def draw_pose_overlay(
    image: np.ndarray,
    prediction: PosePrediction,
    keypoint_threshold: float = 0.25,
) -> np.ndarray:
    """Draw a COCO-17 skeleton and person box on a BGR depth visualization."""

    output = np.asarray(image).copy()
    height, width = output.shape[:2]
    valid = (
        np.isfinite(prediction.keypoints).all(axis=1)
        & (prediction.keypoint_confidences >= keypoint_threshold)
        & (prediction.keypoints[:, 0] >= 0)
        & (prediction.keypoints[:, 0] < width)
        & (prediction.keypoints[:, 1] >= 0)
        & (prediction.keypoints[:, 1] < height)
    )

    for start, end in COCO17_EDGES:
        if valid[start] and valid[end]:
            point_a = tuple(np.rint(prediction.keypoints[start]).astype(int))
            point_b = tuple(np.rint(prediction.keypoints[end]).astype(int))
            cv2.line(output, point_a, point_b, (0, 220, 255), 2, cv2.LINE_AA)

    for index, (x_coord, y_coord) in enumerate(prediction.keypoints):
        if valid[index]:
            cv2.circle(output, (int(round(x_coord)), int(round(y_coord))), 3, (0, 255, 0), -1, cv2.LINE_AA)

    x_min, y_min, x_max, y_max = np.rint(prediction.box_xyxy).astype(int)
    x_min, x_max = int(np.clip(x_min, 0, width - 1)), int(np.clip(x_max, 0, width - 1))
    y_min, y_max = int(np.clip(y_min, 0, height - 1)), int(np.clip(y_max, 0, height - 1))
    cv2.rectangle(output, (x_min, y_min), (x_max, y_max), (255, 120, 0), 1, cv2.LINE_AA)
    cv2.putText(
        output,
        f"person {prediction.box_confidence:.2f}",
        (x_min, max(12, y_min - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (255, 180, 40),
        1,
        cv2.LINE_AA,
    )
    return output


class ArducamDepthStream:
    """Small lifecycle-safe wrapper around the Arducam ToF SDK."""

    def __init__(self, range_mm: float, camera_index: int = 0, rotate_180: bool = True):
        try:
            import ArducamDepthCamera as ac
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "ArducamDepthCamera is required for live preview; install the Arducam ToF SDK on the Raspberry Pi"
            ) from exc
        self.ac = ac
        self.range_mm = float(range_mm)
        self.camera_index = int(camera_index)
        self.rotate_180 = bool(rotate_180)
        self.camera = ac.ArducamCamera()
        self.started = False

    def open(self) -> None:
        result = self.camera.open(self.ac.Connection.CSI, self.camera_index)
        if result != 0:
            self.camera.close()
            raise RuntimeError(f"Failed to open Arducam CSI camera (error {result})")
        try:
            self.camera.setControl(self.ac.Control.RANGE, int(self.range_mm))
            result = self.camera.start(self.ac.FrameType.DEPTH)
            if result != 0:
                raise RuntimeError(f"Failed to start Arducam depth stream (error {result})")
            self.started = True
        except Exception:
            self.camera.close()
            raise

    def read(self, timeout_ms: int = 200) -> np.ndarray | None:
        frame = self.camera.requestFrame(int(timeout_ms))
        if frame is None:
            return None
        try:
            if not isinstance(frame, self.ac.DepthData):
                return None
            depth = frame.depth_data
            if depth is None or depth.size == 0:
                return None
            output = np.array(depth, copy=True)
        finally:
            self.camera.releaseFrame(frame)
        if self.rotate_180:
            output = cv2.rotate(output, cv2.ROTATE_180)
        return output

    def close(self) -> None:
        try:
            if self.started:
                self.camera.stop()
        finally:
            self.started = False
            self.camera.close()


def _prediction_kwargs(image_size: int, confidence: float, device: str) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "imgsz": image_size,
        "conf": confidence,
        "max_det": 1,
        "save": False,
        "verbose": False,
    }
    if device != "auto":
        kwargs["device"] = device
    return kwargs


def _draw_runtime_status(
    image: np.ndarray,
    result: Any,
    pipeline: dict[str, float | int | None],
) -> None:
    speed = getattr(result, "speed", {}) if result is not None else {}
    inference_ms = speed.get("inference")
    fps = pipeline["fps"]
    p50 = pipeline["p50_ms"]
    p95 = pipeline["p95_ms"]
    line_one = "Live FPS: warming up" if fps is None else f"Live FPS: {fps:.1f}"
    line_two = "Inference: n/a" if inference_ms is None else f"Inference: {float(inference_ms):.1f} ms"
    line_three = "Pipeline: warming up" if p50 is None else f"Pipeline p50/p95: {p50:.1f}/{p95:.1f} ms"
    for y_coord, text in ((16, line_one), (32, line_two), (48, line_three)):
        cv2.putText(image, text, (5, y_coord), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (40, 40, 255), 1, cv2.LINE_AA)


def run_live_preview(args: argparse.Namespace) -> dict[str, object]:
    """Run the camera/model/display loop and return final rolling timing statistics."""

    config, project_root = load_config(args.config)
    checkpoint = project_path(project_root, args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(checkpoint)

    try:
        import torch
        from ultralytics import YOLO
    except ModuleNotFoundError as exc:
        raise RuntimeError("PyTorch and Ultralytics are required for live preview") from exc

    if args.threads is not None:
        torch.set_num_threads(args.threads)

    image_size = int(config["yolo"]["image_size"])
    frame_height, frame_width = (int(value) for value in config["data"]["frame_size"])
    maximum_mm = float(args.range_mm or config["data"]["depth_max_mm"])
    model = YOLO(str(checkpoint))
    predict_kwargs = _prediction_kwargs(image_size, args.confidence, args.device)
    dummy = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    for _ in range(args.warmup_frames):
        model.predict(source=dummy, **predict_kwargs)

    stream = ArducamDepthStream(maximum_mm, args.camera_index, not args.no_rotate)
    pipeline_latency = RollingLatency(args.latency_window)
    predict_latency = RollingLatency(args.latency_window)
    frame_count = 0
    stream.open()

    try:
        cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
        while args.max_frames <= 0 or frame_count < args.max_frames:
            frame_started = time.perf_counter()
            depth = stream.read(args.timeout_ms)
            if depth is None:
                if cv2.waitKey(1) & 0xFF in {ord("q"), 27}:
                    break
                continue
            depth_image = depth_to_rgb_uint8(depth, maximum_mm)

            predict_started = time.perf_counter()
            results = model.predict(source=depth_image, **predict_kwargs)
            predict_latency.update(time.perf_counter() - predict_started)
            result = results[0] if results else None
            prediction = extract_highest_confidence_prediction(result)
            display = depth_image if prediction is None else draw_pose_overlay(
                depth_image, prediction, args.keypoint_confidence
            )
            if prediction is None:
                cv2.putText(
                    display,
                    "No person",
                    (5, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 80, 255),
                    1,
                    cv2.LINE_AA,
                )
            _draw_runtime_status(display, result, pipeline_latency.summarize())

            if args.scale != 1:
                display = cv2.resize(
                    display,
                    (display.shape[1] * args.scale, display.shape[0] * args.scale),
                    interpolation=cv2.INTER_NEAREST,
                )
            cv2.imshow(args.window_name, display)
            key = cv2.waitKey(1) & 0xFF
            pipeline_latency.update(time.perf_counter() - frame_started)
            frame_count += 1
            if key in {ord("q"), 27}:
                break
    finally:
        stream.close()
        cv2.destroyAllWindows()

    summary: dict[str, object] = {
        "frames": frame_count,
        "pipeline_camera_to_window": pipeline_latency.summarize(),
        "model_predict_call": predict_latency.summarize(),
    }
    print(json.dumps(summary, indent=2))
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/macchiato_finetune.yaml")
    parser.add_argument(
        "--checkpoint",
        default="experiments/macchiato_b/yolo/yolo26n_pose_depth/weights/best.pt",
    )
    parser.add_argument("--device", default="auto", help="Ultralytics device: auto, cpu, 0, cuda, or mps.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--range-mm", type=float, default=None)
    parser.add_argument("--timeout-ms", type=int, default=200)
    parser.add_argument("--no-rotate", action="store_true", help="Do not rotate the camera depth frame by 180 degrees.")
    parser.add_argument("--confidence", type=float, default=0.25, help="Minimum person detection confidence.")
    parser.add_argument("--keypoint-confidence", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread count.")
    parser.add_argument("--warmup-frames", type=int, default=3)
    parser.add_argument("--latency-window", type=int, default=120)
    parser.add_argument("--scale", type=int, default=3)
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; zero runs until q or Escape.")
    parser.add_argument("--window-name", default="Macchiato v9 ToF Pose")
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0 or not 0.0 <= args.keypoint_confidence <= 1.0:
        parser.error("confidence thresholds must be between 0 and 1")
    if args.warmup_frames < 0 or args.max_frames < 0:
        parser.error("warmup-frames and max-frames must be non-negative")
    if args.scale < 1 or args.latency_window < 1 or args.timeout_ms < 1 or args.threads < 1:
        parser.error("scale, latency-window, timeout-ms, and threads must be positive")
    if args.range_mm is not None and args.range_mm <= 0:
        parser.error("range-mm must be positive")
    return args


def main() -> None:
    run_live_preview(parse_args())


if __name__ == "__main__":
    main()
