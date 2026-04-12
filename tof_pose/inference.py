from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from .types import JOINT_NAMES_15, Pose2D


class InferenceError(RuntimeError):
    """Model execution failed."""


def _load_interpreter(model_path: str, num_threads: int):
    path = Path(model_path)
    if not path.exists():
        raise InferenceError(
            f"Model file not found: {path}. Run `tof-pose fetch-models` for the demo baseline or pass a real .tflite path."
        )
    try:
        from tflite_runtime.interpreter import Interpreter  # type: ignore
    except ImportError:
        try:
            from tensorflow.lite import Interpreter  # type: ignore
        except ImportError as exc:
            raise InferenceError(
                "Neither tflite_runtime nor tensorflow.lite is available. Install a TFLite runtime."
            ) from exc
    interpreter = Interpreter(model_path=str(path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


class PoseEstimator(Protocol):
    def predict(self, roi_tensor: np.ndarray, roi_px: tuple[int, int, int, int]) -> Pose2D:
        ...


@dataclass(slots=True)
class MoveNetEstimator:
    model_path: str
    num_threads: int = 4

    def __post_init__(self) -> None:
        self.interpreter = _load_interpreter(self.model_path, self.num_threads)
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()[0]
        _, self.input_h, self.input_w, self.input_c = self.input_details["shape"]

    @staticmethod
    def _repeat_channels(image: np.ndarray, channels: int) -> np.ndarray:
        if image.ndim == 2:
            image = image[..., None]
        if image.shape[-1] == channels:
            return image
        if image.shape[-1] == 1:
            return np.repeat(image, channels, axis=-1)
        return image[..., :channels]

    def predict(self, roi_tensor: np.ndarray, roi_px: tuple[int, int, int, int]) -> Pose2D:
        amplitude = roi_tensor[..., 2] if roi_tensor.ndim == 3 else roi_tensor
        network_input = self._repeat_channels(amplitude.astype(np.float32), self.input_c)
        network_input = np.expand_dims(network_input, axis=0)
        if np.issubdtype(self.input_details["dtype"], np.integer):
            network_input = np.clip(network_input * 255.0, 0, 255).astype(self.input_details["dtype"])
        else:
            network_input = network_input.astype(np.float32)

        self.interpreter.set_tensor(self.input_details["index"], network_input)
        self.interpreter.invoke()
        output = self.interpreter.get_tensor(self.output_details["index"])
        keypoints = np.asarray(output[0, 0], dtype=np.float32)
        return _movenet_to_pose15(keypoints, roi_px)


@dataclass(slots=True)
class HeatmapOffsetPoseEstimator:
    model_path: str
    num_threads: int = 4
    score_threshold: float = 0.1

    def __post_init__(self) -> None:
        self.interpreter = _load_interpreter(self.model_path, self.num_threads)
        self.input_details = self.interpreter.get_input_details()[0]
        self.output_details = self.interpreter.get_output_details()
        _, self.input_h, self.input_w, self.input_c = self.input_details["shape"]
        if len(self.output_details) < 2:
            raise InferenceError(
                "HeatmapOffsetPoseEstimator expects at least two outputs: heatmaps and offsets."
            )

    def predict(self, roi_tensor: np.ndarray, roi_px: tuple[int, int, int, int]) -> Pose2D:
        network_input = np.expand_dims(roi_tensor.astype(np.float32), axis=0)
        if network_input.shape[-1] != self.input_c:
            raise InferenceError(
                f"Model expects {self.input_c} channels, got {network_input.shape[-1]}"
            )
        if np.issubdtype(self.input_details["dtype"], np.integer):
            scale, zero_point = self.input_details["quantization"]
            scale = scale or 1.0 / 255.0
            network_input = np.round(network_input / scale + zero_point).astype(self.input_details["dtype"])
        self.interpreter.set_tensor(self.input_details["index"], network_input)
        self.interpreter.invoke()

        outputs = [self.interpreter.get_tensor(detail["index"]) for detail in self.output_details[:2]]
        heatmaps, offsets = outputs
        heatmaps = np.asarray(heatmaps[0], dtype=np.float32)
        offsets = np.asarray(offsets[0], dtype=np.float32)
        if heatmaps.ndim != 3 or offsets.ndim != 3:
            raise InferenceError("Unexpected output tensor ranks for heatmaps/offsets")

        heat_h, heat_w, joints = heatmaps.shape
        joints_uv = np.zeros((joints, 2), dtype=np.float32)
        scores = np.zeros((joints,), dtype=np.float32)
        stride_y = self.input_h / max(heat_h, 1)
        stride_x = self.input_w / max(heat_w, 1)

        for joint_idx in range(joints):
            plane = heatmaps[:, :, joint_idx]
            flat_idx = int(np.argmax(plane))
            y, x = np.unravel_index(flat_idx, plane.shape)
            score = float(plane[y, x])
            scores[joint_idx] = score
            off_y = float(offsets[y, x, joint_idx * 2 + 0])
            off_x = float(offsets[y, x, joint_idx * 2 + 1])
            joints_uv[joint_idx, 0] = x * stride_x + off_x
            joints_uv[joint_idx, 1] = y * stride_y + off_y

        scores = np.where(scores >= self.score_threshold, scores, 0.0)
        return Pose2D(joints_uv=joints_uv, scores=scores, roi_px=roi_px)


def _movenet_to_pose15(
    keypoints17: np.ndarray,
    roi_px: tuple[int, int, int, int],
) -> Pose2D:
    x0, y0, x1, y1 = roi_px
    width = max(x1 - x0, 1)
    height = max(y1 - y0, 1)

    def _uv(idx: int) -> np.ndarray:
        y, x, _ = keypoints17[idx]
        return np.asarray([x * width, y * height], dtype=np.float32)

    def _score(idx: int) -> float:
        return float(keypoints17[idx][2])

    left_shoulder = _uv(5)
    right_shoulder = _uv(6)
    left_hip = _uv(11)
    right_hip = _uv(12)
    joints = np.zeros((len(JOINT_NAMES_15), 2), dtype=np.float32)
    scores = np.zeros((len(JOINT_NAMES_15),), dtype=np.float32)

    joints[0] = _uv(0)
    scores[0] = _score(0)
    joints[1] = (left_shoulder + right_shoulder) / 2.0
    scores[1] = min(_score(5), _score(6))
    joints[2] = (left_shoulder + right_shoulder + left_hip + right_hip) / 4.0
    scores[2] = min(_score(5), _score(6), _score(11), _score(12))
    joints[3], scores[3] = _uv(6), _score(6)
    joints[4], scores[4] = _uv(8), _score(8)
    joints[5], scores[5] = _uv(10), _score(10)
    joints[6], scores[6] = _uv(5), _score(5)
    joints[7], scores[7] = _uv(7), _score(7)
    joints[8], scores[8] = _uv(9), _score(9)
    joints[9], scores[9] = _uv(12), _score(12)
    joints[10], scores[10] = _uv(14), _score(14)
    joints[11], scores[11] = _uv(16), _score(16)
    joints[12], scores[12] = _uv(11), _score(11)
    joints[13], scores[13] = _uv(13), _score(13)
    joints[14], scores[14] = _uv(15), _score(15)
    return Pose2D(joints_uv=joints, scores=scores, roi_px=roi_px)
