from __future__ import annotations

import importlib.metadata
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import CameraIntrinsics, DepthFrame


class CameraError(RuntimeError):
    """Camera or SDK interaction failed."""


def _require_sdk() -> Any:
    try:
        import ArducamDepthCamera as ac  # type: ignore
    except ImportError as exc:
        raise CameraError(
            "ArducamDepthCamera is not installed. Install version 0.1.24 on the Raspberry Pi."
        ) from exc
    return ac


@dataclass(slots=True)
class CameraConfig:
    connection: str = "CSI"
    index: int = 0
    range_mode_m: int = 4
    request_timeout_ms: int = 200
    start_frame_type: str = "DEPTH"


class ArducamCameraAdapter:
    """Thin wrapper around the latest ArducamDepthCamera Python API."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._ac = None
        self._cam = None
        self._intrinsics: CameraIntrinsics | None = None

    @staticmethod
    def sdk_version() -> str | None:
        try:
            return importlib.metadata.version("arducamdepthcamera")
        except importlib.metadata.PackageNotFoundError:
            return None

    def open(self) -> None:
        self._ac = _require_sdk()
        self._cam = self._ac.ArducamCamera()
        connection = getattr(self._ac.Connection, self.config.connection)
        frame_type = getattr(self._ac.FrameType, self.config.start_frame_type)

        open_code = self._cam.open(connection, self.config.index)
        if int(open_code) != 0:
            raise CameraError(f"Camera open failed with code {open_code}")

        start_code = self._cam.start(frame_type)
        if int(start_code) != 0:
            self.close()
            raise CameraError(f"Camera start failed with code {start_code}")

        self._set_range_mode(self.config.range_mode_m)
        self._intrinsics = self._read_intrinsics()

    def _set_range_mode(self, range_mode_m: int) -> None:
        if self._cam is None or self._ac is None:
            return
        try:
            self._cam.setControl(self._ac.Control.RANGE, int(range_mode_m))
        except Exception as exc:  # pragma: no cover - depends on SDK presence
            raise CameraError(f"Failed to set RANGE={range_mode_m}m: {exc}") from exc

    def _read_intrinsics(self) -> CameraIntrinsics:
        if self._cam is None or self._ac is None:
            raise CameraError("Camera is not open")

        try:
            fx = float(self._cam.getControl(self._ac.Control.INTRINSIC_FX))
            fy = float(self._cam.getControl(self._ac.Control.INTRINSIC_FY))
            cx = float(self._cam.getControl(self._ac.Control.INTRINSIC_CX))
            cy = float(self._cam.getControl(self._ac.Control.INTRINSIC_CY))
        except Exception as exc:
            raise CameraError(f"Unable to read camera intrinsics: {exc}") from exc
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy)

    def read(self) -> DepthFrame:
        if self._cam is None or self._intrinsics is None:
            raise CameraError("Camera is not open")

        frame = self._cam.requestFrame(self.config.request_timeout_ms)
        if frame is None:
            raise CameraError("requestFrame returned None")

        try:
            depth = np.asarray(frame.depth_data, dtype=np.float32)
            amplitude = np.asarray(frame.amplitude_data, dtype=np.float32)
            confidence = np.asarray(frame.confidence_data, dtype=np.float32)
            timestamp = float(getattr(frame.format, "timestamp", time.time()))
        finally:
            self._cam.releaseFrame(frame)

        valid = np.isfinite(depth) & np.isfinite(amplitude) & np.isfinite(confidence) & (depth > 0.0)
        return DepthFrame(
            depth_m=depth,
            amplitude=amplitude,
            confidence=confidence,
            valid_mask=valid,
            intrinsics=self._intrinsics,
            timestamp=timestamp,
        )

    def inspect(self) -> dict[str, Any]:
        ac = _require_sdk()
        frame_methods: list[str] = []
        with self:
            frame = self._cam.requestFrame(self.config.request_timeout_ms)
            if frame is None:
                raise CameraError("requestFrame returned None during inspect")
            try:
                frame_methods = [name for name in dir(frame) if not name.startswith("_")]
            finally:
                self._cam.releaseFrame(frame)

        return {
            "sdk_version": self.sdk_version(),
            "connection": self.config.connection,
            "frame_type": str(ac.FrameType.DEPTH),
            "frame_methods": frame_methods,
        }

    def close(self) -> None:
        if self._cam is None:
            return
        try:
            self._cam.stop()
        except Exception:
            pass
        try:
            self._cam.close()
        except Exception:
            pass
        self._cam = None
        self._ac = None
        self._intrinsics = None

    def __enter__(self) -> "ArducamCameraAdapter":
        self.open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
