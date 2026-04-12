from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


JOINT_NAMES_15 = (
    "head",
    "neck",
    "mid_spine",
    "right_shoulder",
    "right_elbow",
    "right_hand",
    "left_shoulder",
    "left_elbow",
    "left_hand",
    "right_hip",
    "right_knee",
    "right_foot",
    "left_hip",
    "left_knee",
    "left_foot",
)


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    def back_project(self, u: np.ndarray, v: np.ndarray, z: np.ndarray) -> np.ndarray:
        x = (u - self.cx) * z / max(self.fx, 1e-6)
        y = (v - self.cy) * z / max(self.fy, 1e-6)
        return np.stack((x, y, z), axis=-1)


@dataclass(slots=True)
class DepthFrame:
    depth_m: np.ndarray
    amplitude: np.ndarray
    confidence: np.ndarray
    valid_mask: np.ndarray
    intrinsics: CameraIntrinsics
    timestamp: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.depth_m.shape


@dataclass(slots=True)
class FloorPlane:
    normal: np.ndarray
    offset: float
    support: int

    def signed_distance(self, points_xyz: np.ndarray) -> np.ndarray:
        return points_xyz @ self.normal + self.offset

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["normal"] = self.normal.tolist()
        return payload

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> "FloorPlane":
        return cls(
            normal=np.asarray(payload["normal"], dtype=np.float32),
            offset=float(payload["offset"]),
            support=int(payload.get("support", 0)),
        )


@dataclass(slots=True)
class TrackedPerson:
    roi_px: tuple[int, int, int, int]
    cluster_id: int
    centroid_xyz: np.ndarray
    extent_xyz: np.ndarray
    quality: float
    mask: np.ndarray


@dataclass(slots=True)
class Pose2D:
    joints_uv: np.ndarray
    scores: np.ndarray
    roi_px: tuple[int, int, int, int]


@dataclass(slots=True)
class Pose3D:
    joints_xyz: np.ndarray
    valid: np.ndarray
    scores: np.ndarray
    roi_px: tuple[int, int, int, int]
