from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .geometry import FloorCandidateMasks
from .types import DepthFrame, Pose2D


class VisualizationError(RuntimeError):
    """Preview rendering failed."""


SKELETON_EDGES_15 = (
    (0, 1),
    (1, 2),
    (1, 3),
    (3, 4),
    (4, 5),
    (1, 6),
    (6, 7),
    (7, 8),
    (2, 9),
    (9, 10),
    (10, 11),
    (2, 12),
    (12, 13),
    (13, 14),
    (9, 12),
)


def _require_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ImportError as exc:
        raise VisualizationError(
            "OpenCV is required for preview windows. Install the runtime extras with `python3 -m pip install -e '.[runtime]'`."
        ) from exc
    return cv2


def _normalize_for_display(
    image: np.ndarray,
    valid_mask: np.ndarray | None = None,
    low_percentile: float = 5.0,
    high_percentile: float = 95.0,
) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(arr)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(arr)

    output = np.zeros(arr.shape, dtype=np.uint8)
    if not np.any(valid_mask):
        return output

    values = arr[valid_mask]
    low = float(np.percentile(values, low_percentile))
    high = float(np.percentile(values, high_percentile))
    normalized = np.clip((arr - low) / max(high - low, 1e-6), 0.0, 1.0)
    output[valid_mask] = np.round(normalized[valid_mask] * 255.0).astype(np.uint8)
    return output


def _pose_pixels(pose2d: Pose2D, model_shape: tuple[int, int]) -> np.ndarray:
    x0, y0, x1, y1 = pose2d.roi_px
    model_h, model_w = model_shape
    scale_x = max(x1 - x0, 1) / max(model_w, 1)
    scale_y = max(y1 - y0, 1) / max(model_h, 1)
    points = np.zeros_like(pose2d.joints_uv, dtype=np.int32)
    points[:, 0] = np.round(x0 + pose2d.joints_uv[:, 0] * scale_x).astype(np.int32)
    points[:, 1] = np.round(y0 + pose2d.joints_uv[:, 1] * scale_y).astype(np.int32)
    return points


def _diagnostics_payload(diagnostics: Any) -> dict[str, Any]:
    if diagnostics is None:
        return {}
    if hasattr(diagnostics, "to_json"):
        return diagnostics.to_json()
    if isinstance(diagnostics, dict):
        return diagnostics
    return {}


@dataclass(slots=True)
class PreviewWindow:
    title: str = "ToF Pose Preview"
    wait_key_ms: int = 1
    cv2: Any = field(init=False, repr=False)
    _window_created: bool = field(init=False, default=False, repr=False)

    def __post_init__(self) -> None:
        self.cv2 = _require_cv2()
        self._window_created = False

    def render(
        self,
        frame: DepthFrame,
        result: dict[str, Any] | None,
        fps: float | None = None,
        frame_index: int | None = None,
    ) -> np.ndarray:
        cv2 = self.cv2
        amp_gray = _normalize_for_display(frame.amplitude)
        amp_bgr = cv2.cvtColor(amp_gray, cv2.COLOR_GRAY2BGR)

        depth_valid = frame.valid_mask & np.isfinite(frame.depth_m)
        depth_gray = _normalize_for_display(frame.depth_m, valid_mask=depth_valid)
        depth_bgr = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)
        depth_bgr[~depth_valid] = 0

        self._annotate_panel(amp_bgr, "Amplitude")
        self._annotate_panel(depth_bgr, "Depth")

        if result is None:
            self._put_status(amp_bgr, "No track")
            self._put_status(depth_bgr, "No track")
        else:
            track = result["track"]
            pose2d = result["pose2d"]
            diagnostics = _diagnostics_payload(result.get("tracking_diagnostics"))
            pose_points = _pose_pixels(pose2d, result["roi_tensor"].shape[:2])
            self._draw_tracking_debug(amp_bgr, track.mask, diagnostics)
            self._draw_tracking_debug(depth_bgr, track.mask, diagnostics)
            self._draw_overlay(amp_bgr, track.roi_px, pose_points, pose2d.scores)
            self._draw_overlay(depth_bgr, track.roi_px, pose_points, pose2d.scores)
            self._put_status(amp_bgr, f"Track {track.cluster_id} q={track.quality:.2f}")
            self._put_status(depth_bgr, f"Valid joints {int(np.count_nonzero(result['pose3d'].valid))}")
            debug_line = self._tracking_debug_line(diagnostics)
            if debug_line:
                self._put_lines(amp_bgr, [debug_line])
                self._put_lines(depth_bgr, [debug_line])

        canvas = np.concatenate([amp_bgr, depth_bgr], axis=1)
        self._annotate_canvas(canvas, fps=fps, frame_index=frame_index)
        return canvas

    def show(
        self,
        frame: DepthFrame,
        result: dict[str, Any] | None,
        fps: float | None = None,
        frame_index: int | None = None,
    ) -> bool:
        cv2 = self.cv2
        image = self.render(frame, result, fps=fps, frame_index=frame_index)
        if not self._window_created:
            try:
                cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
                self._window_created = True
            except cv2.error as exc:  # type: ignore[attr-defined]
                raise VisualizationError(
                    "OpenCV preview could not open a GUI window. Run from a local desktop session, VNC, or X11-forwarded shell."
                ) from exc
        try:
            cv2.imshow(self.title, image)
            key = cv2.waitKey(self.wait_key_ms) & 0xFF
        except cv2.error as exc:  # type: ignore[attr-defined]
            raise VisualizationError(
                "OpenCV preview failed while drawing the window. Run from a local desktop session, VNC, or X11-forwarded shell."
            ) from exc
        return key not in (27, ord("q"), ord("Q"))

    def render_calibration(
        self,
        frame: DepthFrame,
        masks: FloorCandidateMasks,
        frame_index: int | None = None,
    ) -> np.ndarray:
        cv2 = self.cv2
        amp_gray = _normalize_for_display(frame.amplitude)
        amp_bgr = cv2.cvtColor(amp_gray, cv2.COLOR_GRAY2BGR)

        depth_valid = frame.valid_mask & np.isfinite(frame.depth_m)
        depth_gray = _normalize_for_display(frame.depth_m, valid_mask=depth_valid)
        depth_bgr = cv2.applyColorMap(depth_gray, cv2.COLORMAP_TURBO)
        depth_bgr[~depth_valid] = 0

        candidate_bgr = np.zeros((*frame.depth_m.shape, 3), dtype=np.uint8)
        candidate_bgr[masks.depth_valid] = (80, 80, 80)
        candidate_bgr[masks.lower_candidates] = (255, 160, 0)
        candidate_bgr[masks.strict_candidates] = (0, 220, 255)
        candidate_bgr[masks.used_mask] = (0, 255, 0)

        self._annotate_panel(amp_bgr, "Amplitude")
        self._annotate_panel(depth_bgr, "Depth")
        self._annotate_panel(candidate_bgr, "Floor Candidates")
        self._put_lines(
            candidate_bgr,
            [
                f"valid {int(np.count_nonzero(masks.depth_valid))}",
                f"lower {int(np.count_nonzero(masks.lower_candidates))}",
                f"strict {int(np.count_nonzero(masks.strict_candidates))}",
                f"used {int(np.count_nonzero(masks.used_mask))}",
            ],
        )

        canvas = np.concatenate([amp_bgr, depth_bgr, candidate_bgr], axis=1)
        self._annotate_canvas(canvas, fps=None, frame_index=frame_index)
        return canvas

    def show_calibration(
        self,
        frame: DepthFrame,
        masks: FloorCandidateMasks,
        frame_index: int | None = None,
    ) -> bool:
        cv2 = self.cv2
        image = self.render_calibration(frame, masks, frame_index=frame_index)
        if not self._window_created:
            try:
                cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
                self._window_created = True
            except cv2.error as exc:  # type: ignore[attr-defined]
                raise VisualizationError(
                    "OpenCV preview could not open a GUI window. Run from a local desktop session, VNC, or X11-forwarded shell."
                ) from exc
        try:
            cv2.imshow(self.title, image)
            key = cv2.waitKey(self.wait_key_ms) & 0xFF
        except cv2.error as exc:  # type: ignore[attr-defined]
            raise VisualizationError(
                "OpenCV preview failed while drawing the window. Run from a local desktop session, VNC, or X11-forwarded shell."
            ) from exc
        return key not in (27, ord("q"), ord("Q"))

    def close(self) -> None:
        if not self._window_created:
            return
        try:
            self.cv2.destroyWindow(self.title)
        except Exception:
            pass
        self._window_created = False

    def _draw_overlay(
        self,
        image: np.ndarray,
        roi_px: tuple[int, int, int, int],
        pose_points: np.ndarray,
        scores: np.ndarray,
    ) -> None:
        cv2 = self.cv2
        x0, y0, x1, y1 = roi_px
        cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 2)
        for start_idx, end_idx in SKELETON_EDGES_15:
            if scores[start_idx] <= 0.0 or scores[end_idx] <= 0.0:
                continue
            start = pose_points[start_idx]
            end = pose_points[end_idx]
            cv2.line(image, (int(start[0]), int(start[1])), (int(end[0]), int(end[1])), (0, 255, 255), 2)
        for idx, score in enumerate(scores):
            if score <= 0.0:
                continue
            u, v = pose_points[idx]
            cv2.circle(image, (int(u), int(v)), 3, (255, 255, 255), -1)
            cv2.circle(image, (int(u), int(v)), 5, (0, 140, 255), 1)

    def _draw_tracking_debug(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        diagnostics: dict[str, Any],
    ) -> None:
        cv2 = self.cv2
        if mask.shape == image.shape[:2] and np.any(mask):
            mask_u8 = (mask.astype(np.uint8) * 255)
            contour_result = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = contour_result[-2]
            cv2.drawContours(image, contours, -1, (255, 0, 255), 1)

        raw_roi = diagnostics.get("raw_roi_px")
        final_roi = diagnostics.get("final_roi_px")
        if raw_roi is not None:
            x0, y0, x1, y1 = [int(v) for v in raw_roi]
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (255, 170, 0), 1)
        if final_roi is not None:
            x0, y0, x1, y1 = [int(v) for v in final_roi]
            cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 0), 1)

    def _tracking_debug_line(self, diagnostics: dict[str, Any]) -> str:
        if not diagnostics:
            return ""
        source = diagnostics.get("track_source", "unknown")
        held = int(bool(diagnostics.get("held", False)))
        roi_area = float(diagnostics.get("roi_area_frac", 0.0))
        candidates = int(diagnostics.get("candidate_count", 0))
        return f"src={source} held={held} roi={roi_area:.2f} cand={candidates}"

    def _annotate_panel(self, image: np.ndarray, label: str) -> None:
        self.cv2.putText(
            image,
            label,
            (8, 20),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )

    def _put_status(self, image: np.ndarray, text: str) -> None:
        self.cv2.putText(
            image,
            text,
            (8, image.shape[0] - 10),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            self.cv2.LINE_AA,
        )

    def _put_lines(self, image: np.ndarray, lines: list[str]) -> None:
        y = image.shape[0] - 10 - max(len(lines) - 1, 0) * 18
        for line in lines:
            self.cv2.putText(
                image,
                line,
                (8, y),
                self.cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                self.cv2.LINE_AA,
            )
            y += 18

    def _annotate_canvas(self, canvas: np.ndarray, fps: float | None, frame_index: int | None) -> None:
        parts: list[str] = []
        if frame_index is not None:
            parts.append(f"frame {frame_index}")
        if fps is not None:
            parts.append(f"{fps:.1f} FPS")
        if not parts:
            return
        text = " | ".join(parts)
        self.cv2.putText(
            canvas,
            text,
            (8, 42),
            self.cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            self.cv2.LINE_AA,
        )
