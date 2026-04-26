from __future__ import annotations

import json
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from tof_pose.cli import _debug_payload, _write_debug_payload, build_parser
from tof_pose.geometry import TrackingDiagnostics


class CliTests(unittest.TestCase):
    def test_calibrate_floor_accepts_preview_and_depth_unit(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["calibrate-floor", "--preview", "--depth-unit", "mm"])
        self.assertTrue(args.preview)
        self.assertEqual(args.depth_unit, "mm")

    def test_run_demo_accepts_preview_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-demo", "--preview"])
        self.assertTrue(args.preview)

    def test_run_demo_accepts_debug_and_tuning_flags(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "run-demo",
                "--debug-jsonl",
                "run.jsonl",
                "--debug-every",
                "2",
                "--confidence-threshold",
                "12",
                "--background-threshold-m",
                "0.2",
                "--fallback-depth-window-m",
                "0.4",
                "--track-max-depth-jump-m",
                "0.5",
                "--roi-margin-px",
                "12",
                "--track-size-step-fraction",
                "0.2",
            ]
        )
        self.assertEqual(str(args.debug_jsonl), "run.jsonl")
        self.assertEqual(args.debug_every, 2)
        self.assertEqual(args.confidence_threshold, 12.0)
        self.assertEqual(args.background_threshold_m, 0.2)
        self.assertEqual(args.fallback_depth_window_m, 0.4)
        self.assertEqual(args.track_max_depth_jump_m, 0.5)
        self.assertEqual(args.roi_margin_px, 12)
        self.assertEqual(args.track_size_step_fraction, 0.2)

    def test_run_production_accepts_preview_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-production", "--pose-model", "model.tflite", "--preview"])
        self.assertTrue(args.preview)

    def test_debug_payload_writes_valid_jsonl(self) -> None:
        diagnostics = TrackingDiagnostics(
            track_source="fallback",
            held=False,
            roi_area_frac=0.5,
            mask_pixels=123,
            candidate_count=4,
            reject_counts={"roi_too_large": 1},
            raw_roi_px=(1, 2, 3, 4),
            final_roi_px=(2, 3, 4, 5),
        )
        result = {
            "track": SimpleNamespace(
                cluster_id=7,
                quality=12.3456,
                roi_px=(2, 3, 4, 5),
                centroid_xyz=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                extent_xyz=np.asarray([0.1, 0.2, 0.3], dtype=np.float32),
            ),
            "pose3d": SimpleNamespace(valid=np.asarray([True, False, True])),
        }
        payload = _debug_payload(9, result, diagnostics, fps=11.25)
        with tempfile.TemporaryFile("w+") as handle:
            _write_debug_payload(handle, payload)
            handle.seek(0)
            loaded = json.loads(handle.readline())
        self.assertEqual(loaded["frame"], 9)
        self.assertEqual(loaded["tracking_diagnostics"]["track_source"], "fallback")
        self.assertEqual(loaded["valid_joints"], 2)


if __name__ == "__main__":
    unittest.main()
