from __future__ import annotations

import unittest

from tof_pose.cli import build_parser


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

    def test_run_production_accepts_preview_flag(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["run-production", "--pose-model", "model.tflite", "--preview"])
        self.assertTrue(args.preview)


if __name__ == "__main__":
    unittest.main()
