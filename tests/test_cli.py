from __future__ import annotations

import unittest

from tof_pose.cli import build_parser


class CliTests(unittest.TestCase):
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
