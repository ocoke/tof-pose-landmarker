from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from tof_pose.visualization import PreviewWindow


class VisualizationTests(unittest.TestCase):
    def test_preview_window_initializes_with_slots(self) -> None:
        fake_cv2 = SimpleNamespace()
        with patch("tof_pose.visualization._require_cv2", return_value=fake_cv2):
            preview = PreviewWindow(title="test")
        self.assertIs(preview.cv2, fake_cv2)
        self.assertFalse(preview._window_created)


if __name__ == "__main__":
    unittest.main()
