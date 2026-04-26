from __future__ import annotations

import unittest

import numpy as np

from tof_pose.geometry import GeometryConfig, GeometricPersonTracker, clean_mask, component_roi, connected_components, estimate_floor_plane
from tof_pose.pipeline import synthetic_frame


class GeometryTests(unittest.TestCase):
    def test_estimate_floor_plane_fits_flat_surface(self) -> None:
        rng = np.random.default_rng(0)
        xs = rng.uniform(-1.0, 1.0, size=200)
        zs = rng.uniform(1.0, 4.0, size=200)
        ys = np.full_like(xs, 1.2) + rng.normal(scale=0.005, size=xs.shape)
        points = np.stack((xs, ys, zs), axis=-1).astype(np.float32)
        plane = estimate_floor_plane(points, iterations=96, distance_threshold_m=0.02)
        self.assertIsNotNone(plane)
        assert plane is not None
        distances = np.abs(plane.signed_distance(points))
        self.assertLess(float(np.median(distances)), 0.02)

    def test_connected_components_and_roi(self) -> None:
        mask = np.zeros((12, 12), dtype=bool)
        mask[2:5, 3:6] = True
        mask[7:10, 8:11] = True
        components = connected_components(mask)
        self.assertEqual(len(components), 2)
        roi = component_roi(components[0], margin_px=1, frame_shape=mask.shape)
        self.assertEqual(len(roi), 4)

    def test_tracker_extracts_person_component(self) -> None:
        depth = np.full((48, 64), 3.8, dtype=np.float32)
        depth[10:38, 24:40] = 2.5
        amplitude = np.ones_like(depth, dtype=np.float32)
        confidence = np.ones_like(depth, dtype=np.float32) * 80.0
        frame = synthetic_frame(depth, amplitude=amplitude, confidence=confidence, timestamp=1.0)

        tracker = GeometricPersonTracker(
            GeometryConfig(
                min_component_pixels=20,
                confidence_threshold=10.0,
                background_threshold_m=0.05,
                min_human_height_m=0.2,
                max_human_height_m=3.5,
                min_human_width_m=0.05,
                max_human_width_m=3.0,
            )
        )
        tracker.background.update(np.full_like(depth, 3.8), np.ones_like(depth, dtype=bool))
        result = tracker.update(frame)
        self.assertIsNotNone(result)
        assert result is not None
        x0, y0, x1, y1 = result.roi_px
        self.assertLess(x0, 28)
        self.assertGreater(x1, 36)
        self.assertLess(y0, 14)
        self.assertGreater(y1, 34)
        diagnostics = tracker.last_diagnostics
        self.assertEqual(diagnostics.track_source, "foreground")
        self.assertFalse(diagnostics.held)
        self.assertGreater(diagnostics.roi_area_frac, 0.0)
        self.assertGreater(diagnostics.mask_pixels, 0)
        self.assertGreaterEqual(diagnostics.candidate_count, 1)
        self.assertIsNotNone(diagnostics.raw_roi_px)
        self.assertIsNotNone(diagnostics.final_roi_px)

    def test_tracker_fallback_extracts_nearest_depth_blob(self) -> None:
        depth = np.full((48, 64), 3.8, dtype=np.float32)
        depth[10:38, 24:40] = 1.8
        confidence = np.zeros_like(depth, dtype=np.float32)
        frame = synthetic_frame(depth, confidence=confidence, timestamp=1.0)

        tracker = GeometricPersonTracker(
            GeometryConfig(
                min_component_pixels=20,
                confidence_threshold=90.0,
                min_human_height_m=2.5,
                fallback_depth_window_m=0.5,
            )
        )
        tracker.background.update(depth.copy(), np.ones_like(depth, dtype=bool))
        result = tracker.update(frame)
        self.assertIsNotNone(result)
        assert result is not None
        x0, y0, x1, y1 = result.roi_px
        self.assertLessEqual(x0, 24)
        self.assertGreaterEqual(x1, 40)
        self.assertLessEqual(y0, 10)
        self.assertGreaterEqual(y1, 38)
        diagnostics = tracker.last_diagnostics
        self.assertEqual(diagnostics.track_source, "fallback")
        self.assertFalse(diagnostics.held)
        self.assertGreater(diagnostics.roi_area_frac, 0.0)
        self.assertGreater(diagnostics.mask_pixels, 0)
        self.assertGreaterEqual(diagnostics.candidate_count, 1)

    def test_tracker_fallback_prefers_compact_depth_slice_over_full_frame(self) -> None:
        depth = np.full((48, 64), 2.4, dtype=np.float32)
        depth[10:38, 24:40] = 1.8
        confidence = np.zeros_like(depth, dtype=np.float32)
        frame = synthetic_frame(depth, confidence=confidence, timestamp=1.0)

        tracker = GeometricPersonTracker(
            GeometryConfig(
                min_component_pixels=20,
                confidence_threshold=90.0,
                min_human_height_m=2.5,
                fallback_depth_window_m=0.9,
            )
        )
        tracker.background.update(depth.copy(), np.ones_like(depth, dtype=bool))
        result = tracker.update(frame)
        self.assertIsNotNone(result)
        assert result is not None
        x0, y0, x1, y1 = result.roi_px
        self.assertGreater(x0, 0)
        self.assertLess(x1, 64)
        self.assertLess(x1 - x0, 64)
        self.assertEqual((y0, y1), (0, 48))

    def test_tracker_fallback_rejects_scene_sized_blob(self) -> None:
        depth = np.full((48, 64), 2.4, dtype=np.float32)
        confidence = np.zeros_like(depth, dtype=np.float32)
        frame = synthetic_frame(depth, confidence=confidence, timestamp=1.0)

        tracker = GeometricPersonTracker(
            GeometryConfig(
                min_component_pixels=20,
                confidence_threshold=90.0,
                min_human_height_m=2.5,
            )
        )
        tracker.background.update(depth.copy(), np.ones_like(depth, dtype=bool))
        self.assertIsNone(tracker.update(frame))
        diagnostics = tracker.last_diagnostics
        self.assertEqual(diagnostics.track_source, "none")
        self.assertFalse(diagnostics.held)
        self.assertGreaterEqual(diagnostics.candidate_count, 1)
        self.assertGreater(sum(diagnostics.reject_counts.values()), 0)

    def test_tracker_holds_previous_roi_on_abrupt_depth_jump(self) -> None:
        depth_a = np.full((48, 64), 3.8, dtype=np.float32)
        depth_a[10:38, 24:40] = 2.5
        depth_b = np.full((48, 64), 3.8, dtype=np.float32)
        depth_b[10:38, 24:40] = 1.8
        confidence = np.zeros_like(depth_a, dtype=np.float32)

        tracker = GeometricPersonTracker(
            GeometryConfig(
                min_component_pixels=20,
                confidence_threshold=90.0,
                min_human_height_m=2.5,
                track_max_depth_jump_m=0.35,
                track_hold_frames=2,
            )
        )
        tracker.background.update(depth_a.copy(), np.ones_like(depth_a, dtype=bool))
        first = tracker.update(synthetic_frame(depth_a, confidence=confidence, timestamp=1.0))
        self.assertIsNotNone(first)
        assert first is not None

        held = tracker.update(synthetic_frame(depth_b, confidence=confidence, timestamp=1.1))
        self.assertIsNotNone(held)
        assert held is not None
        self.assertEqual(held.roi_px, first.roi_px)
        self.assertEqual(float(held.quality), 0.0)
        self.assertAlmostEqual(float(held.centroid_xyz[2]), float(first.centroid_xyz[2]), delta=1e-5)
        diagnostics = tracker.last_diagnostics
        self.assertEqual(diagnostics.track_source, "held")
        self.assertTrue(diagnostics.held)
        self.assertEqual(diagnostics.final_roi_px, first.roi_px)

    def test_clean_mask_preserves_core_blob(self) -> None:
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:6, 2:6] = True
        mask[0, 0] = True
        cleaned = clean_mask(mask)
        self.assertTrue(cleaned[3, 3])
        self.assertFalse(cleaned[0, 0])


if __name__ == "__main__":
    unittest.main()
