"""
Tests for gestures package: FeatureExtractor, FingerDetector, GestureMapper, and GestureController.
"""

import unittest
import numpy as np

from gestures.finger_detector import FingerDetector
from gestures.feature_extractor import FeatureExtractor
from gestures.gesture_mapper import GestureMapper
from gestures.gesture_controller import GestureController


class TestGestures(unittest.TestCase):

    def _make_sample_landmarks(self):
        """Constructs a synthetic 21-landmark hand with open palm."""
        palm = [(0.5, 0.9)]  # wrist
        # MCPs & tips
        palm += [(0.35, 0.7), (0.3, 0.55), (0.25, 0.4), (0.15, 0.25)]  # thumb (extended)
        palm += [(0.4, 0.6), (0.4, 0.45), (0.4, 0.35), (0.4, 0.2)]      # index (extended)
        palm += [(0.5, 0.6), (0.5, 0.45), (0.5, 0.35), (0.5, 0.2)]      # middle (extended)
        palm += [(0.6, 0.6), (0.6, 0.45), (0.6, 0.35), (0.6, 0.2)]      # ring (extended)
        palm += [(0.7, 0.65), (0.7, 0.5), (0.7, 0.4), (0.7, 0.25)]     # pinky (extended)
        return [[i, x, y, 0.0] for i, (x, y) in enumerate(palm)]

    def test_feature_extractor_dimensions_and_invariance(self):
        landmarks = self._make_sample_landmarks()
        features = FeatureExtractor.extract_features(landmarks)

        self.assertIsNotNone(features)
        self.assertEqual(len(features), 42)
        # Wrist is coordinates [0, 1], relativized to (0,0)
        self.assertAlmostEqual(features[0], 0.0)
        self.assertAlmostEqual(features[1], 0.0)
        # Normalized values should be bounded in [-1.0, 1.0]
        self.assertLessEqual(np.max(np.abs(features)), 1.0 + 1e-5)

    def test_finger_detector_open_palm(self):
        landmarks = self._make_sample_landmarks()
        fd = FingerDetector()
        self.assertTrue(fd.is_open_palm(landmarks))
        self.assertEqual(fd.get_navigation_intent(landmarks), "STOP_VOICE")

    def test_finger_detector_closed_fist(self):
        # Wrist at (0.5, 0.9), all tips curled close to wrist
        curled = [(0.5, 0.9)]
        # thumb folded
        curled += [(0.45, 0.8), (0.45, 0.75), (0.45, 0.7), (0.45, 0.68)]
        # index folded
        curled += [(0.45, 0.7), (0.45, 0.65), (0.45, 0.7), (0.45, 0.75)]
        # middle folded
        curled += [(0.5, 0.7), (0.5, 0.65), (0.5, 0.7), (0.5, 0.75)]
        # ring folded
        curled += [(0.55, 0.7), (0.55, 0.65), (0.55, 0.7), (0.55, 0.75)]
        # pinky folded
        curled += [(0.6, 0.7), (0.6, 0.65), (0.6, 0.7), (0.6, 0.75)]
        lm = [[i, x, y, 0.0] for i, (x, y) in enumerate(curled)]

        fd = FingerDetector()
        self.assertTrue(fd.is_closed_fist(lm))
        self.assertEqual(fd.get_navigation_intent(lm), "BACK")

    def test_gesture_mapper(self):
        mapper = GestureMapper()
        self.assertEqual(mapper.map_finger_count(1), "NEXT")
        self.assertEqual(mapper.map_finger_count(2), "PREV")
        self.assertEqual(mapper.map_finger_count(3), "SELECT")
        self.assertEqual(mapper.map_finger_count(0), "BACK")
        self.assertEqual(mapper.map_finger_count(5), "STOP_VOICE")

    def test_gesture_controller_edge_triggering(self):
        dispatched = []
        gc = GestureController(
            callback=lambda i: dispatched.append(i),
            stability_frames=3,
            cooldown=0.0
        )

        for _ in range(3):
            gc.process_intent("NEXT")
        self.assertEqual(dispatched, ["NEXT"])

        # Holding pose does NOT repeat
        for _ in range(10):
            gc.process_intent("NEXT")
        self.assertEqual(dispatched, ["NEXT"])

        # Releasing allows next intent
        gc.process_intent(None)
        for _ in range(3):
            gc.process_intent("NEXT")
        self.assertEqual(dispatched, ["NEXT", "NEXT"])

    def test_gesture_controller_open_palm_fast_path_and_rearm(self):
        dispatched = []
        gc = GestureController(
            callback=lambda i: dispatched.append(i),
            stability_frames=6,
            cooldown=0.0,
            fast_interrupt_frames=2
        )

        # 2 frames triggers fast interrupt
        gc.process_intent("STOP_VOICE")
        gc.process_intent("STOP_VOICE")
        self.assertEqual(dispatched, ["STOP_VOICE"])

        # Holding open palm does NOT re-arm
        for _ in range(15):
            gc.process_intent("STOP_VOICE")
        self.assertEqual(len(dispatched), 1)

        # Re-arms once hand releases
        gc.process_intent(None)
        gc.process_intent("STOP_VOICE")
        gc.process_intent("STOP_VOICE")
        self.assertEqual(len(dispatched), 2)


if __name__ == "__main__":
    unittest.main()
