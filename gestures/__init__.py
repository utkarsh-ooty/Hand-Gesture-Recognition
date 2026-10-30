"""
Gestures package for Hand Gesture AI.
Manages camera capture, hand tracking, feature extraction, finger detection,
Random Forest classification, gesture mapping, and event stability.
"""

from gestures.camera_manager import CameraManager
from gestures.hand_tracker import HandTracker
from gestures.feature_extractor import FeatureExtractor, extract_features
from gestures.finger_detector import FingerDetector
from gestures.gesture_classifier import GestureClassifier
from gestures.gesture_mapper import GestureMapper
from gestures.gesture_controller import GestureController

__all__ = [
    "CameraManager",
    "HandTracker",
    "FeatureExtractor",
    "extract_features",
    "FingerDetector",
    "GestureClassifier",
    "GestureMapper",
    "GestureController",
]
