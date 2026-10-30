"""
MediaPipe hand tracking wrapper with landmark extraction and rendering.
"""

import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from typing import List, Tuple, Optional
import numpy as np

from app.config import Config, get_safe_path

# Hand skeleton connections for visualization
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # base
]


class HandTracker:
    """Wrapper around MediaPipe HandLandmarker Tasks API."""

    def __init__(self, model_path: Optional[str] = None, max_hands: int = 1,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5):
        if model_path is None:
            model_path = Config.HAND_LANDMARKER_PATH

        model_path = get_safe_path(model_path)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe task model not found at {model_path}")

        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_results = None

    def find_hands(self, frame: np.ndarray, draw: bool = True) -> np.ndarray:
        """Processes frame, detects hands, optionally draws landmarks, and returns annotated frame."""
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        self.last_results = self.detector.detect(mp_image)

        if self.last_results and self.last_results.hand_landmarks and draw:
            h, w, _ = frame.shape
            for hand_landmarks in self.last_results.hand_landmarks:
                pts = {}
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts[idx] = (cx, cy)
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)

                for p1, p2 in HAND_CONNECTIONS:
                    if p1 in pts and p2 in pts:
                        cv2.line(frame, pts[p1], pts[p2], (0, 255, 0), 2)

        return frame

    def get_landmarks(self, frame: np.ndarray, hand_no: int = 0) -> List[List[float]]:
        """Extracts normalized landmarks [id, x, y, z] for specified hand."""
        landmark_list = []
        if self.last_results and self.last_results.hand_landmarks:
            if len(self.last_results.hand_landmarks) > hand_no:
                my_hand = self.last_results.hand_landmarks[hand_no]
                for id_idx, lm in enumerate(my_hand):
                    landmark_list.append([id_idx, lm.x, lm.y, lm.z])
        return landmark_list

    def get_handedness(self, hand_no: int = 0) -> Tuple[Optional[str], float]:
        """Returns ('Right' | 'Left', confidence_score) for specified hand."""
        if self.last_results and self.last_results.handedness:
            if len(self.last_results.handedness) > hand_no:
                category = self.last_results.handedness[hand_no][0]
                return category.category_name, category.score
        return None, 0.0

    def get_pixel_landmarks(self, frame: np.ndarray, hand_no: int = 0) -> List[List[int]]:
        """Extracts landmarks as pixel coordinates [id, cx, cy] in the frame."""
        pixel_landmarks = []
        if self.last_results and self.last_results.hand_landmarks:
            if len(self.last_results.hand_landmarks) > hand_no:
                my_hand = self.last_results.hand_landmarks[hand_no]
                h, w, _ = frame.shape
                for id_idx, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append([id_idx, cx, cy])
        return pixel_landmarks
