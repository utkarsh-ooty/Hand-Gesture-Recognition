"""
Rule-based finger detection and counting from normalized landmarks.
"""

import math
from typing import List, Dict, Optional, Tuple


class FingerDetector:
    """Detects extended fingers and identifies canonical navigational gestures."""

    def __init__(self):
        # Tip landmarks according to MediaPipe
        self.TIP_IDS = [4, 8, 12, 16, 20]   # Thumb, Index, Middle, Ring, Pinky
        self.PIP_IDS = [2, 6, 10, 14, 18]   # Joint references
        self.MCP_IDS = [1, 5, 9, 13, 17]

    @staticmethod
    def _get_distance(p1, p2) -> float:
        return math.hypot(p1[1] - p2[1], p1[2] - p2[2])

    def count_fingers(self, landmarks: List[List[float]],
                      handedness: str = "Right") -> Optional[Tuple[int, Dict[str, int]]]:
        """
        Takes normalized landmarks list [[id, x, y, z], ...]
        Returns tuple (total_count, details_dict) or None if invalid.
        """
        if not landmarks or len(landmarks) < 21:
            return None

        fingers_extended = [0, 0, 0, 0, 0]
        wrist = landmarks[0]

        # Calculate palm height (wrist to middle finger MCP) as reference unit
        mid_mcp = landmarks[self.MCP_IDS[2]]
        palm_size = self._get_distance(wrist, mid_mcp)
        if palm_size < 1e-4:
            palm_size = 0.1

        # Index, Middle, Ring, Pinky
        for i in range(1, 5):
            tip = landmarks[self.TIP_IDS[i]]
            pip = landmarks[self.PIP_IDS[i]]

            # Distance from wrist
            dist_tip_wrist = self._get_distance(tip, wrist)
            dist_pip_wrist = self._get_distance(pip, wrist)

            # Using a small margin allows for palm-side perspective shortening
            if dist_tip_wrist > dist_pip_wrist + (palm_size * 0.02):
                fingers_extended[i] = 1

        # Thumb
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        pinky_mcp = landmarks[17]

        if self._get_distance(thumb_tip, pinky_mcp) > self._get_distance(thumb_ip, pinky_mcp) * 1.05:
            fingers_extended[0] = 1

        details = {
            'thumb': fingers_extended[0],
            'index': fingers_extended[1],
            'middle': fingers_extended[2],
            'ring': fingers_extended[3],
            'pinky': fingers_extended[4]
        }
        return sum(fingers_extended), details

    def is_open_palm(self, landmarks: List[List[float]]) -> bool:
        """Returns True if open palm is detected (all 5 or 4 main fingers extended)."""
        res = self.count_fingers(landmarks)
        if not res:
            return False
        count, details = res
        if count == 5:
            return True
        if details['index'] == 1 and details['middle'] == 1 and details['ring'] == 1 and details['pinky'] == 1:
            return True
        return False

    def is_closed_fist(self, landmarks: List[List[float]]) -> bool:
        """Returns True if all non-thumb fingers are folded into a fist."""
        res = self.count_fingers(landmarks)
        if not res:
            return False
        count, details = res
        if count == 0:
            return True
        if details['index'] == 0 and details['middle'] == 0 and details['ring'] == 0 and details['pinky'] == 0:
            return True
        return False

    def get_navigation_intent(self, landmarks: List[List[float]],
                              handedness: str = "Right") -> Optional[str]:
        """
        Authoritative gesture-to-intent mapping:
        - Open palm (5 fingers): STOP_VOICE (Speech Interruption, highest priority)
        - 1 extended index finger: NEXT option
        - 2 extended fingers (index+middle): PREV option
        - 3 extended fingers: SELECT focused option
        - Closed fist (0 fingers): BACK or cancel
        """
        res = self.count_fingers(landmarks, handedness)
        if not res:
            return None

        count, details = res

        # 1. Open palm has absolute highest priority: speech interruption
        if self.is_open_palm(landmarks):
            return "STOP_VOICE"

        # 2. Closed fist: Back / cancel
        if self.is_closed_fist(landmarks):
            return "BACK"

        # 3. Exactly 1 finger (index extended, others folded)
        if details['index'] == 1 and details['middle'] == 0 and details['ring'] == 0 and details['pinky'] == 0:
            return "NEXT"

        # 4. Exactly 2 fingers (index + middle extended, ring + pinky folded)
        if details['index'] == 1 and details['middle'] == 1 and details['ring'] == 0 and details['pinky'] == 0:
            return "PREV"

        # 5. Exactly 3 fingers (index + middle + ring extended, pinky folded) or count == 3
        if (details['index'] == 1 and details['middle'] == 1 and details['ring'] == 1 and details['pinky'] == 0) or count == 3:
            return "SELECT"

        return None

    def get_navigation_token(self, landmarks: List[List[float]], handedness: str = "Right") -> Optional[str]:
        """Backwards compatibility helper."""
        return self.get_navigation_intent(landmarks, handedness)

    def check_gesture_match(self, gesture_name: str, landmarks: List[List[float]]) -> bool:
        """Returns True if landmarks match the requested gesture class (for dataset collection)."""
        if not landmarks or len(landmarks) < 21:
            return False

        res = self.count_fingers(landmarks)
        if res is None:
            return False
        count, _ = res

        if gesture_name == "open_palm":
            return self.is_open_palm(landmarks)
        elif gesture_name == "closed_fist":
            return self.is_closed_fist(landmarks)
        elif gesture_name == "thumbs_up":
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            return count <= 2 and thumb_tip[2] < wrist[2]
        elif gesture_name == "thumbs_down":
            wrist = landmarks[0]
            thumb_tip = landmarks[4]
            return count <= 2 and thumb_tip[2] > wrist[2]

        return False
