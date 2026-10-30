"""
Gesture Mapper translating raw gesture detections into canonical application events.
"""

from typing import Optional, Dict
from app.config import Config


class GestureMapper:
    """Translates finger counts, pose states, and ML predictions into navigation intents."""

    def __init__(self, finger_mapping: Optional[Dict[int, str]] = None):
        self.finger_mapping = finger_mapping or Config.FINGER_MAPPING

    def map_finger_count(self, count: int) -> Optional[str]:
        """Maps an extended finger count to a canonical intent."""
        return self.finger_mapping.get(count)

    @staticmethod
    def map_ml_gesture_to_intent(gesture_name: str) -> Optional[str]:
        """Maps an ML-recognized gesture to navigation intent if applicable."""
        mapping = {
            "open_palm": "STOP_VOICE",
            "closed_fist": "BACK",
            "thumbs_up": "SELECT",
            "thumbs_down": "BACK"
        }
        return mapping.get(gesture_name)
