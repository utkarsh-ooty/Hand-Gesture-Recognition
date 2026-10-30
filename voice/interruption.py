"""
Speech interruption coordinator for open-palm gestures and fast cancel.
"""

import threading
import time


class SpeechInterruptionHandler:
    """Manages thread-safe interruption flags and timing for visual indicators."""

    def __init__(self, visual_feedback_duration: float = 1.5):
        self._interrupted_event = threading.Event()
        self.visual_feedback_duration = visual_feedback_duration
        self.last_interrupted_time = 0.0

    def trigger(self):
        """Signals an immediate interruption."""
        self._interrupted_event.set()
        self.last_interrupted_time = time.time()

    def is_triggered(self) -> bool:
        return self._interrupted_event.is_set()

    def reset(self):
        self._interrupted_event.clear()

    @property
    def is_recently_interrupted(self) -> bool:
        """True for visual_feedback_duration after speech interruption."""
        return (time.time() - self.last_interrupted_time) < self.visual_feedback_duration
