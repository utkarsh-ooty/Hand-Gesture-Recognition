"""
Gesture Controller managing gesture debouncing, edge-triggering, and event dispatch.
"""

import time
from typing import Callable, Optional, List

from app.config import Config


class GestureController:
    """
    Stateful gesture debounce and edge-triggered intent router.
    Satisfies:
    - Open palm: Global highest priority, fast-path trigger, re-arm latch.
    - Edge-triggered navigation: Holding a static pose triggers once.
    - Transition-aware: Changing poses or neutral hand state resets latches.
    """

    def __init__(self, callback: Callable[[str], None],
                 stability_frames: int = Config.STABILITY_FRAMES,
                 cooldown: float = Config.COOLDOWN_SECONDS,
                 fast_interrupt_frames: int = Config.FAST_INTERRUPT_FRAMES):
        self.callback = callback
        self.stability_frames = stability_frames
        self.cooldown = cooldown
        self.fast_interrupt_frames = fast_interrupt_frames

        self.history: List[str] = []
        self.active_intent: Optional[str] = None
        self.last_trigger_time: float = 0.0

        # Open-palm specific debounce and re-arm
        self.open_palm_count: int = 0
        self.open_palm_held: bool = False

    def process_intent(self, intent: Optional[str]):
        """
        Receives intent token ("STOP_VOICE", "NEXT", "PREV", "SELECT", "BACK", or None).
        Evaluates stability, priority, and edge-triggers.
        """
        # Case 1: Hand lost or pose unrecognized
        if intent is None:
            self.history.clear()
            self.open_palm_count = 0
            self.open_palm_held = False
            self.active_intent = None
            return

        # Case 2: Open Palm (Speech Interruption) — Highest Priority Fast Path
        if intent == "STOP_VOICE":
            self.history.clear()
            self.active_intent = None  # Reset any navigation latch
            self.open_palm_count += 1

            if self.open_palm_count >= self.fast_interrupt_frames and not self.open_palm_held:
                self.open_palm_held = True
                self.callback("STOP_VOICE")
            return

        # Any non-open-palm gesture resets the open-palm re-arm state
        self.open_palm_count = 0
        self.open_palm_held = False

        # Case 3: User is holding the exact same pose that has already fired
        if intent == self.active_intent:
            self.history.clear()
            return

        # Case 4: Accumulate new candidate intent in stability buffer
        now = time.time()
        self.history.append(intent)
        if len(self.history) > self.stability_frames:
            self.history.pop(0)

        if len(self.history) == self.stability_frames:
            if all(x == intent for x in self.history):
                # Candidate intent is stable
                if now - self.last_trigger_time >= self.cooldown:
                    self.active_intent = intent
                    self.last_trigger_time = now
                    self.callback(intent)
                    self.history.clear()

    def reset(self):
        """Resets all internal debounce buffers and states."""
        self.history.clear()
        self.open_palm_count = 0
        self.open_palm_held = False
        self.active_intent = None
        self.last_trigger_time = 0.0
