"""
Thread-safe bounded speech queue with deduplication and purging.
"""

import queue
import time
from typing import Optional


class SpeechQueue:
    """Manages outgoing speech messages, ensuring only fresh announcements are queued."""

    def __init__(self, maxsize: int = 3, dedup_window: float = 1.0):
        self._queue = queue.Queue(maxsize=maxsize)
        self.dedup_window = dedup_window
        self.last_spoken_text = ""
        self.last_spoken_time = 0.0

    def push(self, text: str, force: bool = False) -> bool:
        """Pushes a message. Returns True if queued, False if skipped/deduplicated."""
        if not text:
            return False

        now = time.time()
        # Deduplication check
        if not force and text.strip().lower() == self.last_spoken_text.strip().lower():
            if now - self.last_spoken_time < self.dedup_window:
                return False

        self.last_spoken_text = text
        self.last_spoken_time = now

        # Drain older messages to ensure we only speak the freshest state
        self.clear()

        try:
            self._queue.put_nowait(text)
            return True
        except queue.Full:
            return False

    def pop(self, timeout: float = 0.1) -> Optional[str]:
        """Pops the next message with a timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear(self):
        """Discards all pending messages."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()
