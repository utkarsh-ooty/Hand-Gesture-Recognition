"""
High-level Speech Manager coordinating TTS engines, queues, and interruptions.
Accessible, interruptible text-to-speech assistant.
"""

import threading
import time
from typing import Optional

from voice.speech_engine import BaseSpeechEngine, create_speech_engine
from voice.speech_queue import SpeechQueue
from voice.interruption import SpeechInterruptionHandler


class SpeechManager:
    """
    Central speech manager.
    Runs speech synthesis off the Tkinter main thread in a dedicated daemon worker.
    Supports instant open-palm interruption and queue bounded freshness.
    """

    def __init__(self, sapi_rate: int = 2, pyttsx3_rate: int = 200, queue_maxsize: int = 3):
        self._sapi_rate = sapi_rate
        self._pyttsx3_rate = pyttsx3_rate
        self.queue = SpeechQueue(maxsize=queue_maxsize)
        self.interruption = SpeechInterruptionHandler()

        self._lock = threading.Lock()
        self._is_speaking = False
        self._running = True

        self._engine: Optional[BaseSpeechEngine] = None
        self._worker_thread = threading.Thread(
            target=self._speech_worker, daemon=True, name="SpeechWorker"
        )
        self._worker_thread.start()

    @property
    def speech_queue(self) -> SpeechQueue:
        return self.queue

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    @property
    def last_spoken_text(self) -> str:
        return self.queue.last_spoken_text

    @property
    def is_recently_interrupted(self) -> bool:
        return self.interruption.is_recently_interrupted

    def _speech_worker(self):
        """Worker thread entry point."""
        self._engine = create_speech_engine(self._sapi_rate, self._pyttsx3_rate)

        while self._running:
            text = self.queue.pop(timeout=0.1)
            if text is None:
                continue

            if not self._running:
                break

            # If an interruption occurred before starting this utterance, skip
            if self.interruption.is_triggered():
                self.interruption.reset()
                continue

            clean_text = text.strip()
            if not clean_text:
                continue

            with self._lock:
                self._is_speaking = True

            try:
                if self._engine:
                    self._engine.speak(clean_text)

                    # Poll engine until done or interrupted
                    while self._running and not self.interruption.is_triggered():
                        if self._engine.wait_until_done(40):
                            break

                    if self.interruption.is_triggered():
                        self._engine.stop()
                        self.interruption.reset()

            except Exception as e:
                print(f"[SpeechManager] Error speaking '{clean_text}': {e}")
            finally:
                with self._lock:
                    self._is_speaking = False

    def speak(self, message: str, force: bool = False, priority: int = 0):
        """Queues a message to speak, deduplicating repetitive announcements."""
        if not self._running or not message:
            return
        self.queue.push(message, force=force)

    def stop_speaking(self):
        """Immediately cancels ongoing speech and purges the speech queue."""
        self.interruption.trigger()
        self.clear_queue()
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass
        with self._lock:
            self._is_speaking = False

    def interrupt(self):
        """Alias for stop_speaking()."""
        self.stop_speaking()

    def clear_queue(self):
        """Clears all pending announcements in the queue."""
        self.queue.clear()

    def shutdown(self):
        """Gracefully shuts down speech engines and worker threads."""
        self._running = False
        self.stop_speaking()
        if self._engine:
            try:
                self._engine.close()
            except Exception:
                pass

    def close(self):
        """Alias for shutdown()."""
        self.shutdown()
