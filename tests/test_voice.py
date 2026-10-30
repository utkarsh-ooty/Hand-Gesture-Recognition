"""
Tests for voice package: SpeechQueue, SpeechInterruptionHandler, and SpeechManager.
"""

import time
import unittest

from voice.speech_queue import SpeechQueue
from voice.interruption import SpeechInterruptionHandler
from voice.speech_manager import SpeechManager


class TestVoicePackage(unittest.TestCase):

    def test_queue_deduplication(self):
        sq = SpeechQueue(maxsize=3, dedup_window=1.0)
        self.assertTrue(sq.push("Hello"))
        # Immediate duplicate should be rejected
        self.assertFalse(sq.push("Hello"))
        # Force pushes despite duplicate
        self.assertTrue(sq.push("Hello", force=True))

    def test_queue_drains_stale(self):
        sq = SpeechQueue(maxsize=3)
        sq.push("First")
        sq.push("Second")
        # Queue should only retain the latest message
        self.assertEqual(sq.qsize(), 1)
        self.assertEqual(sq.pop(), "Second")

    def test_interruption_handler(self):
        handler = SpeechInterruptionHandler(visual_feedback_duration=0.5)
        self.assertFalse(handler.is_triggered())
        self.assertFalse(handler.is_recently_interrupted)

        handler.trigger()
        self.assertTrue(handler.is_triggered())
        self.assertTrue(handler.is_recently_interrupted)

        handler.reset()
        self.assertFalse(handler.is_triggered())
        # Still recently interrupted within the visual duration window
        self.assertTrue(handler.is_recently_interrupted)

    def test_speech_manager_speak_and_interrupt(self):
        sm = SpeechManager()
        sm.speak("Testing modular speech manager", force=True)
        time.sleep(0.1)
        sm.stop_speaking()
        self.assertFalse(sm.is_speaking)
        sm.shutdown()


if __name__ == "__main__":
    unittest.main()
