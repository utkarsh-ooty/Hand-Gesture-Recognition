"""
Tests for Hand Gesture AI — navigation, speech, and gesture debouncing.
Hardware-independent integration test harness migrated to tests/.
"""

import os
import sys
import time
import unittest

from app.state_manager import ApplicationState, StateManager
from voice.speech_manager import SpeechManager
from navigation.navigation_controller import NavigationController
from gestures.gesture_controller import GestureController
from gestures.finger_detector import FingerDetector


class MockSpeechManager:
    """No-audio stand-in that records calls."""

    def __init__(self):
        self.spoken = []
        self._is_speaking = False
        self.interrupted = False
        self.last_spoken_text = ""
        self.speech_interrupted_time = 0.0

    @property
    def is_speaking(self):
        return self._is_speaking

    def speak(self, text, force=False, priority=0):
        self.spoken.append(text)
        self.last_spoken_text = text

    def interrupt(self):
        self.interrupted = True
        self._is_speaking = False
        self.spoken.clear()
        self.speech_interrupted_time = time.time()

    def stop_speaking(self):
        self.interrupt()

    @property
    def is_recently_interrupted(self) -> bool:
        return (time.time() - self.speech_interrupted_time) < 1.5

    def close(self):
        pass

    def shutdown(self):
        pass


def _noop(*args, **kwargs):
    return True


def _make_controller(state_manager=None, tts=None):
    sm = state_manager or StateManager(ApplicationState.MAIN_MENU)
    va = tts or MockSpeechManager()
    cbs = {
        "start_camera": _noop,
        "stop_camera": _noop,
        "start_dataset_collection": _noop,
        "stop_recording": _noop,
        "train_model": lambda: True,
        "is_model_trained": lambda: True,
        "get_missing_samples": lambda: [],
        "start_recognition": _noop,
        "stop_recognition": _noop,
        "reload_model": _noop,
        "close_app": _noop,
    }
    return NavigationController(sm, va, cbs)


# ======================================================================
# 1. State transitions & menu navigation
# ======================================================================

class TestStateTransitions(unittest.TestCase):

    def test_startup_to_main_menu(self):
        sm = StateManager(ApplicationState.WAITING_FOR_HAND)
        ctrl = _make_controller(sm)
        ctrl.process_intent("STARTUP")
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)

    def test_next_prev_cycle_menu(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = _make_controller(sm)
        ctrl.focus_manager.set_focus("Start")

        ctrl.process_intent("NEXT")
        self.assertEqual(ctrl.focus_manager.get_focused_item(), "Gesture Setup")

        ctrl.process_intent("PREV")
        self.assertEqual(ctrl.focus_manager.get_focused_item(), "Start")

    def test_select_help(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = _make_controller(sm)
        ctrl.focus_manager.set_focus("Help")
        ctrl.process_intent("SELECT")
        self.assertEqual(sm.get_state(), ApplicationState.HELP)

    def test_back_from_help(self):
        sm = StateManager(ApplicationState.HELP)
        ctrl = _make_controller(sm)
        ctrl.process_intent("BACK")
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)

    def test_select_exit_calls_callback(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        closed = []
        cbs = {k: _noop for k in [
            "start_camera", "stop_camera", "start_dataset_collection",
            "stop_recording", "train_model", "is_model_trained",
            "get_missing_samples", "start_recognition", "stop_recognition",
            "reload_model",
        ]}
        cbs["close_app"] = lambda: closed.append(True)
        ctrl = NavigationController(sm, MockSpeechManager(), cbs)
        ctrl.focus_manager.set_focus("Exit")
        ctrl.process_intent("SELECT")
        self.assertTrue(closed)

    def test_gesture_setup_submenu(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = _make_controller(sm)
        ctrl.focus_manager.set_focus("Gesture Setup")
        ctrl.process_intent("SELECT")
        self.assertEqual(sm.get_state(), ApplicationState.GESTURE_SETUP)

        ctrl.process_intent("BACK")
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)

    def test_back_at_main_menu_stays(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = _make_controller(sm)
        ctrl.process_intent("BACK")
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)


# ======================================================================
# 2. Speech queue and interruption
# ======================================================================

class TestSpeechInterruption(unittest.TestCase):

    def test_stop_voice_clears_queue(self):
        va = MockSpeechManager()
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = NavigationController(sm, va, {k: _noop for k in [
            "start_camera", "stop_camera", "start_dataset_collection",
            "stop_recording", "train_model", "is_model_trained",
            "get_missing_samples", "start_recognition", "stop_recognition",
            "reload_model", "close_app",
        ]})

        va.speak("Message one")
        va.speak("Message two")
        ctrl.process_intent("STOP_VOICE")
        self.assertTrue(va.interrupted)

    def test_stop_voice_does_not_change_state(self):
        sm = StateManager(ApplicationState.MAIN_MENU)
        ctrl = _make_controller(sm)
        ctrl.process_intent("STOP_VOICE")
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)

    def test_speech_interrupted_flag_time_limited(self):
        va = MockSpeechManager()
        va.speech_interrupted_time = time.time()
        self.assertTrue(va.is_recently_interrupted)

        va.speech_interrupted_time = time.time() - 2.0
        self.assertFalse(va.is_recently_interrupted)


# ======================================================================
# 3. Navigation debouncing & open-palm re-arm
# ======================================================================

class TestNavigationDebouncing(unittest.TestCase):

    def test_stable_intent_fires_once(self):
        results = []
        gc = GestureController(lambda intent: results.append(intent),
                               stability_frames=3, cooldown=0.0)
        for _ in range(3):
            gc.process_intent("NEXT")
        self.assertEqual(results, ["NEXT"])

        # Holding same pose does NOT fire again
        for _ in range(10):
            gc.process_intent("NEXT")
        self.assertEqual(results, ["NEXT"])

    def test_open_palm_fast_path(self):
        results = []
        gc = GestureController(lambda intent: results.append(intent),
                               stability_frames=6, cooldown=0.0,
                               fast_interrupt_frames=2)
        gc.process_intent("STOP_VOICE")
        gc.process_intent("STOP_VOICE")
        self.assertEqual(results, ["STOP_VOICE"])

    def test_open_palm_rearm(self):
        results = []
        gc = GestureController(lambda intent: results.append(intent),
                               stability_frames=6, cooldown=0.0,
                               fast_interrupt_frames=2)
        # Trigger once
        gc.process_intent("STOP_VOICE")
        gc.process_intent("STOP_VOICE")
        self.assertEqual(len(results), 1)

        # Holding open palm does NOT re-trigger
        for _ in range(20):
            gc.process_intent("STOP_VOICE")
        self.assertEqual(len(results), 1)

        # Release (switch gesture or None), then re-show open palm
        gc.process_intent(None)
        gc.process_intent("STOP_VOICE")
        gc.process_intent("STOP_VOICE")
        self.assertEqual(len(results), 2)

    def test_transition_allows_new_intent(self):
        results = []
        gc = GestureController(lambda intent: results.append(intent),
                               stability_frames=2, cooldown=0.0)
        gc.process_intent("NEXT")
        gc.process_intent("NEXT")
        self.assertIn("NEXT", results)

        gc.process_intent("PREV")
        gc.process_intent("PREV")
        self.assertIn("PREV", results)

    def test_none_resets(self):
        results = []
        gc = GestureController(lambda intent: results.append(intent),
                               stability_frames=2, cooldown=0.0)
        # First trigger
        gc.process_intent("NEXT")
        gc.process_intent("NEXT")
        self.assertEqual(results.count("NEXT"), 1)

        # Hand lost resets active_intent
        gc.process_intent(None)

        # Second trigger (same gesture can fire again after reset)
        gc.process_intent("NEXT")
        gc.process_intent("NEXT")
        self.assertEqual(results.count("NEXT"), 2)


# ======================================================================
# 4. Finger detector
# ======================================================================

class TestFingerDetector(unittest.TestCase):
    """Sanity checks with synthetic landmarks."""

    def _make_landmarks(self, positions):
        return [[i, x, y, 0.0] for i, (x, y) in enumerate(positions)]

    def test_open_palm_detection(self):
        fd = FingerDetector()
        palm = [(0.5, 0.9)]  # wrist
        palm += [(0.35, 0.7), (0.3, 0.55), (0.25, 0.4), (0.15, 0.25)]  # thumb
        palm += [(0.4, 0.6), (0.4, 0.45), (0.4, 0.35), (0.4, 0.2)]      # index
        palm += [(0.5, 0.6), (0.5, 0.45), (0.5, 0.35), (0.5, 0.2)]      # middle
        palm += [(0.6, 0.6), (0.6, 0.45), (0.6, 0.35), (0.6, 0.2)]      # ring
        palm += [(0.7, 0.65), (0.7, 0.5), (0.7, 0.4), (0.7, 0.25)]     # pinky
        lm = self._make_landmarks(palm)
        self.assertTrue(fd.is_open_palm(lm))
        self.assertEqual(fd.get_navigation_intent(lm), "STOP_VOICE")


# ======================================================================
# 5. Voice assistant (real engine, quick smoke test)
# ======================================================================

class TestSpeechManagerReal(unittest.TestCase):

    def test_speak_and_interrupt(self):
        va = SpeechManager()
        va.speak("Testing one two three.", force=True)
        time.sleep(0.15)
        va.interrupt()
        time.sleep(0.1)
        self.assertFalse(va.is_speaking)
        va.shutdown()

    def test_deduplication(self):
        va = SpeechManager()
        va.speak("Same message")
        va.speak("Same message")  # Should be deduplicated
        time.sleep(0.05)
        self.assertTrue(va.speech_queue.qsize() <= 1)
        va.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
