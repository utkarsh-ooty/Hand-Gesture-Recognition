"""
Tests for NavigationController, FocusManager, and multi-modal input sharing (mouse + gestures).
"""

import unittest
from app.state_manager import ApplicationState, StateManager
from navigation.focus_manager import FocusManager
from navigation.navigation_controller import NavigationController


class DummySpeechManager:
    """Mock speech manager for unit testing."""
    def __init__(self):
        self.spoken = []
        self.interrupted = False

    def speak(self, text, force=False, priority=0):
        self.spoken.append(text)

    def stop_speaking(self):
        self.interrupted = True
        self.spoken.clear()

    @property
    def is_speaking(self):
        return False

    @property
    def last_spoken_text(self):
        return self.spoken[-1] if self.spoken else ""

    @property
    def is_recently_interrupted(self):
        return self.interrupted


class TestNavigation(unittest.TestCase):

    def setUp(self):
        self.sm = StateManager(ApplicationState.MAIN_MENU)
        self.speech = DummySpeechManager()
        self.actions_invoked = []
        self.cbs = {
            "start_camera": lambda: True,
            "stop_camera": lambda: True,
            "start_dataset_collection": lambda: self.actions_invoked.append("start_dataset"),
            "stop_recording": lambda: self.actions_invoked.append("stop_recording"),
            "train_model": lambda: True,
            "is_model_trained": lambda: True,
            "get_missing_samples": lambda: [],
            "start_recognition": lambda: self.actions_invoked.append("start_recognition"),
            "stop_recognition": lambda: self.actions_invoked.append("stop_recognition"),
            "reload_model": lambda: True,
            "close_app": lambda: self.actions_invoked.append("close_app"),
        }
        self.nav = NavigationController(self.sm, self.speech, self.cbs)

    def test_focus_manager_cycling(self):
        fm = FocusManager(["A", "B", "C"])
        self.assertEqual(fm.get_focused_item(), "A")
        self.assertEqual(fm.move_next(), "B")
        self.assertEqual(fm.move_next(), "C")
        self.assertEqual(fm.move_next(), "A")  # Wraps around
        self.assertEqual(fm.move_prev(), "C")  # Wraps back

    def test_mouse_hover_updates_focus_without_activating(self):
        """Mouse hover must change focus without activating the option."""
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Start")
        self.nav.handle_mouse_hover("Help")
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Help")
        self.assertEqual(self.sm.get_state(), ApplicationState.MAIN_MENU)
        self.assertNotIn("close_app", self.actions_invoked)

    def test_mouse_click_sets_focus_and_activates(self):
        """Mouse click sets focus and activates option."""
        self.nav.handle_mouse_click("Help")
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Help")
        self.assertEqual(self.sm.get_state(), ApplicationState.HELP)

    def test_gesture_and_mouse_share_one_focus_state(self):
        """Both mouse and gestures update the exact same focus manager state."""
        self.nav.handle_mouse_hover("Gesture Setup")
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Gesture Setup")

        # Now send NEXT gesture intent
        self.nav.process_intent("NEXT")
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Help")

        # Now mouse hover back
        self.nav.handle_mouse_hover("Start")
        self.assertEqual(self.nav.focus_manager.get_focused_item(), "Start")

        # Gesture SELECT activates Start
        self.nav.process_intent("SELECT")
        self.assertIn("start_recognition", self.actions_invoked)
        self.assertEqual(self.sm.get_state(), ApplicationState.RECOGNITION)

    def test_open_palm_interrupts_speech_without_triggering_menu_action(self):
        """Open palm (STOP_VOICE) must immediately mute without activating any menu item."""
        self.speech.speak("Ongoing sentence.")
        initial_state = self.sm.get_state()
        initial_focus = self.nav.focus_manager.get_focused_item()

        self.nav.process_intent("STOP_VOICE")

        self.assertTrue(self.speech.interrupted)
        self.assertEqual(self.sm.get_state(), initial_state)
        self.assertEqual(self.nav.focus_manager.get_focused_item(), initial_focus)
        self.assertEqual(len(self.actions_invoked), 0)

    def test_closed_fist_back(self):
        """Fist gesture routes to BACK."""
        self.nav.handle_mouse_click("Gesture Setup")
        self.assertEqual(self.sm.get_state(), ApplicationState.GESTURE_SETUP)

        self.nav.process_intent("BACK")
        self.assertEqual(self.sm.get_state(), ApplicationState.MAIN_MENU)


if __name__ == "__main__":
    unittest.main()
