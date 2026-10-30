"""
Tests for ApplicationState and StateManager.
"""

import unittest
from app.state_manager import ApplicationState, StateManager


class TestStateManager(unittest.TestCase):

    def test_initial_state(self):
        sm = StateManager()
        self.assertEqual(sm.get_state(), ApplicationState.STARTUP)

    def test_state_transition(self):
        sm = StateManager(ApplicationState.STARTUP)
        sm.set_state(ApplicationState.MAIN_MENU)
        self.assertEqual(sm.get_state(), ApplicationState.MAIN_MENU)

    def test_listener_notification(self):
        sm = StateManager(ApplicationState.STARTUP)
        history = []

        def listener(old_state, new_state):
            history.append((old_state, new_state))

        sm.add_listener(listener)
        sm.set_state(ApplicationState.WAITING_FOR_HAND)
        sm.set_state(ApplicationState.MAIN_MENU)

        self.assertEqual(len(history), 2)
        self.assertEqual(history[0], (ApplicationState.STARTUP, ApplicationState.WAITING_FOR_HAND))
        self.assertEqual(history[1], (ApplicationState.WAITING_FOR_HAND, ApplicationState.MAIN_MENU))

    def test_remove_listener(self):
        sm = StateManager(ApplicationState.STARTUP)
        history = []

        def listener(old_state, new_state):
            history.append(new_state)

        sm.add_listener(listener)
        sm.remove_listener(listener)
        sm.set_state(ApplicationState.MAIN_MENU)
        self.assertEqual(len(history), 0)


if __name__ == "__main__":
    unittest.main()
