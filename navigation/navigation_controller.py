"""
Unified Navigation Controller coordinating focus management, action dispatching,
and multi-modal input synchronization (gestures + mouse).
"""

from typing import Dict, Any, Optional

from app.state_manager import ApplicationState, StateManager
from navigation.focus_manager import FocusManager
from navigation.action_dispatcher import ActionDispatcher
from voice.speech_manager import SpeechManager


class NavigationController:
    """
    Central navigation coordinator.
    Ensures mouse clicks/hover and hand gestures share one focused state.
    """

    MAIN_MENU_ITEMS = ["Start", "Gesture Setup", "Help", "Exit"]
    SETUP_MENU_ITEMS = ["Collect Dataset", "Train Model", "Back"]

    def __init__(self, state_manager: StateManager, speech_manager: SpeechManager,
                 app_callbacks: Dict[str, Any]):
        self.state_manager = state_manager
        self.speech_manager = speech_manager
        self.focus_manager = FocusManager(self.MAIN_MENU_ITEMS)
        self.action_dispatcher = ActionDispatcher(state_manager, speech_manager, app_callbacks)

        # Wire state listener to update active menu items
        self.state_manager.add_listener(self._on_state_change)

        self.last_intent: str = ""

    def _on_state_change(self, old_state: ApplicationState, new_state: ApplicationState):
        """Synchronizes focus manager items when application state changes."""
        if new_state == ApplicationState.MAIN_MENU:
            self.focus_manager.set_items(self.MAIN_MENU_ITEMS, reset_index=True)
        elif new_state == ApplicationState.GESTURE_SETUP:
            self.focus_manager.set_items(self.SETUP_MENU_ITEMS, reset_index=True)

    # ------------------------------------------------------------------
    # Intent / Gesture Processing
    # ------------------------------------------------------------------

    def process_intent(self, intent: str, payload: Optional[str] = None):
        """Processes normalized intent from gesture controller or test harness."""
        self.last_intent = intent
        state = self.state_manager.get_state()

        # STOP_VOICE: Highest priority global action
        if intent == "STOP_VOICE":
            self.action_dispatcher.dispatch_stop_voice()
            return

        if state in (ApplicationState.STARTUP, ApplicationState.WAITING_FOR_HAND):
            self.state_manager.set_state(ApplicationState.MAIN_MENU)
            self.speech_manager.speak("Main menu.")
            return

        if state == ApplicationState.MAIN_MENU:
            if intent == "NEXT":
                item = self.focus_manager.move_next()
                if item:
                    self.speech_manager.speak(item)
            elif intent == "PREV":
                item = self.focus_manager.move_prev()
                if item:
                    self.speech_manager.speak(item)
            elif intent == "SELECT":
                focused = self.focus_manager.get_focused_item()
                if focused:
                    self.action_dispatcher.dispatch_main_menu_action(focused)
            elif intent == "BACK":
                self.speech_manager.speak("Main menu.")

        elif state == ApplicationState.GESTURE_SETUP:
            if intent == "NEXT":
                item = self.focus_manager.move_next()
                if item:
                    self.speech_manager.speak(item)
            elif intent == "PREV":
                item = self.focus_manager.move_prev()
                if item:
                    self.speech_manager.speak(item)
            elif intent == "SELECT":
                focused = self.focus_manager.get_focused_item()
                if focused:
                    self.action_dispatcher.dispatch_setup_menu_action(focused)
            elif intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.DATASET_MENU:
            if intent == "SELECT":
                self.action_dispatcher.dispatch_dataset_menu_action()
            elif intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.DATASET_COLLECTING:
            if intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.TRAINING_CONFIRMATION:
            if intent == "SELECT":
                self.action_dispatcher.dispatch_training_confirmation()
            elif intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.MODEL_READY:
            if intent == "SELECT":
                self.action_dispatcher.dispatch_main_menu_action("Start")
            elif intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.RECOGNITION:
            if intent == "BACK":
                self.action_dispatcher.dispatch_back()

        elif state == ApplicationState.HELP:
            if intent == "BACK":
                self.action_dispatcher.dispatch_back()

    # ------------------------------------------------------------------
    # Mouse Interactions (Unifies mouse and gestures into the same FocusManager)
    # ------------------------------------------------------------------

    def handle_mouse_hover(self, item_name: str):
        """Mouse hover updates the focus indicator without activating the option."""
        state = self.state_manager.get_state()
        if state in (ApplicationState.MAIN_MENU, ApplicationState.GESTURE_SETUP):
            self.focus_manager.set_focus(item_name)

    def handle_mouse_click(self, item_name: str):
        """Mouse click sets focus and immediately activates the option."""
        state = self.state_manager.get_state()
        self.focus_manager.set_focus(item_name)

        if state == ApplicationState.MAIN_MENU:
            self.action_dispatcher.dispatch_main_menu_action(item_name)
        elif state == ApplicationState.GESTURE_SETUP:
            self.action_dispatcher.dispatch_setup_menu_action(item_name)

    # ------------------------------------------------------------------
    # Guidance / Option descriptions
    # ------------------------------------------------------------------

    @staticmethod
    def get_option_description(selection: Optional[str] = None) -> Dict[str, str]:
        """Provides high-contrast accessibility descriptions for UI display cards."""
        desc_map = {
            "Start": {
                "title": "Start",
                "description": "Enter live gesture recognition and navigation.",
                "trigger": "3 fingers to select"
            },
            "Gesture Setup": {
                "title": "Gesture Setup",
                "description": "Collect gesture samples and train your model.",
                "trigger": "3 fingers to select"
            },
            "Help": {
                "title": "Help",
                "description": "Learn the gesture controls.",
                "trigger": "3 fingers to select"
            },
            "Exit": {
                "title": "Exit",
                "description": "Close the application.",
                "trigger": "3 fingers to select"
            },
            "Collect Dataset": {
                "title": "Collect Dataset",
                "description": "Record gesture samples for AI training.",
                "trigger": "3 fingers to select"
            },
            "Train Model": {
                "title": "Train Model",
                "description": "Train the gesture recognition model.",
                "trigger": "3 fingers to select"
            },
            "Back": {
                "title": "Back",
                "description": "Return to the previous menu.",
                "trigger": "3 fingers or fist"
            }
        }
        if selection and selection in desc_map:
            return desc_map[selection]
        return {
            "title": selection or "",
            "description": "",
            "trigger": "3 fingers to select"
        }
