"""
Action Dispatcher handling execution of activated menu commands and state actions.
"""

from typing import Dict, Callable, Any, Optional

from app.state_manager import ApplicationState, StateManager
from voice.speech_manager import SpeechManager
from app.config import Config


class ActionDispatcher:
    """Dispatches actions triggered by user activations (gesture Select or mouse Click)."""

    def __init__(self, state_manager: StateManager, speech_manager: SpeechManager,
                 app_callbacks: Dict[str, Callable[[], Any]]):
        self.state_manager = state_manager
        self.speech_manager = speech_manager
        self.app_callbacks = app_callbacks

    def announce(self, message: str, force: bool = False):
        """Announces a message via speech manager."""
        self.speech_manager.speak(message, force=force)

    def dispatch_stop_voice(self):
        """Global speech interruption."""
        self.speech_manager.stop_speaking()

    def dispatch_back(self):
        """Handles Back command (Fist gesture or Back button)."""
        state = self.state_manager.get_state()

        if state == ApplicationState.HELP:
            self._go_main_menu()
        elif state == ApplicationState.GESTURE_SETUP:
            self._go_main_menu()
        elif state == ApplicationState.DATASET_MENU:
            self.state_manager.set_state(ApplicationState.GESTURE_SETUP)
            self.announce("Gesture Setup.")
        elif state == ApplicationState.DATASET_COLLECTING:
            stop_cb = self.app_callbacks.get("stop_recording")
            if stop_cb:
                stop_cb()
            self._go_main_menu()
        elif state == ApplicationState.TRAINING_CONFIRMATION:
            self.state_manager.set_state(ApplicationState.GESTURE_SETUP)
            self.announce("Gesture Setup.")
        elif state == ApplicationState.MODEL_READY:
            self._go_main_menu()
        elif state == ApplicationState.RECOGNITION:
            stop_rec = self.app_callbacks.get("stop_recognition")
            if stop_rec:
                stop_rec()
            self._go_main_menu()
        elif state == ApplicationState.MAIN_MENU:
            self.announce("Main menu.")

    def dispatch_main_menu_action(self, item: str):
        """Activates a selected main menu option."""
        if item == "Start":
            self._start_recognition()
        elif item == "Gesture Setup":
            self.state_manager.set_state(ApplicationState.GESTURE_SETUP)
            self.announce("Gesture Setup.")
        elif item == "Help":
            self.state_manager.set_state(ApplicationState.HELP)
            self.announce("Help. One finger next. Two fingers previous. Three fingers select. Fist goes back. Open palm stops speech.")
        elif item == "Exit":
            self.announce("Goodbye.")
            close_cb = self.app_callbacks.get("close_app")
            if close_cb:
                close_cb()

    def dispatch_setup_menu_action(self, item: str):
        """Activates a selected gesture setup menu option."""
        if item == "Collect Dataset":
            self.state_manager.set_state(ApplicationState.DATASET_MENU)
            self.announce("Dataset collection. Select to begin.")
        elif item == "Train Model":
            missing_cb = self.app_callbacks.get("get_missing_samples")
            if missing_cb and missing_cb():
                self.announce("Samples missing. Collect data first.")
                return
            self.state_manager.set_state(ApplicationState.TRAINING_CONFIRMATION)
            self.announce("Train model? Select to confirm. Fist to cancel.")
        elif item == "Back":
            self._go_main_menu()

    def dispatch_dataset_menu_action(self, item: Optional[str] = None):
        """Begins dataset collection sequence."""
        self.state_manager.set_state(ApplicationState.DATASET_COLLECTING)
        self.announce("Recording started.")
        start_cb = self.app_callbacks.get("start_dataset_collection")
        if start_cb:
            start_cb()

    def dispatch_training_confirmation(self):
        """Executes model training."""
        self.state_manager.set_state(ApplicationState.TRAINING)
        self.announce("Training. Please wait.")
        train_cb = self.app_callbacks.get("train_model")
        success = train_cb() if train_cb else False
        if success:
            self.state_manager.set_state(ApplicationState.MODEL_READY)
            self.announce("Training complete.")
        else:
            self._go_main_menu()
            self.announce("Training failed. Check dataset.")

    def _start_recognition(self):
        is_trained_cb = self.app_callbacks.get("is_model_trained")
        if is_trained_cb and not is_trained_cb():
            self.announce("No model. Go to Gesture Setup first.")
            return
        self.state_manager.set_state(ApplicationState.RECOGNITION)
        rec_cb = self.app_callbacks.get("start_recognition")
        if rec_cb:
            rec_cb()
        self.announce("Recognition active.")

    def _go_main_menu(self):
        self.state_manager.set_state(ApplicationState.MAIN_MENU)
        self.announce("Main menu.")
