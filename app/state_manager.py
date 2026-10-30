"""
State management for Hand Gesture & Voice Navigation AI.
Tracks application states and notifies registered listeners on state changes.
"""

from enum import Enum
from typing import Callable, List


class ApplicationState(Enum):
    STARTUP = "STARTUP"
    WAITING_FOR_HAND = "WAITING_FOR_HAND"
    MAIN_MENU = "MAIN_MENU"
    GESTURE_SETUP = "GESTURE_SETUP"
    DATASET_MENU = "DATASET_MENU"
    DATASET_COLLECTING = "DATASET_COLLECTING"
    TRAINING_CONFIRMATION = "TRAINING_CONFIRMATION"
    TRAINING = "TRAINING"
    MODEL_READY = "MODEL_READY"
    RECOGNITION = "RECOGNITION"
    HELP = "HELP"
    SETTINGS = "SETTINGS"
    ERROR = "ERROR"
    EXIT = "EXIT"


class StateManager:
    """Thread-safe state manager with change notifications."""

    def __init__(self, initial_state: ApplicationState = ApplicationState.STARTUP):
        self._state = initial_state
        self._listeners: List[Callable[[ApplicationState, ApplicationState], None]] = []

    def get_state(self) -> ApplicationState:
        return self._state

    def set_state(self, new_state: ApplicationState):
        old_state = self._state
        if old_state != new_state:
            self._state = new_state
            for listener in self._listeners:
                try:
                    listener(old_state, new_state)
                except Exception as e:
                    print(f"[StateManager] Listener error: {e}")

    def add_listener(self, listener: Callable[[ApplicationState, ApplicationState], None]):
        """Register a callback when state changes: callback(old_state, new_state)."""
        if listener not in self._listeners:
            self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[ApplicationState, ApplicationState], None]):
        if listener in self._listeners:
            self._listeners.remove(listener)
