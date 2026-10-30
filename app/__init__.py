"""
App package for Hand Gesture & Voice Navigation AI.
Coordinates configuration, state management, and high-level application control.
"""

from app.config import Config
from app.state_manager import ApplicationState, StateManager

__all__ = ["Config", "ApplicationState", "StateManager", "AppController"]


def __getattr__(name):
    if name == "AppController":
        from app.app_controller import AppController
        return AppController
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
