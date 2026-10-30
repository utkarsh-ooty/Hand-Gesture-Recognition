"""
Navigation package for Hand Gesture AI.
Unifies input from mouse and hand gestures into a single focus manager and action dispatcher.
"""

from navigation.focus_manager import FocusManager
from navigation.action_dispatcher import ActionDispatcher
from navigation.navigation_controller import NavigationController

__all__ = [
    "FocusManager",
    "ActionDispatcher",
    "NavigationController",
]
