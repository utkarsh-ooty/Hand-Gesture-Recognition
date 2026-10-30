"""
Focus indicator managing highlight states across a collection of navigation buttons.
"""

from typing import Dict
from ui.components.navigation_button import NavigationButton


class FocusIndicator:
    """Coordinates focus rendering across active navigation buttons."""

    @staticmethod
    def update_focus(buttons: Dict[str, NavigationButton], selected_name: str):
        """Highlights the selected button and resets others."""
        for name, btn in buttons.items():
            btn.set_focused(name == selected_name)
