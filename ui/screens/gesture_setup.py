"""
Gesture Setup screen component.
"""

import tkinter as tk
from typing import List, Callable, Dict

from ui.styles import BG_PANEL, FG_ACCENT, FONT_SECTION_TITLE
from ui.components.navigation_button import NavigationButton
from ui.components.focus_indicator import FocusIndicator


class GestureSetupScreen:
    """Manages the sub-menu button widgets for Gesture Setup (Collect, Train, Back)."""

    def __init__(self, parent: tk.Widget, items: List[str],
                 on_click: Callable[[str], None],
                 on_hover: Callable[[str], None]):
        self.parent = parent
        self.on_click = on_click
        self.on_hover = on_hover

        self.container = tk.Frame(parent, bg=BG_PANEL, padx=12, pady=10)
        # Note: Managed by MainWindow/Screen coordinator

        self.title_lbl = tk.Label(
            self.container, text="Gesture Setup",
            font=FONT_SECTION_TITLE, fg=FG_ACCENT, bg=BG_PANEL
        )
        self.title_lbl.pack(anchor="w", pady=(0, 6))

        self.buttons: Dict[str, NavigationButton] = {}
        self.rebuild(items)

    def rebuild(self, items: List[str]):
        for btn in self.buttons.values():
            btn.destroy()
        self.buttons.clear()

        for item in items:
            self.buttons[item] = NavigationButton(
                self.container, item,
                on_click=self.on_click,
                on_hover=self.on_hover
            )

    def update_focus(self, focused_name: str):
        FocusIndicator.update_focus(self.buttons, focused_name)
