"""
Accessible navigation button with high contrast, focus styling, and mouse hover/click handlers.
"""

import tkinter as tk
from typing import Callable, Optional

from ui.styles import (
    BG_PANEL, BG_FOCUS, FG_PRIMARY, FG_SECONDARY,
    FONT_MENU_ITEM, FONT_MENU_ITEM_FOCUSED
)


class NavigationButton:
    """Accessible menu button widget that visually distinguishes focused vs unselected state."""

    def __init__(self, parent: tk.Widget, text: str,
                 on_click: Optional[Callable[[str], None]] = None,
                 on_hover: Optional[Callable[[str], None]] = None):
        self.text = text
        self.on_click = on_click
        self.on_hover = on_hover
        self.is_focused = False

        self.widget = tk.Label(
            parent,
            text=f"   {self.text}",
            font=FONT_MENU_ITEM,
            fg=FG_SECONDARY,
            bg=BG_PANEL,
            anchor="w",
            padx=16,
            pady=10,
            cursor="hand2",
            relief="flat",
            bd=0
        )
        self.widget.pack(fill=tk.X, pady=3, padx=4)

        self.widget.bind("<Button-1>", self._handle_click)
        self.widget.bind("<Enter>", self._handle_hover)

    def _handle_click(self, event):
        if self.on_click:
            self.on_click(self.text)

    def _handle_hover(self, event):
        if self.on_hover:
            self.on_hover(self.text)

    def set_focused(self, focused: bool):
        """Updates styling based on focus state."""
        self.is_focused = focused
        if focused:
            self.widget.config(
                text=f" ▶  {self.text}",
                font=FONT_MENU_ITEM_FOCUSED,
                fg=FG_PRIMARY,
                bg=BG_FOCUS
            )
        else:
            self.widget.config(
                text=f"   {self.text}",
                font=FONT_MENU_ITEM,
                fg=FG_SECONDARY,
                bg=BG_PANEL
            )

    def destroy(self):
        self.widget.destroy()
