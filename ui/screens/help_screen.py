"""
Accessible Help Screen explaining gesture controls.
"""

import tkinter as tk
from ui.styles import BG_PANEL, FG_ACCENT, FG_PRIMARY, FONT_SECTION_TITLE, FONT_BODY


class HelpScreen:
    """Help screen explaining gesture navigation controls with high-contrast presentation."""

    def __init__(self, parent: tk.Widget):
        self.container = tk.Frame(parent, bg=BG_PANEL, padx=14, pady=12)

        tk.Label(
            self.container, text="Help & Controls",
            font=FONT_SECTION_TITLE, fg=FG_ACCENT, bg=BG_PANEL
        ).pack(anchor="w", pady=(0, 6))

        instructions = (
            "☝  1 Finger (Index extended)    → Next Option\n"
            "✌  2 Fingers (Index + Middle)   → Previous Option\n"
            "🤟  3 Fingers                   → Select Focused Option\n"
            "✊  Closed Fist                 → Back / Cancel\n"
            "🖐  Open Palm (All 5 Fingers)   → STOP SPEECH (Instant Mute)\n\n"
            "• Mouse navigation: Hover over any option to highlight it.\n"
            "• Click any option to activate it immediately."
        )

        self.text_lbl = tk.Label(
            self.container,
            text=instructions,
            font=FONT_BODY,
            fg=FG_PRIMARY,
            bg=BG_PANEL,
            justify="left"
        )
        self.text_lbl.pack(anchor="w", pady=4)
