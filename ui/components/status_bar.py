"""
Status Bar component rendering the status banner, camera health, and voice announcements.
"""

import tkinter as tk
from typing import Optional

from ui.styles import (
    BG_DARK, BG_CARD, FG_WARN, FG_PRIMARY, FG_SUCCESS, FG_ERROR,
    FG_VOICE_BUBBLE, FG_SECONDARY, FONT_BANNER, FONT_STATUS_SMALL, FONT_VOICE
)


class StatusBar:
    """Encapsulates status banners, camera status indicator, and speech bubbles."""

    def __init__(self, parent_status: tk.Widget, parent_cam: tk.Widget, parent_voice: tk.Widget):
        # 1. Status Banner
        self.status_banner = tk.Label(
            parent_status,
            text="Starting...",
            font=FONT_BANNER,
            bg=BG_CARD,
            fg=FG_WARN,
            padx=14,
            pady=10,
            anchor="w"
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 10))

        # 2. Camera Status Indicator
        self.cam_status_lbl = tk.Label(
            parent_cam,
            text="● Starting...",
            font=FONT_STATUS_SMALL,
            fg=FG_WARN,
            bg=BG_DARK
        )
        self.cam_status_lbl.pack(anchor="w")

        # 3. Voice Bubble
        self.voice_bubble = tk.Label(
            parent_voice,
            text='Voice: "Welcome."',
            font=FONT_VOICE,
            fg=FG_VOICE_BUBBLE,
            bg=BG_DARK,
            anchor="w",
            wraplength=500,
            justify="left"
        )
        self.voice_bubble.pack(fill=tk.X, pady=(4, 0), anchor="w")

    def set_banner(self, text: str, bg: str, fg: str):
        self.status_banner.config(text=text, bg=bg, fg=fg)

    def set_camera_status(self, is_running: bool, hand_present: bool):
        if is_running and hand_present:
            self.cam_status_lbl.config(text="● LIVE — Hand detected", fg=FG_SUCCESS)
        elif is_running:
            self.cam_status_lbl.config(text="● LIVE — No hand", fg=FG_WARN)
        else:
            self.cam_status_lbl.config(text="○ Camera off", fg=FG_ERROR)

    def set_voice_text(self, text: str, is_interrupted: bool = False):
        if is_interrupted:
            self.voice_bubble.config(text="Voice: [INTERRUPTED]", fg=FG_ERROR)
            return

        clean = (text or "").strip()
        if len(clean) > 80:
            clean = clean[:77] + "..."

        if clean:
            self.voice_bubble.config(text=f'Voice: "{clean}"', fg=FG_VOICE_BUBBLE)
        else:
            self.voice_bubble.config(text="Voice: Ready", fg=FG_SECONDARY)
