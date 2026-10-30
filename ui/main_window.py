"""
Main Window for Hand Gesture & Voice Navigation AI.
Builds the GUI layout, renders webcam frames, and displays accessible high-contrast navigation elements.
"""

import tkinter as tk
import cv2
from PIL import Image, ImageTk
from typing import Callable, List, Optional
import numpy as np

from ui.styles import (
    BG_DARK, BG_PANEL, BG_CARD, FG_PRIMARY, FG_SECONDARY, FG_ACCENT,
    FONT_APP_TITLE, FONT_SECTION_TITLE, FONT_DESC_TITLE, FONT_DESC_BODY,
    FONT_LEGEND_TITLE, FONT_LEGEND_BODY
)
from ui.components.status_bar import StatusBar
from ui.screens.main_menu import MainMenuScreen


class MainWindow:
    """Main Tkinter window view."""

    def __init__(self, root: tk.Tk,
                 on_menu_click: Callable[[str], None],
                 on_menu_hover: Callable[[str], None],
                 on_close: Callable[[], None]):
        self.root = root
        self.root.title("Hand Gesture AI")
        self.root.geometry("1100x700")
        self.root.minsize(900, 600)
        self.root.configure(bg=BG_DARK)
        self.root.protocol("WM_DELETE_WINDOW", on_close)

        self.on_menu_click = on_menu_click
        self.on_menu_hover = on_menu_hover

        # Reference to prevent garbage collection of image
        self._current_photo_image: Optional[ImageTk.PhotoImage] = None

        self._build_layout()

    def _build_layout(self):
        # ── Left Panel: Camera & Gesture Legend ──
        self.left_frame = tk.Frame(self.root, bg=BG_DARK, width=360)
        self.left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(12, 6), pady=12)
        self.left_frame.pack_propagate(False)

        tk.Label(
            self.left_frame, text="Camera",
            font=FONT_SECTION_TITLE, fg=FG_ACCENT, bg=BG_DARK
        ).pack(anchor="w", pady=(0, 4))

        self.cam_status_container = tk.Frame(self.left_frame, bg=BG_DARK)
        self.cam_status_container.pack(anchor="w")

        self.video_frame = tk.Frame(
            self.left_frame, bg="#000", bd=2, relief="solid",
            highlightbackground=BG_CARD, highlightthickness=1
        )
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(4, 8))

        self.video_label = tk.Label(self.video_frame, bg="#000")
        self.video_label.pack(expand=True)

        # Legend frame
        legend_frame = tk.Frame(self.left_frame, bg=BG_PANEL, padx=10, pady=8)
        legend_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            legend_frame, text="Gesture Controls",
            font=FONT_LEGEND_TITLE, fg=FG_ACCENT, bg=BG_PANEL
        ).pack(anchor="w")

        legend_text = (
            "☝  1 finger  →  Next\n"
            "✌  2 fingers →  Previous\n"
            "🤟  3 fingers →  Select\n"
            "✊  Fist       →  Back\n"
            "🖐  Open palm →  Stop speech"
        )
        tk.Label(
            legend_frame, text=legend_text,
            font=FONT_LEGEND_BODY, fg=FG_PRIMARY, bg=BG_PANEL,
            justify="left"
        ).pack(anchor="w", pady=(2, 0))

        # ── Right Panel: Controls & Guidance ──
        self.right_frame = tk.Frame(self.root, bg=BG_DARK)
        self.right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(6, 12), pady=12)

        # Title
        tk.Label(
            self.right_frame, text="Hand Gesture AI",
            font=FONT_APP_TITLE, fg=FG_PRIMARY, bg=BG_DARK
        ).pack(anchor="w", pady=(0, 6))

        # Status container (for status banner)
        self.status_container = tk.Frame(self.right_frame, bg=BG_DARK)
        self.status_container.pack(fill=tk.X)

        # Menu Screen Container
        self.menu_screen = MainMenuScreen(
            self.right_frame,
            items=["Start", "Gesture Setup", "Help", "Exit"],
            on_click=self.on_menu_click,
            on_hover=self.on_menu_hover
        )

        # Description Card
        self.desc_card = tk.Frame(self.right_frame, bg=BG_PANEL, padx=14, pady=10)
        self.desc_card.pack(fill=tk.X, pady=(0, 10))

        self.desc_title = tk.Label(
            self.desc_card, text="",
            font=FONT_DESC_TITLE, fg=FG_PRIMARY, bg=BG_PANEL, anchor="w"
        )
        self.desc_title.pack(fill=tk.X)

        self.desc_body = tk.Label(
            self.desc_card, text="",
            font=FONT_DESC_BODY, fg=FG_SECONDARY, bg=BG_PANEL,
            anchor="w", wraplength=500, justify="left"
        )
        self.desc_body.pack(fill=tk.X, pady=(2, 0))

        # Voice bubble container
        self.voice_container = tk.Frame(self.right_frame, bg=BG_DARK)
        self.voice_container.pack(fill=tk.X)

        # Initialize StatusBar
        self.status_bar = StatusBar(
            parent_status=self.status_container,
            parent_cam=self.cam_status_container,
            parent_voice=self.voice_container
        )

    # ------------------------------------------------------------------
    # Rendering & View Updates (Main Thread Safe)
    # ------------------------------------------------------------------

    def render_camera(self, frame: Optional[np.ndarray]):
        """Renders an OpenCV BGR frame onto the Tkinter video label."""
        if frame is None:
            return

        try:
            panel_w = self.left_frame.winfo_width()
            if panel_w < 100:
                panel_w = 340

            target_w = max(280, panel_w - 20)
            orig_h, orig_w = frame.shape[:2]
            target_h = int(target_w * (orig_h / orig_w))

            panel_h = self.video_frame.winfo_height()
            if panel_h > 100 and target_h > panel_h - 10:
                target_h = panel_h - 10
                target_w = int(target_h * (orig_w / orig_h))

            disp = cv2.resize(
                frame, (target_w, target_h),
                interpolation=cv2.INTER_AREA if target_w < orig_w else cv2.INTER_LINEAR
            )
            img_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            self.video_label.imgtk = img_tk
            self.video_label.config(image=img_tk)
            self._current_photo_image = img_tk
        except Exception:
            pass

    def update_status_banner(self, text: str, bg: str, fg: str):
        self.status_bar.set_banner(text, bg, fg)

    def update_camera_status(self, is_running: bool, hand_present: bool):
        self.status_bar.set_camera_status(is_running, hand_present)

    def update_voice_bubble(self, text: str, is_interrupted: bool = False):
        self.status_bar.set_voice_text(text, is_interrupted)

    def update_menu_buttons(self, items: List[str]):
        self.menu_screen.rebuild(items)

    def update_menu_highlight(self, selected_name: str):
        self.menu_screen.update_focus(selected_name)

    def update_description(self, title: str, description: str):
        self.desc_title.config(text=title)
        self.desc_body.config(text=description)
