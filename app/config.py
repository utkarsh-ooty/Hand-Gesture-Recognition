"""
Centralized configuration for Hand Gesture & Voice Navigation AI.
Manages absolute path resolution, design tokens, gestures, and hardware settings.
"""

import os
import platform


def get_safe_path(path_str: str) -> str:
    """Return a Windows short path if on Windows and file exists, otherwise standard path."""
    if platform.system() == "Windows" and os.path.exists(path_str):
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            buf = ctypes.create_unicode_buffer(1024)
            kernel32.GetShortPathNameW(path_str, buf, 1024)
            if buf.value:
                return buf.value
        except Exception:
            pass
    return path_str


class Config:
    """Central configuration container."""

    # Project directories
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = os.path.join(BASE_DIR, "data")
    CONFIG_DIR = os.path.join(DATA_DIR, "config")

    # Canonical dataset paths in data/
    DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
    DATASET_CSV_PATH = os.path.join(DATASETS_DIR, "keypoint_dataset.csv")

    # Canonical model paths in data/
    MODELS_DIR = os.path.join(DATA_DIR, "models")
    HAND_LANDMARKER_PATH = os.path.join(MODELS_DIR, "hand_landmarker.task")
    RF_MODEL_PATH = os.path.join(MODELS_DIR, "rf_model.pkl")
    CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "confusion_matrix.png")

    # Gesture definitions
    GESTURES = [
        "open_palm",
        "closed_fist",
        "thumbs_up",
        "thumbs_down"
    ]

    # Canonical Gesture Mapping:
    # 0 fingers (closed fist)     -> BACK / Cancel
    # 1 finger  (index extended)   -> NEXT option
    # 2 fingers (index + middle)   -> PREVIOUS option
    # 3 fingers (3 fingers)        -> SELECT focused option
    # 5 fingers (open palm)        -> STOP_VOICE (Immediate speech interruption)
    FINGER_MAPPING = {
        0: "BACK",
        1: "NEXT",
        2: "PREV",
        3: "SELECT",
        5: "STOP_VOICE"
    }

    # Camera hardware settings
    CAMERA_INDEX = 0
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    FRAME_RATE_MS = 33  # ~30 FPS

    # Navigation debounce & timing
    STABILITY_FRAMES = 6
    COOLDOWN_SECONDS = 0.4
    FAST_INTERRUPT_FRAMES = 2

    # Speech settings
    SPEECH_QUEUE_MAXSIZE = 3
    SPEECH_DEDUP_WINDOW = 1.0  # seconds
    SAPI_RATE = 2
    PYTTSX3_RATE = 200

    # Accessibility Design tokens
    BG_DARK = "#0F172A"
    BG_PANEL = "#1E293B"
    BG_CARD = "#334155"
    BG_FOCUS = "#2563EB"
    FG_PRIMARY = "#F8FAFC"
    FG_SECONDARY = "#94A3B8"
    FG_ACCENT = "#38BDF8"
    FG_WARN = "#FBBF24"
    FG_SUCCESS = "#34D399"
    FG_ERROR = "#F87171"
    FONT_FAMILY = "Segoe UI"
