"""
Hand Gesture & Voice Navigation AI — Application Entry Point
=============================================================
Accessibility-first desktop application for gesture-based navigation.
Supports multi-modal input (mouse hover/click and hand gestures via webcam)
with interruptible voice output.
"""

import os
import sys
import platform
import tkinter as tk

# Suppress MediaPipe & TensorFlow C++ runtime noise
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Ensure repository root is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# Windows short-path safety for MediaPipe C++ backend
if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        buf = ctypes.create_unicode_buffer(1024)
        kernel32.GetShortPathNameW(SCRIPT_DIR, buf, 1024)
        if buf.value:
            os.chdir(buf.value)
        else:
            os.chdir(SCRIPT_DIR)
    except Exception:
        os.chdir(SCRIPT_DIR)
else:
    os.chdir(SCRIPT_DIR)

from app.app_controller import AppController

# Backwards-compatibility alias
HandGestureApp = AppController


def main():
    root = tk.Tk()
    app = AppController(root)
    root.mainloop()


if __name__ == "__main__":
    main()
