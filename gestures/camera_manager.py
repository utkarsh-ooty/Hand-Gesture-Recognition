"""
Camera lifecycle and frame capture manager using OpenCV.
"""

import cv2
from typing import Optional
import numpy as np

from app.config import Config


class CameraManager:
    """Manages OpenCV webcam initialization, frame capture, and release."""

    def __init__(self, camera_index: int = Config.CAMERA_INDEX,
                 width: int = Config.CAMERA_WIDTH,
                 height: int = Config.CAMERA_HEIGHT):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.cap: Optional[cv2.VideoCapture] = None

    def start(self) -> bool:
        """Starts video capture. Returns True on success, raises RuntimeError on failure."""
        if not self.is_running():
            self.cap = cv2.VideoCapture(self.camera_index)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            if not self.cap.isOpened():
                self.cap = None
                raise RuntimeError(f"Failed to open camera index {self.camera_index}")
        return True

    def read_frame(self) -> Optional[np.ndarray]:
        """Reads a frame and returns a horizontally flipped (selfie-view) image."""
        if self.is_running() and self.cap is not None:
            ret, frame = self.cap.read()
            if ret and frame is not None:
                return cv2.flip(frame, 1)
        return None

    def is_running(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def stop(self):
        """Stops capture and releases webcam resources."""
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def __del__(self):
        self.stop()
