"""
Dataset management and collection tool for hand gesture samples and normalized keypoints.
"""

import os
import cv2
import pandas as pd
import time
from typing import List, Optional, Dict
import numpy as np

from app.config import Config


class DatasetManager:
    """Manages gesture image saving and CSV keypoint feature collection."""

    def __init__(self, dataset_dir: Optional[str] = None, csv_path: Optional[str] = None):
        self.dataset_dir = dataset_dir or Config.DATASETS_DIR
        self.csv_path = csv_path or Config.DATASET_CSV_PATH
        self.ensure_dirs()

        self.is_recording: bool = False
        self.current_class: Optional[str] = None
        self.frame_counter: int = 0

    def ensure_dirs(self):
        """Creates dataset directory and class subdirectories if they do not exist."""
        os.makedirs(self.dataset_dir, exist_ok=True)
        for gesture in Config.GESTURES:
            os.makedirs(os.path.join(self.dataset_dir, gesture), exist_ok=True)

    def start_recording(self, gesture_class: str):
        """Starts recording frames and keypoints for a given gesture class."""
        self.current_class = gesture_class
        self.is_recording = True
        self.frame_counter = 0

    def stop_recording(self):
        """Stops active recording session."""
        self.is_recording = False
        self.current_class = None

    def record(self, frame: np.ndarray, features: Optional[np.ndarray]) -> bool:
        """
        Records the given image frame and normalized 42-D feature vector.
        Returns True if recorded, False otherwise.
        """
        if not self.is_recording or self.current_class is None or features is None:
            return False

        # Save image with millisecond timestamp
        ts = int(time.time() * 1000)
        img_name = f"{ts}.jpg"
        class_folder = os.path.join(self.dataset_dir, self.current_class)
        os.makedirs(class_folder, exist_ok=True)
        img_path = os.path.join(class_folder, img_name)
        cv2.imwrite(img_path, frame)

        # Format features row: class_label, x0, y0, ..., x20, y20
        row = [self.current_class] + features.tolist()

        # Append to CSV
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)

        self.frame_counter += 1
        return True

    def get_sample_counts(self) -> Dict[str, int]:
        """Returns a dict mapping gesture names to sample counts in the CSV."""
        if not os.path.exists(self.csv_path):
            return {g: 0 for g in Config.GESTURES}
        try:
            df = pd.read_csv(self.csv_path)
            if df.empty:
                return {g: 0 for g in Config.GESTURES}
            counts = df.iloc[:, 0].value_counts().to_dict()
            return {g: int(counts.get(g, 0)) for g in Config.GESTURES}
        except Exception:
            return {g: 0 for g in Config.GESTURES}

    def get_missing_samples(self, min_samples: int = 10) -> List[str]:
        """Returns list of gesture names that have fewer than min_samples."""
        counts = self.get_sample_counts()
        return [g for g, cnt in counts.items() if cnt < min_samples]
