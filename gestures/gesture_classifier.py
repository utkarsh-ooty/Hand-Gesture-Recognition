"""
Inference wrapper for trained Scikit-Learn Random Forest gesture classifier.
"""

import os
import joblib
import numpy as np
from typing import Tuple, Optional

from app.config import Config, get_safe_path


class GestureClassifier:
    """Classifies normalized hand keypoint vectors using a trained Random Forest model."""

    def __init__(self, model_path: Optional[str] = None, confidence_threshold: float = 0.6):
        if model_path is None:
            model_path = Config.RF_MODEL_PATH

        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False

        self.load_model()

    def load_model(self) -> bool:
        """Loads or reloads the serialized model."""
        safe_path = get_safe_path(self.model_path)
        if os.path.exists(safe_path):
            try:
                self.model = joblib.load(safe_path)
                self.is_loaded = True
                return True
            except Exception as e:
                print(f"[GestureClassifier] Error loading model from {safe_path}: {e}")
                self.model = None
                self.is_loaded = False
                return False
        self.is_loaded = False
        return False

    def predict(self, features: np.ndarray) -> Tuple[str, float]:
        """
        Takes a normalized feature vector of length 42.
        Returns a tuple (gesture_name, confidence).
        """
        if not self.is_loaded or self.model is None or features is None:
            return "No Mod/Unknown", 0.0

        features_arr = np.array(features).reshape(1, -1)

        try:
            probs = self.model.predict_proba(features_arr)[0]
            max_prob = float(np.max(probs))
            gesture = str(self.model.classes_[np.argmax(probs)])

            if max_prob < self.confidence_threshold:
                return "Unknown", max_prob

            return gesture, max_prob
        except Exception as e:
            print(f"[GestureClassifier] Prediction error: {e}")
            return "Error", 0.0
