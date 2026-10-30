"""
Model loading and verification helper.
"""

import os
import joblib
from typing import Optional, Any, List

from app.config import Config, get_safe_path


class ModelLoader:
    """Safely verifies and loads serialized machine learning models."""

    @staticmethod
    def is_model_trained(model_path: Optional[str] = None) -> bool:
        """Checks if the Random Forest model exists on disk."""
        path = model_path or Config.RF_MODEL_PATH
        safe_p = get_safe_path(path)
        return os.path.exists(safe_p) and os.path.getsize(safe_p) > 0

    @staticmethod
    def is_task_model_available(task_path: Optional[str] = None) -> bool:
        """Checks if the MediaPipe hand_landmarker.task model exists."""
        path = task_path or Config.HAND_LANDMARKER_PATH
        safe_p = get_safe_path(path)
        return os.path.exists(safe_p) and os.path.getsize(safe_p) > 0

    @staticmethod
    def load_rf_model(model_path: Optional[str] = None) -> Optional[Any]:
        """Loads and returns the trained Random Forest classifier."""
        path = model_path or Config.RF_MODEL_PATH
        safe_p = get_safe_path(path)
        if not os.path.exists(safe_p):
            return None
        try:
            return joblib.load(safe_p)
        except Exception as e:
            print(f"[ModelLoader] Error loading model from {safe_p}: {e}")
            return None

    @staticmethod
    def get_classes(model_path: Optional[str] = None) -> List[str]:
        """Returns the list of classes known by the model."""
        model = ModelLoader.load_rf_model(model_path)
        if model is not None and hasattr(model, "classes_"):
            return list(model.classes_)
        return []
