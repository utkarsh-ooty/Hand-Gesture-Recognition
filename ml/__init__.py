"""
Machine Learning package for Hand Gesture AI.
Handles dataset management, model training, evaluation, and serialization.
"""

from ml.dataset_manager import DatasetManager
from ml.model_trainer import ModelTrainer, train_gesture_model
from ml.model_loader import ModelLoader
from ml.evaluation import ModelEvaluator

__all__ = [
    "DatasetManager",
    "ModelTrainer",
    "train_gesture_model",
    "ModelLoader",
    "ModelEvaluator",
]
