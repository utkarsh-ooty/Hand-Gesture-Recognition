"""
Random Forest model trainer for hand gesture recognition.
Parses keypoint CSV dataset, trains classifier, and exports evaluation artifacts.
"""

import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, Callable

from app.config import Config
from ml.evaluation import ModelEvaluator


class ModelTrainer:
    """Trains a Random Forest classifier from keypoint datasets."""

    def __init__(self, dataset_path: Optional[str] = None,
                 model_save_path: Optional[str] = None,
                 cm_save_path: Optional[str] = None):
        self.dataset_path = dataset_path or Config.DATASET_CSV_PATH
        self.model_save_path = model_save_path or Config.RF_MODEL_PATH
        self.cm_save_path = cm_save_path or Config.CONFUSION_MATRIX_PATH

    def train(self, message_callback: Optional[Callable[[str], None]] = None) -> bool:
        """
        Executes model training workflow.
        Returns True on success, False on failure.
        """
        def say(msg: str):
            if message_callback:
                message_callback(msg)
            print(f"[ModelTrainer] {msg}")

        if not os.path.exists(self.dataset_path):
            say("Error. Dataset not found.")
            return False

        say("Dataset loading.")
        try:
            df = pd.read_csv(self.dataset_path)
        except Exception as e:
            say(f"Could not read dataset: {e}")
            return False

        if df.empty or len(df) == 0:
            say("Dataset is empty. Please collect gesture samples first.")
            return False

        print(f"[ModelTrainer] Dataset shape: {df.shape}")

        # First column is class label, remaining 42 are coordinates
        y = df.iloc[:, 0].to_numpy()
        X = df.iloc[:, 1:].values

        # Check that we have at least 2 distinct classes to train
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            say("At least two distinct gestures required to train.")
            return False

        # Split dataset with stratification
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError:
            # Fallback if a class has only 1 sample
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

        print(f"[ModelTrainer] Training on {len(X_train)} samples, validating on {len(X_val)} samples.")
        say("Training started. Please wait.")

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)

        # Evaluate
        y_pred = rf.predict(X_val)
        acc, cm = ModelEvaluator.evaluate(y_val, y_pred, labels=rf.classes_)
        say(f"Training completed. Validation accuracy is {acc * 100:.0f} percent.")

        print(f"[ModelTrainer] Confusion Matrix:\n{cm}")

        # Save confusion matrix plot
        try:
            ModelEvaluator.save_confusion_matrix_plot(
                cm, list(rf.classes_), self.cm_save_path
            )
            print(f"[ModelTrainer] Confusion matrix plot saved to {self.cm_save_path}")
        except Exception as e:
            print(f"[ModelTrainer] Warning saving confusion matrix: {e}")

        # Save serialized model
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump(rf, self.model_save_path)
        say("Model saved successfully.")

        return True


def train_gesture_model(dataset_path: Optional[str] = None,
                        model_save_path: Optional[str] = None,
                        message_callback: Optional[Callable[[str], None]] = None) -> bool:
    """Convenience wrapper preserving legacy functional API."""
    trainer = ModelTrainer(dataset_path=dataset_path, model_save_path=model_save_path)
    return trainer.train(message_callback=message_callback)


if __name__ == "__main__":
    train_gesture_model()
