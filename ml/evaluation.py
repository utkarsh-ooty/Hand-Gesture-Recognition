"""
Evaluation metrics and confusion matrix plotting for gesture classification models.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from typing import Tuple, List, Optional


class ModelEvaluator:
    """Computes evaluation metrics and exports confusion matrix visualizations."""

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
                 labels: Optional[List[str]] = None) -> Tuple[float, np.ndarray]:
        """Calculates accuracy and confusion matrix."""
        acc = float(accuracy_score(y_true, y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        return acc, cm

    @staticmethod
    def save_confusion_matrix_plot(cm: np.ndarray, classes: List[str],
                                   save_path: str, title: str = "Confusion Matrix") -> str:
        """Plots confusion matrix heatmap and saves it to an image file."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title(title)
        plt.colorbar()

        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        plt.savefig(save_path)
        plt.close()
        return save_path
