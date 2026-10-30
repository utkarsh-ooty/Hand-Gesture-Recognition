"""
Tests for ML package: ModelLoader, DatasetManager, and ModelEvaluator.
"""

import unittest
import numpy as np

from ml.model_loader import ModelLoader
from ml.dataset_manager import DatasetManager
from ml.evaluation import ModelEvaluator
from app.config import Config


class TestMLPackage(unittest.TestCase):

    def test_model_loader_real_model(self):
        """Verifies loading the project's trained Random Forest model."""
        self.assertTrue(ModelLoader.is_model_trained(Config.RF_MODEL_PATH))
        model = ModelLoader.load_rf_model(Config.RF_MODEL_PATH)
        self.assertIsNotNone(model)
        classes = ModelLoader.get_classes(Config.RF_MODEL_PATH)
        self.assertGreater(len(classes), 0)

    def test_model_loader_missing_model_fails_gracefully(self):
        """Verifies missing model returns False/None without crashing."""
        fake_path = "non_existent_model_dir/fake.pkl"
        self.assertFalse(ModelLoader.is_model_trained(fake_path))
        self.assertIsNone(ModelLoader.load_rf_model(fake_path))
        self.assertEqual(ModelLoader.get_classes(fake_path), [])

    def test_dataset_manager_sample_counts(self):
        dm = DatasetManager()
        counts = dm.get_sample_counts()
        self.assertIsInstance(counts, dict)
        for gesture in Config.GESTURES:
            self.assertIn(gesture, counts)
            self.assertIsInstance(counts[gesture], int)

    def test_model_evaluator(self):
        y_true = np.array(["open_palm", "closed_fist", "thumbs_up"])
        y_pred = np.array(["open_palm", "closed_fist", "closed_fist"])
        labels = ["open_palm", "closed_fist", "thumbs_up"]

        acc, cm = ModelEvaluator.evaluate(y_true, y_pred, labels=labels)
        self.assertAlmostEqual(acc, 2.0 / 3.0)
        self.assertEqual(cm.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
