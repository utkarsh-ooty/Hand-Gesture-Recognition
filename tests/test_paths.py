"""
Tests for centralized path resolution and file preservation.
"""

import os
import unittest
from app.config import Config, get_safe_path


class TestPathResolution(unittest.TestCase):
    """Verifies all application and ML paths resolve reliably."""

    def test_base_dir_exists(self):
        self.assertTrue(os.path.exists(Config.BASE_DIR))
        self.assertTrue(os.path.isdir(Config.BASE_DIR))

    def test_data_dir_structure(self):
        self.assertTrue(os.path.exists(Config.DATA_DIR))
        self.assertTrue(os.path.exists(Config.DATASETS_DIR))
        self.assertTrue(os.path.exists(Config.MODELS_DIR))

    def test_dataset_intact(self):
        """Verifies dataset CSV exists and is non-empty."""
        self.assertTrue(os.path.exists(Config.DATASET_CSV_PATH))
        self.assertGreater(os.path.getsize(Config.DATASET_CSV_PATH), 0)

    def test_model_files_intact(self):
        """Verifies model artifacts exist and are preserved."""
        self.assertTrue(os.path.exists(Config.HAND_LANDMARKER_PATH))
        self.assertGreater(os.path.getsize(Config.HAND_LANDMARKER_PATH), 0)

        self.assertTrue(os.path.exists(Config.RF_MODEL_PATH))
        self.assertGreater(os.path.getsize(Config.RF_MODEL_PATH), 0)

    def test_safe_path_helper(self):
        safe_p = get_safe_path(Config.BASE_DIR)
        self.assertIsInstance(safe_p, str)
        self.assertTrue(len(safe_p) > 0)


if __name__ == "__main__":
    unittest.main()
