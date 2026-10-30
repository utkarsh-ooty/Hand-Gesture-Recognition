"""
Feature extraction and normalization from MediaPipe hand landmarks.
"""

from typing import List, Optional
import numpy as np


class FeatureExtractor:
    """Extracts scale- and translation-invariant normalized feature vectors from hand landmarks."""

    @staticmethod
    def extract_features(landmarks: List[List[float]]) -> Optional[np.ndarray]:
        """
        Normalizes the 21 MediaPipe hand landmarks.
        Expects a list of [id, x, y, z] or [id, x, y].
        Returns a flattened, normalized feature vector of length 42 (x, y for 21 points).
        """
        if not landmarks or len(landmarks) < 21:
            return None

        # Extract only x and y
        coords = [[lm[1], lm[2]] for lm in landmarks]
        coords = np.array(coords, dtype=np.float32)

        # 1. Translation invariance: Relativize to wrist (0, 0)
        wrist = coords[0]
        coords = coords - wrist

        # 2. Scale invariance: Normalize by the maximum absolute value
        max_val = np.max(np.abs(coords))
        if max_val > 0:
            coords = coords / max_val

        # Flatten to 1D array of length 42
        return coords.flatten()


# Function alias for backwards compatibility
extract_features = FeatureExtractor.extract_features
