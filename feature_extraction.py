import numpy as np

def extract_features(landmarks):
    """
    Normalizes the 21 MediaPipe hand landmarks.
    Expects a list of [id, x, y, z] or [id, x, y].
    Returns a flattened, normalized feature vector of length 42 (x, y for 21 points).
    """
    # Extract only x and y for simplicity and robustness
    # landmarks[0] is the wrist
    if len(landmarks) < 21:
        return None

    # We extract x and y
    coords = []
    for lm in landmarks:
        coords.append([lm[1], lm[2]])
    
    coords = np.array(coords, dtype=np.float32)

    # 1. Translation invariance: Relativize to wrist (0,0)
    wrist = coords[0]
    coords = coords - wrist
    
    # 2. Scale invariance: Normalize by the maximum absolute value
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords = coords / max_val
        
    # Flatten to 1D array of length 42
    return coords.flatten()
