import joblib
import os
import numpy as np

class GestureClassifier:
    def __init__(self, model_path=None, confidence_threshold=0.6):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'rf_model.pkl')
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.is_loaded = False
        
        self.load_model()
        
    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
                self.is_loaded = True
            except Exception as e:
                print(f"Error loading model: {e}")
                
    def predict(self, features):
        """
        Takes a normalized feature vector of length 42.
        Returns a tuple (gesture_name, confidence)
        """
        if not self.is_loaded or self.model is None or features is None:
            return "No Mod/Unknown", 0.0
            
        features = np.array(features).reshape(1, -1)
        
        # Determine probabilities
        probs = self.model.predict_proba(features)[0]
        max_prob = np.max(probs)
        gesture = self.model.classes_[np.argmax(probs)]
        
        if max_prob < self.confidence_threshold:
            return "Unknown", max_prob
            
        return gesture, max_prob
