import os
import platform
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Define hand connections manually since mp.solutions.drawing_utils might be missing
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # base
]

class HandTracker:
    def __init__(self, model_path=None, max_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        if model_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            model_path = os.path.join(base_dir, 'models', 'hand_landmarker.task')

        # Convert to Windows short path if possible to prevent path issues in C++ backend
        if platform.system() == "Windows":
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                buf = ctypes.create_unicode_buffer(1024)
                kernel32.GetShortPathNameW(model_path, buf, 1024)
                if buf.value:
                    model_path = buf.value
            except Exception:
                pass

        # Setup modern Tasks API for MediaPipe > 0.10.x
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.last_results = None

    def find_hands(self, frame, draw=True):
        """
        Processes the frame, finds hands, draws landmarks manually, and returns the frame.
        """
        # Convert BGR to RGB for MediaPipe format
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect
        self.last_results = self.detector.detect(mp_image)
        
        # Draw manually if we have outcomes
        if self.last_results and self.last_results.hand_landmarks and draw:
            h, w, c = frame.shape
            for hand_landmarks in self.last_results.hand_landmarks:
                # Cache points by their ID
                pts = {}
                for idx, lm in enumerate(hand_landmarks):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pts[idx] = (cx, cy)
                    # Draw joint
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), cv2.FILLED)
                
                # Draw connections
                for connection in HAND_CONNECTIONS:
                    if connection[0] in pts and connection[1] in pts:
                        cv2.line(frame, pts[connection[0]], pts[connection[1]], (0, 255, 0), 2)
                        
        return frame
        
    def get_landmarks(self, frame, hand_no=0):
        """
        Extracts the landmarks of the specified hand as normalized values [id, x, y, z].
        """
        landmark_list = []
        if self.last_results and self.last_results.hand_landmarks:
            if len(self.last_results.hand_landmarks) > hand_no:
                my_hand = self.last_results.hand_landmarks[hand_no]
                for id, lm in enumerate(my_hand):
                    landmark_list.append([id, lm.x, lm.y, lm.z])
                    
        return landmark_list

    def get_handedness(self, hand_no=0):
        """
        Returns 'Right' or 'Left' and score for the specified hand.
        """
        if self.last_results and self.last_results.handedness:
            if len(self.last_results.handedness) > hand_no:
                category = self.last_results.handedness[hand_no][0]
                return category.category_name, category.score
        return None, 0.0
    
    def get_pixel_landmarks(self, frame, hand_no=0):
        """
        Extracts landmarks as absolute pixel coordinates in the frame.
        """
        pixel_landmarks = []
        if self.last_results and self.last_results.hand_landmarks:
            if len(self.last_results.hand_landmarks) > hand_no:
                my_hand = self.last_results.hand_landmarks[hand_no]
                h, w, c = frame.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    pixel_landmarks.append([id, cx, cy])
        return pixel_landmarks
