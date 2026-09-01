import os
import cv2
import pandas as pd
import time

class DatasetCollector:
    def __init__(self, dataset_dir=None):
        if dataset_dir is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            dataset_dir = os.path.join(base_dir, "dataset")
        self.dataset_dir = dataset_dir
        self.csv_path = os.path.join(dataset_dir, "keypoint_dataset.csv")
        self.ensure_dirs()
        self.is_recording = False
        self.current_class = None
        self.frame_counter = 0

    def ensure_dirs(self):
        if not os.path.exists(self.dataset_dir):
            os.makedirs(self.dataset_dir)
            
        classes = ["open_palm", "closed_fist", "thumbs_up", "thumbs_down", "swipe_left", "swipe_right"]
        for c in classes:
            c_dir = os.path.join(self.dataset_dir, c)
            if not os.path.exists(c_dir):
                os.makedirs(c_dir)

    def start_recording(self, gesture_class):
        self.current_class = gesture_class
        self.is_recording = True
        self.frame_counter = 0
        print(f"Started recording for {gesture_class}")

    def stop_recording(self):
        print(f"Stopped recording. Captured {self.frame_counter} samples for {self.current_class}.")
        self.is_recording = False
        self.current_class = None

    def record(self, frame, features):
        """
        Records the given frame and features under the current class.
        features should be the normalized 1D array of 42 coordinates.
        """
        if not self.is_recording or self.current_class is None or features is None:
            return

        # Create timestamp for unique filename
        ts = int(time.time() * 1000)
        
        # Save image
        img_name = f"{ts}.jpg"
        img_path = os.path.join(self.dataset_dir, self.current_class, img_name)
        cv2.imwrite(img_path, frame)
        
        # Format features as a list with the label at index 0
        row = [self.current_class] + features.tolist()
        
        # Convert to dataframe and append to CSV
        df = pd.DataFrame([row])
        df.to_csv(self.csv_path, mode='a', header=not os.path.exists(self.csv_path), index=False)
        
        self.frame_counter += 1
