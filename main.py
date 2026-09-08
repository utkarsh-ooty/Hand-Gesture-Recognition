import os
import sys
import platform

# Set working directory to the directory where this script resides
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if platform.system() == "Windows":
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        buf = ctypes.create_unicode_buffer(1024)
        kernel32.GetShortPathNameW(SCRIPT_DIR, buf, 1024)
        if buf.value:
            os.chdir(buf.value)
        else:
            os.chdir(SCRIPT_DIR)
    except Exception:
        os.chdir(SCRIPT_DIR)
else:
    os.chdir(SCRIPT_DIR)

if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time
import pandas as pd

from camera import Camera
from hand_tracker import HandTracker
from dataset_collector import DatasetCollector
from feature_extraction import extract_features
from gesture_classifier import GestureClassifier
from swipe_detector import SwipeDetector
from train_model import train_gesture_model

from gesture_config import GESTURES
from application_state import ApplicationState, StateManager
from voice_assistant import VoiceAssistant
from finger_detector import FingerDetector
from navigation_manager import NavigationManager
from command_engine import CommandEngine

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition AI (Accessibility Mode Enabled)")
        self.root.geometry("1000x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # ML / CV Modules
        self.camera = Camera(camera_index=0)
        self.tracker = HandTracker()
        self.collector = DatasetCollector()
        self.classifier = GestureClassifier()
        self.swipe_detector = SwipeDetector()
        
        # Accessibility & State Modules
        self.state_manager = StateManager()
        self.voice_assistant = VoiceAssistant()
        self.finger_detector = FingerDetector()
        self.navigation_manager = NavigationManager(self.on_intent, stability_frames=12, cooldown=2.0)
        self.command_engine = CommandEngine(self.state_manager, self.voice_assistant, self._build_callbacks())
        
        # State
        self.is_camera_running = False
        self.is_recognizing = False
        self.display_swipe = None
        self.display_swipe_time = 0
        self.last_no_hand_time = 0
        self.warned_no_hand = False
        self.hand_frames = 0
        self.last_startup_voice_time = time.time()
        
        # Build UI
        self._build_ui()
        
        # Delay startup to ensure UI renders
        self.root.after(1000, self.start_accessibility_mode)
        self.update_loop()
        
    def _build_callbacks(self):
        return {
            "start_camera": self.start_camera_logic,
            "stop_camera": self.stop_camera_logic,
            "select_gesture": lambda gn: self.gesture_class_var.set(gn),
            "get_current_gesture": lambda: self.gesture_class_var.get(),
            "start_recording": self.start_recording_logic,
            "stop_recording": self.stop_recording_logic,
            "is_recording": lambda: self.collector.is_recording,
            "train_model": self.train_model_logic,
            "is_model_trained": self.check_if_model_trained,
            "get_missing_samples": self.get_missing_samples_logic,
            "start_recognition": self.start_recognition_logic,
            "stop_recognition": self.stop_recognition_logic,
            "reload_model": self.reload_model_logic,
            "close_app": self.on_close
        }
        
    def start_accessibility_mode(self):
        self.command_engine.say("Welcome to Hand Gesture AI. Accessibility mode is active.")
        self.state_manager.set_state(ApplicationState.WAITING_FOR_HAND)
        self.start_camera_logic()
        self.command_engine.say("Camera is active. Please show your hand clearly in front of the camera.")
        self.last_startup_voice_time = time.time()
        
    def on_intent(self, intent, payload=None):
        """Callback from NavigationManager OR GestureClassifier (when running)."""
        self.command_engine.process_intent(intent, payload)
        
    def _build_ui(self):
        # Left Panel (Video)
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        
        # Right Panel (Controls - purely for dev debugging)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.control_frame, text="ACCESSIBILITY MODE: ON", font=("Arial", 16, "bold"), fg="purple").pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(self.control_frame, text="Status: Init", font=("Arial", 11), fg="blue")
        self.status_label.pack(pady=5)
        
        self.finger_lbl = tk.Label(self.control_frame, text="Fingers: 0", font=("Arial", 11))
        self.finger_lbl.pack()
        
        # Camera Controls
        cam_frame = tk.LabelFrame(self.control_frame, text="Camera Controls")
        cam_frame.pack(fill=tk.X, pady=5)
        tk.Button(cam_frame, text="Start Camera", command=self.start_camera_logic).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        tk.Button(cam_frame, text="Stop Camera", command=self.stop_camera_logic).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        
        # Dataset Collection
        dataset_frame = tk.LabelFrame(self.control_frame, text="Dataset Collection (Debug)")
        dataset_frame.pack(fill=tk.X, pady=5)
        
        self.gesture_class_var = tk.StringVar(value=GESTURES[0])
        self.gesture_cb = ttk.Combobox(dataset_frame, textvariable=self.gesture_class_var, values=GESTURES, state="readonly")
        self.gesture_cb.pack(pady=5)
        
        # Inference Feedback
        self.inference_frame = tk.LabelFrame(self.control_frame, text="Inference Feedback")
        self.inference_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gesture_lbl = tk.Label(self.inference_frame, text="Detected Gesture: None", font=("Arial", 12))
        self.gesture_lbl.pack(anchor="w")
        self.conf_lbl = tk.Label(self.inference_frame, text="Confidence: 0%", font=("Arial", 12))
        self.conf_lbl.pack(anchor="w")
        
    def start_camera_logic(self):
        try:
            self.camera.start()
            self.is_camera_running = True
            self.status_label.config(text="Status: Camera Running", fg="green")
        except Exception as e:
            print(f"Could not start camera: {e}")
            
    def stop_camera_logic(self):
        self.is_camera_running = False
        self.camera.stop()
        self.video_label.config(image='')
        self.status_label.config(text="Status: Camera Stopped", fg="red")

    def start_recording_logic(self):
        cls_name = self.gesture_class_var.get()
        self.collector.start_recording(cls_name)
        
    def stop_recording_logic(self):
        self.collector.stop_recording()
            
    def train_model_logic(self):
        self.status_label.config(text="Status: Training Model...", fg="orange")
        success = train_gesture_model()
        if success:
            self.classifier.load_model()
        self.root.after(0, lambda: self.status_label.config(text="Status: Ready", fg="blue"))
        return success
        
    def get_missing_samples_logic(self):
        csv_path = os.path.join(SCRIPT_DIR, "dataset", "keypoint_dataset.csv")
        if not os.path.exists(csv_path):
            return GESTURES
        try:
            df = pd.read_csv(csv_path)
            if len(df) == 0:
                return GESTURES
            counts = df.iloc[:, 0].value_counts()
            missing = [g for g in GESTURES if g not in counts or counts[g] < 10]
            return missing
        except Exception:
            return GESTURES

    def check_if_model_trained(self):
        return os.path.exists(os.path.join(SCRIPT_DIR, "models", "rf_model.pkl"))

    def start_recognition_logic(self):
        self.classifier.load_model()
        if not self.classifier.is_loaded:
            return False
        self.is_recognizing = True
        self.swipe_detector.reset()
        return True
        
    def stop_recognition_logic(self):
        self.is_recognizing = False
        self.gesture_lbl.config(text="Detected Gesture: None")
        self.conf_lbl.config(text="Confidence: 0%")
            
    def reload_model_logic(self):
        self.classifier.load_model()

    def update_loop(self):
        self.status_label.config(text=f"State: {self.state_manager.get_state().value}")
        
        if self.is_camera_running:
            frame = self.camera.read_frame()
            if frame is not None:
                # 1. Process with MediaPipe
                processed_frame = self.tracker.find_hands(frame.copy(), draw=True)
                
                # Default states
                current_gesture = "No Hand Detected"
                current_conf = 0.0
                
                # Check for swipe clear
                if time.time() - self.display_swipe_time > 1.5:
                    self.display_swipe = None
                    self.swipe_detector.reset() # clear out temp
                
                # If hands detected
                landmarks = self.tracker.get_landmarks(frame, hand_no=0)
                
                current_state = self.state_manager.get_state()
                
                if not landmarks:
                    self.navigation_manager.process_finger_count(None)
                    self.hand_frames = 0
                    if not self.warned_no_hand:
                        if self.last_no_hand_time == 0:
                            self.last_no_hand_time = time.time()
                        elif time.time() - self.last_no_hand_time > 3.0:
                            # Only warn after startup flow is completed
                            if current_state != ApplicationState.STARTUP and current_state != ApplicationState.WAITING_FOR_HAND:
                                self.command_engine.say("No hand detected. Please show your hand in front of the camera.")
                            self.warned_no_hand = True
                    
                    if current_state == ApplicationState.WAITING_FOR_HAND:
                        if time.time() - self.last_startup_voice_time > 6.0:
                            self.command_engine.say("No hand detected. Please show your hand in front of the camera.")
                            self.last_startup_voice_time = time.time()

                else:
                    self.warned_no_hand = False
                    self.last_no_hand_time = 0
                    
                    if current_state == ApplicationState.WAITING_FOR_HAND:
                        self.hand_frames += 1
                        if self.hand_frames >= 5:
                            self.command_engine.say("Hand detected successfully. Accessibility navigation is ready.")
                            self.on_intent("STARTUP")
                    else:
                        self.hand_frames = 0
                    
                    # Core Processing
                    features = extract_features(landmarks)
                    hand_category, hand_score = self.tracker.get_handedness(0)
                    
                    finger_result = self.finger_detector.count_fingers(landmarks, hand_category)
                    if finger_result:
                        finger_count, finger_details = finger_result
                    else:
                        finger_count = 0
                        finger_details = {'thumb': 0, 'index': 0, 'middle': 0, 'ring': 0, 'pinky': 0}
                        
                    details_str = f"T:{finger_details['thumb']} I:{finger_details['index']} M:{finger_details['middle']} R:{finger_details['ring']} P:{finger_details['pinky']}"
                    self.finger_lbl.config(text=f"Fingers: {finger_count}\n{hand_category} ({hand_score:.2f})\n{details_str}")
                    cv2.putText(processed_frame, f"Hands: 1 | Detected: YES", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if current_state in (ApplicationState.STARTUP, ApplicationState.WAITING_FOR_HAND):
                        pass # Handled above
                    
                    elif current_state == ApplicationState.DATASET_COLLECTING:
                        current_gesture = "Collecting Mode"
                        # Automated checking for the active dataset gesture request
                        current_gest = GESTURES[self.command_engine.dataset_gesture_index]
                        
                        if self.finger_detector.check_gesture_match(current_gest, landmarks):
                            if not self.collector.is_recording:
                                pretty = current_gest.replace('_', ' ').title()
                                self.command_engine.say(f"{pretty} detected. Collecting samples.")
                                self.collector.start_recording(current_gest)
                        
                        if self.collector.is_recording:
                            self.collector.record(frame, features)
                            current_gesture = f"Recording {current_gest}: {self.collector.frame_counter}/50"
                            if self.collector.frame_counter >= 50:
                                self.collector.stop_recording()
                                self.command_engine.advance_dataset_collection()
                            
                    elif not self.is_recognizing:
                        current_gesture = "Navigation Mode"
                        # Standard visual finger navigation
                        if finger_count is not None:
                            self.navigation_manager.process_finger_count(finger_count)
                            cv2.putText(processed_frame, f"Fingers: {finger_count}", 
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                            
                    else:
                        # 3. Recognition Mode ML pipeline
                        
                        # A. Swipe Detection
                        pixel_lms = self.tracker.get_pixel_landmarks(frame, hand_no=0)
                        if pixel_lms:
                            wrist_x, wrist_y = pixel_lms[0][1], pixel_lms[0][2]
                            swipe = self.swipe_detector.add_position(wrist_x, wrist_y)
                            if swipe:
                                self.display_swipe = swipe
                                self.display_swipe_time = time.time()
                                # Trigger Command Engine for swipes! (Shares actions, same as UI intent)
                                if swipe == "swipe_right":
                                    self.on_intent("NEXT")
                                elif swipe == "swipe_left":
                                    self.on_intent("PREV")

                        # B. Static Gesture Recognition
                        if features is not None:
                            static_gesture, conf = self.classifier.predict(features)
                            
                            if self.display_swipe:
                                current_gesture = self.display_swipe
                                current_conf = 1.0  # Temporal event
                            else:
                                current_gesture = static_gesture
                                current_conf = conf

                            # Route static classification to command intent mapped if confidence > 0.8
                            if current_conf > 0.85:
                                if current_gesture == "thumbs_up":
                                    # Debouncing logic applies to ML too, NavigationManager processes all intents cleanly.
                                    # Wait, NavigationManager expects ints. 
                                    pass

                            cv2.putText(processed_frame, f"{current_gesture} ({current_conf:.2f})", 
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # Update UI inference labels
                self.gesture_lbl.config(text=f"Detected Gesture: {current_gesture}")
                self.conf_lbl.config(text=f"Confidence: {current_conf*100:.1f}%")
                
                # Render to Tkinter
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk
                self.video_label.config(image=img_tk)
                
        # Schedule next update
        self.root.after(33, self.update_loop)
        
    def on_close(self):
        self.camera.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
