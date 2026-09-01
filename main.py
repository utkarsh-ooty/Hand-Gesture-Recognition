import os
import ctypes
try:
    kernel32 = ctypes.windll.kernel32
    buf = ctypes.create_unicode_buffer(1024)
    kernel32.GetShortPathNameW(os.getcwd(), buf, 1024)
    if buf.value:
        os.chdir(buf.value)
except Exception:
    pass

import tkinter as tk
from tkinter import ttk, messagebox
import cv2
from PIL import Image, ImageTk
import threading
import time

from camera import Camera
from hand_tracker import HandTracker
from dataset_collector import DatasetCollector
from feature_extraction import extract_features
from gesture_classifier import GestureClassifier
from swipe_detector import SwipeDetector
from train_model import train_gesture_model

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture Recognition AI")
        self.root.geometry("1000x600")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Modules
        self.camera = Camera(camera_index=0)
        self.tracker = HandTracker()
        self.collector = DatasetCollector()
        self.classifier = GestureClassifier()
        self.swipe_detector = SwipeDetector()
        
        # State
        self.is_camera_running = False
        self.is_recognizing = False
        self.display_swipe = None
        self.display_swipe_time = 0
        
        # Build UI
        self._build_ui()
        self.update_loop()
        
    def _build_ui(self):
        # Left Panel (Video)
        self.video_frame = tk.Frame(self.root, width=640, height=480, bg="black")
        self.video_frame.pack(side=tk.LEFT, padx=10, pady=10)
        
        self.video_label = tk.Label(self.video_frame)
        self.video_label.pack()
        
        # Right Panel (Controls)
        self.control_frame = tk.Frame(self.root)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        tk.Label(self.control_frame, text="HAND GESTURE AI", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Status Label
        self.status_label = tk.Label(self.control_frame, text="Status: Ready", font=("Arial", 11), fg="blue")
        self.status_label.pack(pady=5)
        
        # Camera Controls
        cam_frame = tk.LabelFrame(self.control_frame, text="Camera Controls")
        cam_frame.pack(fill=tk.X, pady=5)
        tk.Button(cam_frame, text="Start Camera", command=self.start_camera).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        tk.Button(cam_frame, text="Stop Camera", command=self.stop_camera).pack(side=tk.LEFT, padx=5, pady=5, expand=True)
        
        # Dataset Collection
        dataset_frame = tk.LabelFrame(self.control_frame, text="Dataset Collection")
        dataset_frame.pack(fill=tk.X, pady=5)
        
        self.gesture_class_var = tk.StringVar(value="open_palm")
        classes = ["open_palm", "closed_fist", "thumbs_up", "thumbs_down", "swipe_left", "swipe_right"]
        cb = ttk.Combobox(dataset_frame, textvariable=self.gesture_class_var, values=classes, state="readonly")
        cb.pack(pady=5)
        
        self.record_btn = tk.Button(dataset_frame, text="Start Recording", command=self.toggle_recording, bg="lightcoral")
        self.record_btn.pack(pady=5)
        
        # Training
        train_frame = tk.LabelFrame(self.control_frame, text="Model Training")
        train_frame.pack(fill=tk.X, pady=5)
        tk.Button(train_frame, text="Train Model", command=self.train_model).pack(pady=5)
        
        # Recognition
        recog_frame = tk.LabelFrame(self.control_frame, text="Recognition")
        recog_frame.pack(fill=tk.X, pady=5)
        self.recog_btn = tk.Button(recog_frame, text="Start Recognition", command=self.toggle_recognition, bg="lightgreen")
        self.recog_btn.pack(pady=5)
        
        tk.Button(recog_frame, text="Reload Model", command=self.reload_model).pack(pady=5)
        
        # Inference Feedback
        self.inference_frame = tk.Frame(self.control_frame)
        self.inference_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self.gesture_lbl = tk.Label(self.inference_frame, text="Detected Gesture: None", font=("Arial", 12))
        self.gesture_lbl.pack(anchor="w")
        self.conf_lbl = tk.Label(self.inference_frame, text="Confidence: 0%", font=("Arial", 12))
        self.conf_lbl.pack(anchor="w")
        
    def start_camera(self):
        try:
            self.camera.start()
            self.is_camera_running = True
            self.status_label.config(text="Status: Camera Running", fg="green")
        except Exception as e:
            messagebox.showerror("Error", f"Could not start camera: {e}")
            
    def stop_camera(self):
        self.is_camera_running = False
        self.camera.stop()
        self.video_label.config(image='')
        self.status_label.config(text="Status: Camera Stopped", fg="red")
        
    def toggle_recording(self):
        if not self.collector.is_recording:
            cls_name = self.gesture_class_var.get()
            self.collector.start_recording(cls_name)
            self.record_btn.config(text="Stop Recording", bg="red")
            self.status_label.config(text=f"Status: Recording '{cls_name}'...", fg="red")
        else:
            self.collector.stop_recording()
            self.record_btn.config(text="Start Recording", bg="lightcoral")
            self.status_label.config(text="Status: Camera Running", fg="green")
            
    def toggle_recognition(self):
        self.is_recognizing = not self.is_recognizing
        if self.is_recognizing:
            if not self.classifier.is_loaded:
                messagebox.showwarning("Warning", "No model found! Please collect data and Train Model first.")
                self.is_recognizing = False
                return
            self.recog_btn.config(text="Stop Recognition", bg="orange")
            self.swipe_detector.reset()
        else:
            self.recog_btn.config(text="Start Recognition", bg="lightgreen")
            self.gesture_lbl.config(text="Detected Gesture: None")
            self.conf_lbl.config(text="Confidence: 0%")
            
    def train_model(self):
        self.status_label.config(text="Status: Training Model...", fg="orange")
        def run_train():
            success = train_gesture_model()
            if success:
                self.root.after(0, lambda: messagebox.showinfo("Success", "Model trained successfully!"))
                self.classifier.load_model()
            else:
                self.root.after(0, lambda: messagebox.showerror("Error", "Could not train model. Check if dataset exists."))
            self.root.after(0, lambda: self.status_label.config(text="Status: Ready", fg="blue"))
        threading.Thread(target=run_train, daemon=True).start()
        
    def reload_model(self):
        self.classifier.load_model()
        if self.classifier.is_loaded:
            messagebox.showinfo("Success", "Model reloaded.")
        else:
            messagebox.showerror("Error", "Failed to load model file.")

    def update_loop(self):
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
                
                # If hands detected
                landmarks = self.tracker.get_landmarks(frame, hand_no=0)
                if landmarks:
                    features = extract_features(landmarks)
                    
                    if features is not None:
                        # 2. Data Collection
                        if self.collector.is_recording:
                            self.collector.record(frame, features)
                            
                        # 3. Recognition
                        if self.is_recognizing:
                            # A. Swipe Detection
                            pixel_lms = self.tracker.get_pixel_landmarks(frame, hand_no=0)
                            if pixel_lms:
                                wrist_x, wrist_y = pixel_lms[0][1], pixel_lms[0][2]
                                swipe = self.swipe_detector.add_position(wrist_x, wrist_y)
                                if swipe:
                                    self.display_swipe = swipe
                                    self.display_swipe_time = time.time()
                                    
                            # B. Static Gesture Recognition
                            static_gesture, conf = self.classifier.predict(features)
                            
                            if self.display_swipe:
                                current_gesture = self.display_swipe
                                current_conf = 1.0  # Swipes represent high confidence temporal event
                            else:
                                current_gesture = static_gesture
                                current_conf = conf

                            # Visualization overlay
                            cv2.putText(processed_frame, f"{current_gesture} ({current_conf:.2f})", 
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # Update UI inference labels
                self.gesture_lbl.config(text=f"Detected Gesture: {current_gesture}")
                self.conf_lbl.config(text=f"Confidence: {current_conf*100:.1f}%")
                
                # Render to Tkinter
                img_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk  # Keep ref
                self.video_label.config(image=img_tk)
                
        # Schedule next update (e.g ~30 fps is 33ms)
        self.root.after(33, self.update_loop)
        
    def on_close(self):
        self.camera.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
