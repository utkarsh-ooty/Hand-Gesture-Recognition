import os
import sys
import platform

# Suppress MediaPipe & TensorFlow C++ warning noise
os.environ["GLOG_minloglevel"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
from speech_controller import SpeechController
from finger_detector import FingerDetector
from navigation_manager import NavigationManager
from command_engine import CommandEngine

class HandGestureApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Hand Gesture & Voice Navigation AI")
        self.root.geometry("1180x680")
        self.root.minsize(1050, 620)
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
        self.speech_controller = SpeechController(self.on_intent, voice_assistant=self.voice_assistant)
        self.finger_detector = FingerDetector()
        self.navigation_manager = NavigationManager(self.on_intent, stability_frames=12, cooldown=2.0)
        self.camera_ratio = 0.40  # Default: 40% camera, 60% commands & guide
        self._last_root_w = 0
        self.command_engine = CommandEngine(self.state_manager, self.voice_assistant, self._build_callbacks())
        
        # State
        self.is_camera_running = False
        self.is_recognizing = False
        self.display_swipe = None
        self.display_swipe_time = 0
        self.hand_present = False
        self.hand_lost_time = 0
        self.last_no_hand_speech_time = 0
        self.hand_frames = 0
        self.last_startup_voice_time = time.time()
        
        self.last_ml_gesture = None
        self.ml_gesture_frames = 0
        
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
            "adjust_camera_size": lambda delta: self.set_camera_ratio(self.camera_ratio + delta),
            "set_camera_ratio": self.set_camera_ratio,
            "close_app": self.on_close
        }
        
    def start_accessibility_mode(self):
        self.command_engine.say("Welcome to Hand Gesture AI. This application allows you to control the interface using simple hand gestures, without needing a mouse or keyboard. Accessibility mode is now active, so I'll guide you through the application using voice instructions.")
        self.state_manager.set_state(ApplicationState.WAITING_FOR_HAND)
        self.start_camera_logic()
        self.speech_controller.start_listening()
        self.command_engine.say("Your camera is now starting. Voice commands and hand gestures are both active.")
        self.last_startup_voice_time = time.time()
        
    def on_intent(self, intent, payload=None):
        """Callback from NavigationManager, SpeechController, OR GestureClassifier."""
        self.command_engine.process_intent(intent, payload)
        
    def _build_ui(self):
        # Configure root window dark theme
        self.root.configure(bg="#11111b")
        
        # Horizontal PanedWindow allowing split-divider dragging between camera and guidance
        self.paned_window = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg="#11111b",
            sashwidth=6,
            sashrelief="flat",
            sashpad=2
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Panel (Camera Stream - Customizable 40% default width)
        self.video_panel = tk.Frame(self.paned_window, bg="#11111b")
        self.paned_window.add(self.video_panel, minsize=260)
        
        cam_header = tk.Frame(self.video_panel, bg="#11111b")
        cam_header.pack(fill=tk.X, pady=(0, 4))
        tk.Label(cam_header, text="📷 Camera View", font=("Segoe UI", 11, "bold"), fg="#89b4fa", bg="#11111b").pack(side=tk.LEFT)
        self.cam_live_badge = tk.Label(cam_header, text="● LIVE", font=("Segoe UI", 9, "bold"), fg="#a6e3a1", bg="#11111b")
        self.cam_live_badge.pack(side=tk.RIGHT)

        # Camera Width Customizer Bar (Presets + Smooth Slider)
        size_bar = tk.Frame(self.video_panel, bg="#181825", padx=8, pady=4)
        size_bar.pack(fill=tk.X, pady=(0, 8))
        tk.Label(size_bar, text="📐 Width:", font=("Segoe UI", 9, "bold"), fg="#a6adc8", bg="#181825").pack(side=tk.LEFT, padx=(0, 6))

        self.ratio_btn_30 = tk.Button(size_bar, text="30%", font=("Segoe UI", 8), bg="#313244", fg="#cdd6f4", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.30))
        self.ratio_btn_30.pack(side=tk.LEFT, padx=2)

        self.ratio_btn_40 = tk.Button(size_bar, text="40%", font=("Segoe UI", 8, "bold"), bg="#89b4fa", fg="#11111b", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.40))
        self.ratio_btn_40.pack(side=tk.LEFT, padx=2)

        self.ratio_btn_50 = tk.Button(size_bar, text="50%", font=("Segoe UI", 8), bg="#313244", fg="#cdd6f4", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.50))
        self.ratio_btn_50.pack(side=tk.LEFT, padx=2)

        self.ratio_slider = tk.Scale(
            size_bar,
            from_=25, to=60,
            orient=tk.HORIZONTAL,
            showvalue=True,
            bg="#181825", fg="#bac2de",
            highlightthickness=0,
            troughcolor="#313244",
            activebackground="#89b4fa",
            length=80,
            command=self.on_slider_ratio
        )
        self.ratio_slider.set(40)
        self.ratio_slider.pack(side=tk.RIGHT)
        tk.Label(size_bar, text="Slide:", font=("Segoe UI", 8), fg="#6c7086", bg="#181825").pack(side=tk.RIGHT, padx=(4, 2))
        
        self.video_frame = tk.Frame(self.video_panel, bg="#000000", bd=2, relief="groove")
        self.video_frame.pack(fill=tk.X, pady=(0, 8))
        
        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(expand=True)
        
        # Quick Camera Tip on Left Panel
        cam_tip_frame = tk.Frame(self.video_panel, bg="#1e1e2e", padx=10, pady=8)
        cam_tip_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Label(cam_tip_frame, text="💡 Camera Tip:", font=("Segoe UI", 9, "bold"), fg="#f9e2af", bg="#1e1e2e").pack(anchor="w")
        self.cam_tip_text = tk.Label(
            cam_tip_frame,
            text="Hold hand 1-2 ft away. Spread fingers clearly. Drag divider or use buttons above to adjust view width.",
            font=("Segoe UI", 8), fg="#bac2de", bg="#1e1e2e", wraplength=260, justify="left"
        )
        self.cam_tip_text.pack(anchor="w", pady=(2, 0))

        # Right Panel (User Guidance & Controls - Takes 60% of Window)
        self.control_frame = tk.Frame(self.paned_window, bg="#181825", padx=20, pady=15)
        self.paned_window.add(self.control_frame, minsize=420)
        
        # --- 1. APP HEADER ---
        header_frame = tk.Frame(self.control_frame, bg="#181825")
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(header_frame, text="✨ Hand Gesture & Voice Navigation AI", font=("Segoe UI", 16, "bold"), fg="#cdd6f4", bg="#181825").pack(side=tk.LEFT)
        
        # --- 2. FRIENDLY STATUS BANNER (No programmer jargon) ---
        self.status_banner = tk.Label(
            self.control_frame,
            text="👋 Please hold your hand in front of the camera to begin",
            font=("Segoe UI", 11, "bold"),
            bg="#313244", fg="#f9e2af",
            padx=14, pady=9, wraplength=460
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 10))

        # --- 3. INTERACTIVE VISUAL MENU (Shows users where they are) ---
        menu_card = tk.LabelFrame(
            self.control_frame,
            text=" 📋 Menu Options ",
            font=("Segoe UI", 10, "bold"),
            fg="#89b4fa", bg="#1e1e2e",
            padx=12, pady=8, bd=1
        )
        menu_card.pack(fill=tk.X, pady=(0, 10))
        
        self.menu_icons = {
            "Dataset Collection": "📁",
            "Model Training": "🧠",
            "Recognition": "✨",
            "Settings": "⚙️",
            "Help": "❓",
            "Exit": "🚪"
        }
        
        self.menu_labels = {}
        for item in self.command_engine.main_menu_items:
            icon = self.menu_icons.get(item, "📌")
            lbl = tk.Label(
                menu_card,
                text=f"  {icon}  {item}",
                font=("Segoe UI", 10),
                fg="#a6adc8", bg="#1e1e2e",
                anchor="w", padx=12, pady=4
            )
            lbl.pack(fill=tk.X, pady=1)
            self.menu_labels[item] = lbl

        # --- 4. ACTION EXPLANATION CARD ("What will this do?") ---
        action_card = tk.LabelFrame(
            self.control_frame,
            text=" 💡 What Does This Option Do? ",
            font=("Segoe UI", 10, "bold"),
            fg="#a6e3a1", bg="#1e1e2e",
            padx=14, pady=10, bd=1
        )
        action_card.pack(fill=tk.X, pady=(0, 10))
        
        self.action_title_lbl = tk.Label(action_card, text="📁 Dataset Collection", font=("Segoe UI", 13, "bold"), fg="#89b4fa", bg="#1e1e2e")
        self.action_title_lbl.pack(anchor="w", pady=(0, 2))
        
        self.action_desc_lbl = tk.Label(
            action_card,
            text="Records camera sample photos of your hand gestures so the AI learns how your hand looks.",
            font=("Segoe UI", 10),
            fg="#cdd6f4", bg="#1e1e2e",
            wraplength=460, justify="left"
        )
        self.action_desc_lbl.pack(anchor="w", pady=(0, 6))
        
        self.action_trigger_lbl = tk.Label(
            action_card,
            text="👉 TO SELECT: Show 3 Fingers 🤟  OR  Say 'Select'",
            font=("Segoe UI", 10, "bold"),
            fg="#f9e2af", bg="#1e1e2e",
            justify="left"
        )
        self.action_trigger_lbl.pack(anchor="w")

        # --- 5. HOW TO CONTROL (Cheatsheet at a glance) ---
        guide_card = tk.LabelFrame(
            self.control_frame,
            text=" 🎮 Quick Controls Guide ",
            font=("Segoe UI", 10, "bold"),
            fg="#cba6f7", bg="#1e1e2e",
            padx=12, pady=8, bd=1
        )
        guide_card.pack(fill=tk.X, pady=(0, 10))
        
        guide_col1 = (
            "☝️ 1 Finger: Scroll Next\n"
            "✌️ 2 Fingers: Scroll Previous\n"
            "🤟 3 Fingers: Select / Confirm"
        )
        guide_col2 = (
            "🖐️ 5 Fingers: Return Home\n"
            "✊ Closed Fist: Go Back\n"
            "🎤 Mic: Say 'Next', 'Select', 'Train'"
        )
        g_row = tk.Frame(guide_card, bg="#1e1e2e")
        g_row.pack(fill=tk.X)
        tk.Label(g_row, text=guide_col1, font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e", justify="left").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(g_row, text=guide_col2, font=("Segoe UI", 9), fg="#bac2de", bg="#1e1e2e", justify="left").pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- 6. AI ASSISTANT & VOICE BUBBLE ---
        voice_card = tk.Frame(self.control_frame, bg="#181825")
        voice_card.pack(fill=tk.X, pady=(4, 0))
        
        self.speech_bubble_lbl = tk.Label(
            voice_card,
            text="💬 AI Assistant: \"Welcome. I'm ready to guide you.\"",
            font=("Segoe UI", 9, "italic"),
            fg="#b4befe", bg="#181825",
            wraplength=460, justify="left"
        )
        self.speech_bubble_lbl.pack(anchor="w")
        
        self.mic_bubble_lbl = tk.Label(
            voice_card,
            text="🎙️ Microphone: Listening for voice commands",
            font=("Segoe UI", 9),
            fg="#a6e3a1", bg="#181825",
            wraplength=460, justify="left"
        )
        self.mic_bubble_lbl.pack(anchor="w", pady=(2, 0))

        # Hidden variables for backwards compatibility with existing callbacks
        self.gesture_class_var = tk.StringVar(value=GESTURES[0])
        self.status_label = tk.Label(self.root, text="")

        # Dynamic Window resize & sash ratio initialization
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.after(150, lambda: self.set_camera_ratio(0.40))

    def set_camera_ratio(self, ratio: float):
        """Sets the camera width ratio (e.g. 0.40 for 40%) and updates sash & buttons."""
        self.camera_ratio = max(0.25, min(0.60, ratio))
        if hasattr(self, 'ratio_slider'):
            self.ratio_slider.set(int(round(self.camera_ratio * 100)))
        self._update_ratio_button_styles()
        self.apply_camera_ratio()

    def on_slider_ratio(self, val):
        """Callback from Tkinter scale slider."""
        ratio = float(val) / 100.0
        self.camera_ratio = max(0.25, min(0.60, ratio))
        self._update_ratio_button_styles()
        self.apply_camera_ratio()

    def _update_ratio_button_styles(self):
        """Highlights the active preset button corresponding to the current ratio."""
        cur_pct = int(round(self.camera_ratio * 100))
        btns = [
            (30, getattr(self, 'ratio_btn_30', None)),
            (40, getattr(self, 'ratio_btn_40', None)),
            (50, getattr(self, 'ratio_btn_50', None)),
        ]
        for pct, btn in btns:
            if btn:
                if abs(cur_pct - pct) <= 2:
                    btn.config(bg="#89b4fa", fg="#11111b", font=("Segoe UI", 8, "bold"))
                else:
                    btn.config(bg="#313244", fg="#cdd6f4", font=("Segoe UI", 8))

    def apply_camera_ratio(self):
        """Places the PanedWindow sash to match the configured camera width ratio."""
        if not hasattr(self, 'paned_window'):
            return
        total_w = self.paned_window.winfo_width()
        if total_w < 200:
            total_w = self.root.winfo_width() - 24
        if total_w > 200:
            sash_x = int(total_w * self.camera_ratio)
            try:
                self.paned_window.sash_place(0, sash_x, 0)
            except Exception:
                pass

    def _on_window_configure(self, event):
        """Handles window resize events so camera ratio stays consistent."""
        if event.widget == self.root:
            w = event.width
            if hasattr(self, '_last_root_w') and self._last_root_w == w:
                return
            self._last_root_w = w
            self.apply_camera_ratio()

    def start_camera_logic(self):
        try:
            self.camera.start()
            self.is_camera_running = True
        except Exception as e:
            print(f"Could not start camera: {e}")
            
    def stop_camera_logic(self):
        self.is_camera_running = False
        self.camera.stop()
        self.video_label.config(image='')

    def start_recording_logic(self):
        cls_name = self.gesture_class_var.get()
        self.collector.start_recording(cls_name)
        
    def stop_recording_logic(self):
        self.collector.stop_recording()
            
    def train_model_logic(self):
        success = train_gesture_model(message_callback=self.command_engine.say)
        if success:
            self.classifier.load_model()
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
            
    def reload_model_logic(self):
        self.classifier.load_model()

    def update_loop(self):
        if self.is_camera_running:
            frame = self.camera.read_frame()
            if frame is not None:
                # 1. Process with MediaPipe
                processed_frame = self.tracker.find_hands(frame.copy(), draw=True)
                
                # Default states
                current_gesture = "NONE"
                current_conf = 0.0
                finger_count = 0
                
                # Check for swipe clear
                if time.time() - self.display_swipe_time > 1.5:
                    self.display_swipe = None
                    self.swipe_detector.reset()
                
                landmarks = self.tracker.get_landmarks(frame, hand_no=0)
                current_state = self.state_manager.get_state()
                
                if not landmarks:
                    self.navigation_manager.process_finger_count(None)
                    self.hand_frames = 0
                    
                    if self.hand_present or self.hand_lost_time == 0:
                        self.hand_present = False
                        self.hand_lost_time = time.time()
                        
                    time_lost = time.time() - self.hand_lost_time
                    
                    if time_lost > 3.0:
                        if time.time() - self.last_no_hand_speech_time > 8.0:
                            if current_state != ApplicationState.STARTUP and current_state != ApplicationState.WAITING_FOR_HAND:
                                self.command_engine.say("No hand detected. Please show your hand in front of the camera.")
                            self.last_no_hand_speech_time = time.time()
                    
                    if current_state == ApplicationState.WAITING_FOR_HAND:
                        if time.time() - self.last_startup_voice_time > 6.0:
                            self.command_engine.say("I cannot see your hand. Please move your hand into the camera view.")
                            self.last_startup_voice_time = time.time()

                else:
                    if not self.hand_present:
                        self.hand_present = True
                        if time.time() - self.hand_lost_time > 3.0 and self.hand_lost_time != 0:
                            if current_state != ApplicationState.STARTUP and current_state != ApplicationState.WAITING_FOR_HAND:
                                self.command_engine.say("Hand detected successfully. Accessibility navigation is ready.")
                        self.hand_lost_time = 0
                        self.last_no_hand_speech_time = 0
                    
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
                        
                    cv2.putText(processed_frame, f"Hands: 1 | Fingers: {finger_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    if current_state in (ApplicationState.STARTUP, ApplicationState.WAITING_FOR_HAND):
                        pass
                    
                    elif current_state == ApplicationState.DATASET_COLLECTING:
                        current_gesture = "Collecting Mode"
                        current_gest = GESTURES[self.command_engine.dataset_gesture_index]
                        
                        if self.finger_detector.check_gesture_match(current_gest, landmarks):
                            if not self.collector.is_recording:
                                pretty = current_gest.replace('_', ' ').title()
                                self.command_engine.say(f"{pretty} detected. Collecting samples.")
                                self.collector.start_recording(current_gest)
                        
                        if self.collector.is_recording:
                            self.collector.record(frame, features)
                            current_gesture = f"Recording {current_gest}: {self.collector.frame_counter}/50"
                            if self.collector.frame_counter == 1:
                                self.command_engine.say("Recording has started. Your samples are being collected.")
                            elif self.collector.frame_counter == 25:
                                self.command_engine.say("25 samples recorded.")
                            elif self.collector.frame_counter == 50:
                                self.collector.stop_recording()
                                self.command_engine.advance_dataset_collection()
                            
                    elif not self.is_recognizing:
                        current_gesture = "Navigation Mode"
                        if finger_count is not None:
                            self.navigation_manager.process_finger_count(finger_count)
                            cv2.putText(processed_frame, f"Fingers: {finger_count}", 
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                            
                    else:
                        # 3. Recognition Mode ML pipeline
                        pixel_lms = self.tracker.get_pixel_landmarks(frame, hand_no=0)
                        if pixel_lms:
                            wrist_x, wrist_y = pixel_lms[0][1], pixel_lms[0][2]
                            swipe = self.swipe_detector.add_position(wrist_x, wrist_y)
                            if swipe:
                                self.display_swipe = swipe
                                self.display_swipe_time = time.time()
                                if swipe == "swipe_right":
                                    self.on_intent("NEXT")
                                elif swipe == "swipe_left":
                                    self.on_intent("PREV")

                        if features is not None:
                            static_gesture, conf = self.classifier.predict(features)
                            
                            if self.display_swipe:
                                current_gesture = self.display_swipe
                                current_conf = 1.0
                            else:
                                current_gesture = static_gesture
                                current_conf = conf

                            if current_conf > 0.85:
                                if current_gesture != self.last_ml_gesture:
                                    self.ml_gesture_frames = 0
                                    self.last_ml_gesture = current_gesture
                                else:
                                    self.ml_gesture_frames += 1
                                    if self.ml_gesture_frames == 12:
                                        pretty_gest = current_gesture.replace('_', ' ').title()
                                        self.command_engine.say(f"{pretty_gest} recognized.")
                                        
                                        if current_gesture == "closed_fist":
                                            self.on_intent("BACK")
                            else:
                                self.ml_gesture_frames = 0
                                self.last_ml_gesture = None

                            cv2.putText(processed_frame, f"{current_gesture} ({current_conf:.2f})", 
                                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)

                # --- UPDATE FRIENDLY UI COMPONENTS (Non-Programmer Perspective) ---
                
                # 1. Update Friendly Status Banner
                if not self.hand_present:
                    self.status_banner.config(
                        text="👋 Please hold your hand up in front of the camera",
                        bg="#45475a", fg="#f9e2af"
                    )
                else:
                    if current_state == ApplicationState.DATASET_COLLECTING:
                        curr_g = GESTURES[self.command_engine.dataset_gesture_index].replace('_', ' ').title()
                        self.status_banner.config(
                            text=f"🔴 RECORDING: Hold '{curr_g}' steady ({self.collector.frame_counter}/50 photos)",
                            bg="#89b4fa", fg="#11111b"
                        )
                    elif current_state == ApplicationState.TRAINING:
                        self.status_banner.config(
                            text="🧠 AI is learning your gestures... Please wait a moment",
                            bg="#f9e2af", fg="#11111b"
                        )
                    elif current_state == ApplicationState.RECOGNITION:
                        self.status_banner.config(
                            text=f"✨ Live Recognition Active! Last Detected: {current_gesture.replace('_', ' ').title()}",
                            bg="#a6e3a1", fg="#11111b"
                        )
                    else:
                        self.status_banner.config(
                            text=f"✅ Hand Connected! Showing {finger_count} Finger{'s' if finger_count != 1 else ''}",
                            bg="#a6e3a1", fg="#11111b"
                        )

                # 2. Update Interactive Visual Menu Highlights
                current_sel_item = self.command_engine.main_menu_items[self.command_engine.main_menu_index]
                for item_name, lbl in self.menu_labels.items():
                    icon = self.menu_icons.get(item_name, "📌")
                    if current_state == ApplicationState.MAIN_MENU and item_name == current_sel_item:
                        # Highlighted active item
                        lbl.config(
                            text=f" 👉  {icon}  {item_name}   ◀ (Selected)",
                            font=("Segoe UI", 10, "bold"),
                            fg="#89b4fa", bg="#313244"
                        )
                    else:
                        lbl.config(
                            text=f"     {icon}  {item_name}",
                            font=("Segoe UI", 10),
                            fg="#6c7086", bg="#1e1e2e"
                        )

                # 3. Update Action Explanation Card
                if current_state == ApplicationState.DATASET_COLLECTING:
                    opt_info = {
                        "title": "📁 Dataset Collection in Progress",
                        "description": "Show the requested gesture clearly to camera. Samples are automatically photographed.",
                        "trigger": "Make a fist ✊ or say 'Stop Recording' to cancel"
                    }
                elif current_state == ApplicationState.RECOGNITION:
                    opt_info = {
                        "title": "✨ Live Hand Control Active",
                        "description": "Swipe your hand left/right or show gestures to control the app.",
                        "trigger": "Make a fist ✊ or say 'Stop Recognition' to exit to menu"
                    }
                else:
                    opt_info = self.command_engine.get_option_description(current_sel_item)

                self.action_title_lbl.config(text=opt_info["title"])
                self.action_desc_lbl.config(text=opt_info["description"])
                self.action_trigger_lbl.config(text=f"👉 {opt_info['trigger']}")

                # 4. Update Speech Bubble and Microphone Status
                spoken = self.command_engine.last_spoken_text
                if spoken:
                    # Truncate if too long for clean display
                    if len(spoken) > 80:
                        spoken = spoken[:77] + "..."
                    self.speech_bubble_lbl.config(text=f"💬 AI Voice: \"{spoken}\"")
                
                stt_text = self.speech_controller.last_parsed_text
                if self.voice_assistant.is_speaking:
                    self.mic_bubble_lbl.config(text="🎙️ Microphone: Paused while AI is talking (prevents echo)", fg="#f9e2af")
                elif stt_text:
                    self.mic_bubble_lbl.config(text=f"🎙️ Microphone: Heard \"{stt_text}\"", fg="#a6e3a1")
                else:
                    self.mic_bubble_lbl.config(text="🎙️ Microphone: Listening for voice commands...", fg="#a6adc8")
                
                # Render Camera to Tkinter with dynamic scaling to match current panel width
                panel_w = self.video_panel.winfo_width()
                if panel_w < 100:
                    panel_w = int(self.root.winfo_width() * self.camera_ratio)

                target_w = max(320, panel_w - 16)
                orig_h, orig_w = processed_frame.shape[:2]
                target_h = int(target_w * (orig_h / orig_w))

                panel_h = self.video_panel.winfo_height()
                if panel_h > 300:
                    max_h = max(240, panel_h - 170)
                    if target_h > max_h:
                        target_h = max_h
                        target_w = int(target_h * (orig_w / orig_h))

                disp_frame = cv2.resize(
                    processed_frame,
                    (target_w, target_h),
                    interpolation=cv2.INTER_AREA if target_w < orig_w else cv2.INTER_LINEAR
                )

                img_rgb = cv2.cvtColor(disp_frame, cv2.COLOR_BGR2RGB)
                img_pil = Image.fromarray(img_rgb)
                img_tk = ImageTk.PhotoImage(image=img_pil)
                self.video_label.imgtk = img_tk
                self.video_label.config(image=img_tk)

                if hasattr(self, 'cam_tip_text'):
                    self.cam_tip_text.config(wraplength=max(220, panel_w - 30))

                ctrl_w = self.control_frame.winfo_width()
                if ctrl_w > 100:
                    dyn_wrap = max(360, ctrl_w - 40)
                    if hasattr(self, 'status_banner'):
                        self.status_banner.config(wraplength=dyn_wrap)
                    if hasattr(self, 'action_desc_lbl'):
                        self.action_desc_lbl.config(wraplength=dyn_wrap)
                    if hasattr(self, 'speech_bubble_lbl'):
                        self.speech_bubble_lbl.config(wraplength=dyn_wrap)
                    if hasattr(self, 'mic_bubble_lbl'):
                        self.mic_bubble_lbl.config(wraplength=dyn_wrap)
                
        self.root.after(33, self.update_loop)

        
    def on_close(self):
        if hasattr(self, 'speech_controller') and self.speech_controller:
            self.speech_controller.stop_listening()
        self.camera.stop()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = HandGestureApp(root)
    root.mainloop()
