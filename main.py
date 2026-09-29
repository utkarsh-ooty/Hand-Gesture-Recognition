import os
import sys
import platform

# Suppress MediaPipe & TensorFlow C++ warning and error noise (including clearcut telemetry)
os.environ["GLOG_minloglevel"] = "3"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

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
        self.root.title("Hand Gesture AI")
        self.root.geometry("1180x680")
        self.root.minsize(1050, 620)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # ML / CV Modules
        self.camera = Camera(camera_index=0)
        self.tracker = HandTracker()
        self.collector = DatasetCollector()
        self.classifier = GestureClassifier()
        
        # Accessibility & State Modules
        self.state_manager = StateManager()
        self.voice_assistant = VoiceAssistant()
        self.finger_detector = FingerDetector()
        self.navigation_manager = NavigationManager(self.on_intent, stability_frames=12, cooldown=2.0)
        self.camera_ratio = 0.40  # Default: 40% camera, 60% commands & guide
        self._last_root_w = 0
        self.command_engine = CommandEngine(self.state_manager, self.voice_assistant, self._build_callbacks())
        
        # State
        self.is_camera_running = False
        self.is_recognizing = False
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
        self.command_engine.say("Your camera is now starting. Hand gestures are active.")
        self.last_startup_voice_time = time.time()
        
    def on_intent(self, intent, payload=None):
        """Callback from NavigationManager, SpeechController, OR GestureClassifier."""
        self.command_engine.process_intent(intent, payload)
        
    def _build_ui(self):
        # Configure root window dark theme
        self.root.configure(bg="#0A1128")
        
        # Horizontal PanedWindow allowing split-divider dragging between camera and guidance
        self.paned_window = tk.PanedWindow(
            self.root,
            orient=tk.HORIZONTAL,
            bg="#0A1128",
            sashwidth=6,
            sashrelief="flat",
            sashpad=2
        )
        self.paned_window.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # Left Panel (Camera Stream - Customizable 40% default width)
        self.video_panel = tk.Frame(self.paned_window, bg="#0A1128")
        self.paned_window.add(self.video_panel, minsize=260)
        
        cam_header = tk.Frame(self.video_panel, bg="#0A1128")
        cam_header.pack(fill=tk.X, pady=(0, 4))
        tk.Label(cam_header, text="📷 Camera View", font=("Segoe UI", 11, "bold"), fg="#3B82F6", bg="#0A1128").pack(side=tk.LEFT)
        self.cam_live_badge = tk.Label(cam_header, text="● LIVE", font=("Segoe UI", 9, "bold"), fg="#10B981", bg="#0A1128")
        self.cam_live_badge.pack(side=tk.RIGHT)

        # Camera Width Customizer Bar (Presets + Smooth Slider)
        size_bar = tk.Frame(self.video_panel, bg="#131B2F", padx=8, pady=4)
        size_bar.pack(fill=tk.X, pady=(0, 8))
        tk.Label(size_bar, text="📐 Width:", font=("Segoe UI", 9, "bold"), fg="#94A3B8", bg="#131B2F").pack(side=tk.LEFT, padx=(0, 6))

        self.ratio_btn_30 = tk.Button(size_bar, text="30%", font=("Segoe UI", 8), bg="#2C365E", fg="#F8FAFC", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.30))
        self.ratio_btn_30.pack(side=tk.LEFT, padx=2)

        self.ratio_btn_40 = tk.Button(size_bar, text="40%", font=("Segoe UI", 8, "bold"), bg="#3B82F6", fg="#0A1128", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.40))
        self.ratio_btn_40.pack(side=tk.LEFT, padx=2)

        self.ratio_btn_50 = tk.Button(size_bar, text="50%", font=("Segoe UI", 8), bg="#2C365E", fg="#F8FAFC", relief="flat", padx=6, pady=1, cursor="hand2", command=lambda: self.set_camera_ratio(0.50))
        self.ratio_btn_50.pack(side=tk.LEFT, padx=2)

        self.ratio_slider = tk.Scale(
            size_bar,
            from_=25, to=60,
            orient=tk.HORIZONTAL,
            showvalue=True,
            bg="#131B2F", fg="#CBD5E1",
            highlightthickness=0,
            troughcolor="#2C365E",
            activebackground="#3B82F6",
            length=80,
            command=self.on_slider_ratio
        )
        self.ratio_slider.set(40)
        self.ratio_slider.pack(side=tk.RIGHT)
        tk.Label(size_bar, text="Slide:", font=("Segoe UI", 8), fg="#475569", bg="#131B2F").pack(side=tk.RIGHT, padx=(4, 2))
        
        self.video_frame = tk.Frame(self.video_panel, bg="#000000", bd=2, relief="groove")
        self.video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 8))
        
        self.video_label = tk.Label(self.video_frame, bg="#000000")
        self.video_label.pack(expand=True)
        
        # Quick Camera Tip on Left Panel
        cam_tip_frame = tk.Frame(self.video_panel, bg="#1C2541", padx=10, pady=8)
        cam_tip_frame.pack(fill=tk.X, pady=(4, 0))
        tk.Label(cam_tip_frame, text="💡 Camera Tip:", font=("Segoe UI", 9, "bold"), fg="#F59E0B", bg="#1C2541").pack(anchor="w")
        self.cam_tip_text = tk.Label(
            cam_tip_frame,
            text="Hold hand 1-2 ft away. Spread fingers clearly. Drag divider or use buttons above to adjust view width.",
            font=("Segoe UI", 8), fg="#CBD5E1", bg="#1C2541", wraplength=260, justify="left"
        )
        self.cam_tip_text.pack(anchor="w", pady=(2, 0))

        # Right Panel (User Guidance & Controls - Takes 60% of Window)
        self.control_frame = tk.Frame(self.paned_window, bg="#131B2F", padx=20, pady=15)
        self.paned_window.add(self.control_frame, minsize=420)
        
        # --- 1. APP HEADER ---
        header_frame = tk.Frame(self.control_frame, bg="#131B2F")
        header_frame.pack(fill=tk.X, pady=(0, 8))
        
        tk.Label(header_frame, text="✨ Hand Gesture AI", font=("Segoe UI", 16, "bold"), fg="#F8FAFC", bg="#131B2F").pack(side=tk.LEFT)
        
        # --- 2. FRIENDLY STATUS BANNER (No programmer jargon) ---
        self.status_banner = tk.Label(
            self.control_frame,
            text="👋 Please hold your hand in front of the camera to begin",
            font=("Segoe UI", 11, "bold"),
            bg="#2C365E", fg="#F59E0B",
            padx=14, pady=9, wraplength=460
        )
        self.status_banner.pack(fill=tk.X, pady=(0, 10))

        # --- 3. INTERACTIVE VISUAL MENU (Shows users where they are) ---
        menu_card = tk.LabelFrame(
            self.control_frame,
            text=" 📋 Menu Options ",
            font=("Segoe UI", 10, "bold"),
            fg="#3B82F6", bg="#1C2541",
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
                fg="#94A3B8", bg="#1C2541",
                anchor="w", padx=12, pady=6,
                cursor="hand2",
                relief="flat",
                bd=0
            )
            lbl.pack(fill=tk.X, pady=2, padx=4)
            lbl.bind("<Button-1>", lambda e, i=item: self._on_menu_click(i))
            lbl.bind("<Enter>", lambda e, i=item, l=lbl: self._on_menu_hover(i, l))
            lbl.bind("<Leave>", lambda e, i=item, l=lbl: self._on_menu_leave(i, l))
            self.menu_labels[item] = lbl

        # --- 4. ACTION EXPLANATION CARD ("What will this do?") ---
        action_card = tk.LabelFrame(
            self.control_frame,
            text=" 💡 What Does This Option Do? ",
            font=("Segoe UI", 10, "bold"),
            fg="#10B981", bg="#1C2541",
            padx=14, pady=10, bd=1
        )
        action_card.pack(fill=tk.X, pady=(0, 10))
        
        self.action_title_lbl = tk.Label(action_card, text="📁 Dataset Collection", font=("Segoe UI", 13, "bold"), fg="#3B82F6", bg="#1C2541")
        self.action_title_lbl.pack(anchor="w", pady=(0, 2))
        
        self.action_desc_lbl = tk.Label(
            action_card,
            text="Records camera sample photos of your hand gestures so the AI learns how your hand looks.",
            font=("Segoe UI", 10),
            fg="#F8FAFC", bg="#1C2541",
            wraplength=460, justify="left"
        )
        self.action_desc_lbl.pack(anchor="w", pady=(0, 6))
        
        self.action_trigger_lbl = tk.Label(
            action_card,
            text="👉 TO SELECT: Thumbs Up 🖒  OR  Click",
            font=("Segoe UI", 10, "bold"),
            fg="#F59E0B", bg="#1C2541",
            justify="left"
        )
        self.action_trigger_lbl.pack(anchor="w")

        # --- 5. HOW TO CONTROL (Cheatsheet at a glance) ---
        guide_card = tk.LabelFrame(
            self.control_frame,
            text=" 🎮 Quick Controls Guide ",
            font=("Segoe UI", 10, "bold"),
            fg="#8B5CF6", bg="#1C2541",
            padx=12, pady=8, bd=1
        )
        guide_card.pack(fill=tk.X, pady=(0, 10))
        
        guide_col1 = (
            "☝️ 1 Finger: Scroll Next\n"
            "✌️ 2 Fingers: Scroll Previous\n"
            "🖒 Thumbs Up: Select / Confirm\n"
            "👎 Thumbs Down: Go Back / Exit"
        )
        guide_col2 = (
            "Mouse clicks are also fully supported."
        )
        g_row = tk.Frame(guide_card, bg="#1C2541")
        g_row.pack(fill=tk.X)
        tk.Label(g_row, text=guide_col1, font=("Segoe UI", 9), fg="#CBD5E1", bg="#1C2541", justify="left").pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Label(g_row, text=guide_col2, font=("Segoe UI", 9), fg="#CBD5E1", bg="#1C2541", justify="left").pack(side=tk.LEFT, expand=True, fill=tk.X)

        # --- 6. AI ASSISTANT & VOICE BUBBLE ---
        voice_card = tk.Frame(self.control_frame, bg="#131B2F")
        voice_card.pack(fill=tk.X, pady=(4, 0))
        
        self.speech_bubble_lbl = tk.Label(
            voice_card,
            text="💬 AI Assistant: \"Welcome. I'm ready to guide you.\"",
            font=("Segoe UI", 9, "italic"),
            fg="#b4befe", bg="#131B2F",
            wraplength=460, justify="left"
        )
        self.speech_bubble_lbl.pack(anchor="w")

        # Hidden variables for backwards compatibility with existing callbacks
        self.gesture_class_var = tk.StringVar(value=GESTURES[0])
        self.status_label = tk.Label(self.root, text="")

        # Dynamic Window resize & sash ratio initialization
        self.root.bind("<Configure>", self._on_window_configure)
        self.root.after(150, lambda: self.set_camera_ratio(0.40))

    # --- Menu Interaction Handlers ---
    def _on_menu_click(self, item_name):
        """Mouse click on a menu option: sync index and execute."""
        items = self.command_engine.main_menu_items
        if item_name in items:
            self.command_engine.main_menu_index = items.index(item_name)
        self.command_engine._execute_main_menu_selection(override=item_name)

    def _on_menu_hover(self, item_name, lbl):
        """Mouse enters a menu option: glow highlight + preview action card."""
        icon = self.menu_icons.get(item_name, "\ud83d\udccc")
        lbl.config(
            text=f" \u25b6  {icon}  {item_name}",
            font=("Segoe UI", 11, "bold"),
            fg="#F8FAFC", bg="#3B82F6",
            relief="flat"
        )
        # Live-preview this option in the action card
        opt_info = self.command_engine.get_option_description(item_name)
        if hasattr(self, 'action_title_lbl'):
            self.action_title_lbl.config(text=opt_info["title"])
            self.action_desc_lbl.config(text=opt_info["description"])
            self.action_trigger_lbl.config(text=f"\ud83d\udc49 {opt_info['trigger']}")

    def _on_menu_leave(self, item_name, lbl):
        """Mouse leaves a menu option: reset to default style."""
        icon = self.menu_icons.get(item_name, "\ud83d\udccc")
        lbl.config(
            text=f"  {icon}  {item_name}",
            font=("Segoe UI", 10),
            fg="#94A3B8", bg="#1C2541",
            relief="flat"
        )

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
                    btn.config(bg="#3B82F6", fg="#0A1128", font=("Segoe UI", 8, "bold"))
                else:
                    btn.config(bg="#2C365E", fg="#F8FAFC", font=("Segoe UI", 8))

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
                nav_token = None
                
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
                    
                    nav_token = self.finger_detector.get_navigation_token(landmarks, hand_category)
                        
                    cv2.putText(processed_frame, f"Hands: 1 | Mode: {nav_token or 'NONE'}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
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
                        if nav_token is not None:
                            self.navigation_manager.process_finger_count(nav_token)
                            cv2.putText(processed_frame, f"Token: {nav_token}", 
                                        (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)
                            
                    else:
                        # 3. Recognition Mode ML pipeline
                        
                        if features is not None:
                            static_gesture, conf = self.classifier.predict(features)
                            
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
                                        
                                        if current_gesture == "thumbs_down":
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
                        bg="#334155", fg="#F59E0B"
                    )
                else:
                    if current_state == ApplicationState.DATASET_COLLECTING:
                        curr_g = GESTURES[self.command_engine.dataset_gesture_index].replace('_', ' ').title()
                        self.status_banner.config(
                            text=f"🔴 RECORDING: Hold '{curr_g}' steady ({self.collector.frame_counter}/50 photos)",
                            bg="#3B82F6", fg="#0A1128"
                        )
                    elif current_state == ApplicationState.TRAINING:
                        self.status_banner.config(
                            text="🧠 AI is learning your gestures... Please wait a moment",
                            bg="#F59E0B", fg="#0A1128"
                        )
                    elif current_state == ApplicationState.RECOGNITION:
                        self.status_banner.config(
                            text=f"✨ Live Recognition Active! Last Detected: {current_gesture.replace('_', ' ').title()}",
                            bg="#10B981", fg="#0A1128"
                        )
                    else:
                        if nav_token:
                            nav_pretty = nav_token.replace('_', ' ')
                            self.status_banner.config(
                                text=f"✅ Hand Connected! Detected: {nav_pretty}",
                                bg="#10B981", fg="#0A1128"
                            )
                        else:
                            self.status_banner.config(
                                text=f"✅ Hand Connected! Show a gesture to navigate...",
                                bg="#10B981", fg="#0A1128"
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
                            fg="#3B82F6", bg="#2C365E"
                        )
                    else:
                        lbl.config(
                            text=f"     {icon}  {item_name}",
                            font=("Segoe UI", 10),
                            fg="#475569", bg="#1C2541"
                        )

                # 3. Update Action Explanation Card
                if current_state == ApplicationState.DATASET_COLLECTING:
                    opt_info = {
                        "title": "📁 Dataset Collection in Progress",
                        "description": "Show the requested gesture clearly to camera. Samples are automatically photographed.",
                        "trigger": "Thumbs Down 👎 to cancel"
                    }
                elif current_state == ApplicationState.RECOGNITION:
                    opt_info = {
                        "title": "✨ Live Hand Control Active",
                        "description": "Show trained gestures in front of your camera to control the app.",
                        "trigger": "Thumbs Down 👎 to exit to menu"
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
                
                if hasattr(self, 'mic_bubble_lbl'):
                    self.mic_bubble_lbl.config(text="🎙️ Voice navigation removed in this version.", fg="#475569")
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
