"""
Application Controller for Hand Gesture & Voice Navigation AI.
Coordinates UI, gesture recognition pipeline, speech assistant, navigation, and model lifecycles.
"""

import os
import time
import tkinter as tk
import cv2
from typing import Optional
import numpy as np

from app.config import Config
from app.state_manager import ApplicationState, StateManager
from gestures.camera_manager import CameraManager
from gestures.hand_tracker import HandTracker
from gestures.feature_extractor import FeatureExtractor
from gestures.finger_detector import FingerDetector
from gestures.gesture_classifier import GestureClassifier
from gestures.gesture_mapper import GestureMapper
from gestures.gesture_controller import GestureController
from voice.speech_manager import SpeechManager
from navigation.navigation_controller import NavigationController
from ml.dataset_manager import DatasetManager
from ml.model_trainer import ModelTrainer
from ml.model_loader import ModelLoader
from ui.main_window import MainWindow
from ui.styles import (
    BG_CARD, BG_FOCUS, FG_WARN, FG_PRIMARY, FG_SUCCESS, FG_ACCENT,
    BG_ALERT_INTERRUPT, BG_ALERT_SUCCESS
)


class AppController:
    """Central Application Controller owning all subsystem components and the runtime loop."""

    def __init__(self, root: tk.Tk):
        self.root = root

        # 1. State & Speech
        self.state_manager = StateManager(ApplicationState.STARTUP)
        self.speech_manager = SpeechManager(
            sapi_rate=Config.SAPI_RATE,
            pyttsx3_rate=Config.PYTTSX3_RATE,
            queue_maxsize=Config.SPEECH_QUEUE_MAXSIZE
        )

        # 2. ML & Datasets
        self.dataset_manager = DatasetManager()
        self.model_trainer = ModelTrainer()
        self.classifier = GestureClassifier()

        # 3. Gesture recognition & vision
        self.camera_manager = CameraManager()
        self.hand_tracker = HandTracker()
        self.finger_detector = FingerDetector()
        self.gesture_mapper = GestureMapper()
        self.gesture_controller = GestureController(
            callback=self._on_gesture_intent,
            stability_frames=Config.STABILITY_FRAMES,
            cooldown=Config.COOLDOWN_SECONDS,
            fast_interrupt_frames=Config.FAST_INTERRUPT_FRAMES
        )

        # 4. Navigation
        self.navigation_controller = NavigationController(
            state_manager=self.state_manager,
            speech_manager=self.speech_manager,
            app_callbacks=self._build_callbacks()
        )

        # 5. UI View
        self.main_window = MainWindow(
            root=self.root,
            on_menu_click=self.navigation_controller.handle_mouse_click,
            on_menu_hover=self.navigation_controller.handle_mouse_hover,
            on_close=self.shutdown
        )

        # 6. Runtime tracking flags
        self.is_camera_running: bool = False
        self.is_recognizing: bool = False
        self.hand_present: bool = False
        self.hand_lost_time: float = 0.0
        self.last_no_hand_speech: float = 0.0

        # Dataset collection index
        self.dataset_gesture_index: int = 0

        # Recognition mode smoothing
        self.last_ml_gesture: Optional[str] = None
        self.ml_gesture_frames: int = 0

        # Startup delayed sequence
        self.root.after(800, self._startup)
        self.update_loop()

    # ==================================================================
    # Application Callbacks
    # ==================================================================

    def _build_callbacks(self):
        return {
            "start_camera": self._start_camera,
            "stop_camera": self._stop_camera,
            "start_dataset_collection": self._start_dataset_collection,
            "stop_recording": self.dataset_manager.stop_recording,
            "train_model": self._train_model,
            "is_model_trained": self._is_model_trained,
            "get_missing_samples": self.dataset_manager.get_missing_samples,
            "start_recognition": self._start_recognition,
            "stop_recognition": self._stop_recognition,
            "reload_model": self.classifier.load_model,
            "close_app": self.shutdown,
        }

    # ==================================================================
    # Lifecycle & Startup
    # ==================================================================

    def _startup(self):
        self.state_manager.set_state(ApplicationState.WAITING_FOR_HAND)
        self._start_camera()
        self.speech_manager.speak("Welcome.", force=True)

    def _start_camera(self):
        try:
            self.camera_manager.start()
            self.is_camera_running = True
        except Exception as e:
            print(f"[AppController] Camera start error: {e}")
            self.speech_manager.speak("Error. Camera unavailable.")

    def _stop_camera(self):
        self.is_camera_running = False
        self.camera_manager.stop()

    def shutdown(self):
        """Clean and deterministic shutdown."""
        try:
            self._stop_camera()
        except Exception:
            pass
        try:
            self.speech_manager.shutdown()
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    # ==================================================================
    # ML & Recognition Callbacks
    # ==================================================================

    def _train_model(self) -> bool:
        success = self.model_trainer.train(message_callback=self.speech_manager.speak)
        if success:
            self.classifier.load_model()
        return success

    def _is_model_trained(self) -> bool:
        return ModelLoader.is_model_trained()

    def _start_recognition(self) -> bool:
        self.classifier.load_model()
        if not self.classifier.is_loaded:
            return False
        self.is_recognizing = True
        return True

    def _stop_recognition(self):
        self.is_recognizing = False

    def _start_dataset_collection(self):
        self.dataset_gesture_index = 0
        self._prompt_next_dataset_gesture()

    def _prompt_next_dataset_gesture(self):
        gesture_name = Config.GESTURES[self.dataset_gesture_index]
        pretty = gesture_name.replace('_', ' ').title()
        self.speech_manager.speak(f"Show {pretty}.")

    def _advance_dataset_collection(self):
        current_gest = Config.GESTURES[self.dataset_gesture_index]
        pretty = current_gest.replace('_', ' ').title()
        self.speech_manager.speak(f"{pretty} done.")

        self.dataset_gesture_index += 1
        if self.dataset_gesture_index < len(Config.GESTURES):
            self._prompt_next_dataset_gesture()
        else:
            self.dataset_gesture_index = 0
            self.state_manager.set_state(ApplicationState.GESTURE_SETUP)
            self.speech_manager.speak("Dataset complete.")

    # ==================================================================
    # Gesture Dispatch Handler
    # ==================================================================

    def _on_gesture_intent(self, stable_intent: str):
        """Called when GestureController confirms a stable, edge-triggered intent."""
        self.navigation_controller.process_intent(stable_intent)

    # ==================================================================
    # Main Processing Loop (~30 FPS)
    # ==================================================================

    def update_loop(self):
        try:
            self._process_frame()
        except Exception as e:
            print(f"[AppController] Update loop error: {e}")
        self.root.after(Config.FRAME_RATE_MS, self.update_loop)

    def _process_frame(self):
        if not self.is_camera_running:
            return

        frame = self.camera_manager.read_frame()
        if frame is None:
            return

        processed_frame = self.hand_tracker.find_hands(frame.copy(), draw=True)
        landmarks = self.hand_tracker.get_landmarks(frame, hand_no=0)
        current_state = self.state_manager.get_state()

        nav_intent: Optional[str] = None

        if not landmarks:
            # ── Hand lost ──
            self.gesture_controller.process_intent(None)

            if self.hand_present:
                self.hand_present = False
                self.hand_lost_time = time.time()

            time_lost = time.time() - self.hand_lost_time if self.hand_lost_time else 0.0

            if time_lost > 4.0 and (time.time() - self.last_no_hand_speech > 10.0):
                if current_state not in (ApplicationState.STARTUP, ApplicationState.WAITING_FOR_HAND):
                    self.speech_manager.speak("No hand detected.")
                    self.last_no_hand_speech = time.time()

            if current_state == ApplicationState.WAITING_FOR_HAND:
                if time_lost > 6.0 and (time.time() - self.last_no_hand_speech > 8.0):
                    self.speech_manager.speak("Show your hand.")
                    self.last_no_hand_speech = time.time()

        else:
            # ── Hand found ──
            if not self.hand_present:
                self.hand_present = True
                self.hand_lost_time = 0.0
                self.last_no_hand_speech = 0.0

            if current_state == ApplicationState.WAITING_FOR_HAND:
                self.navigation_controller.process_intent("STARTUP")
                return

            hand_category, _ = self.hand_tracker.get_handedness(0)
            nav_intent = self.finger_detector.get_navigation_intent(landmarks, hand_category)

            # Route by current mode
            if current_state == ApplicationState.DATASET_COLLECTING:
                self._handle_dataset_frame(landmarks, frame, processed_frame)
            elif self.is_recognizing and current_state == ApplicationState.RECOGNITION:
                self._handle_recognition_frame(landmarks, frame, processed_frame)
                if nav_intent in ("STOP_VOICE", "BACK"):
                    self.gesture_controller.process_intent(nav_intent)
            else:
                self.gesture_controller.process_intent(nav_intent)

            if nav_intent:
                cv2.putText(processed_frame, nav_intent,
                            (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

        # ── Synchronize View Updates ──
        self._sync_view(current_state, nav_intent, processed_frame)

    def _handle_dataset_frame(self, landmarks, frame, processed_frame):
        current_gest = Config.GESTURES[self.dataset_gesture_index]

        if self.finger_detector.check_gesture_match(current_gest, landmarks):
            if not self.dataset_manager.is_recording:
                pretty = current_gest.replace('_', ' ').title()
                self.speech_manager.speak(f"{pretty} detected.")
                self.dataset_manager.start_recording(current_gest)

        if self.dataset_manager.is_recording:
            features = FeatureExtractor.extract_features(landmarks)
            self.dataset_manager.record(frame, features)

            if self.dataset_manager.frame_counter == 25:
                self.speech_manager.speak("Halfway.")
            elif self.dataset_manager.frame_counter >= 50:
                self.dataset_manager.stop_recording()
                self._advance_dataset_collection()

    def _handle_recognition_frame(self, landmarks, frame, processed_frame):
        features = FeatureExtractor.extract_features(landmarks)
        if features is None:
            return

        gesture, conf = self.classifier.predict(features)

        if conf > 0.85:
            if gesture != self.last_ml_gesture:
                self.ml_gesture_frames = 0
                self.last_ml_gesture = gesture
            else:
                self.ml_gesture_frames += 1
                if self.ml_gesture_frames == 10:
                    pretty = gesture.replace('_', ' ').title()
                    self.speech_manager.speak(f"{pretty}.")
        else:
            self.ml_gesture_frames = 0
            self.last_ml_gesture = None

        cv2.putText(processed_frame, f"{gesture} ({conf:.0%})",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # ==================================================================
    # View Synchronization
    # ==================================================================

    def _sync_view(self, state: ApplicationState, nav_intent: Optional[str],
                   processed_frame: np.ndarray):
        # 1. Status Banner
        if self.speech_manager.is_recently_interrupted:
            self.main_window.update_status_banner(
                "SPEECH STOPPED", bg=BG_ALERT_INTERRUPT, fg=FG_PRIMARY
            )
        elif not self.hand_present:
            self.main_window.update_status_banner(
                "Show your hand in front of the camera", bg=BG_CARD, fg=FG_WARN
            )
        elif state == ApplicationState.DATASET_COLLECTING:
            g = Config.GESTURES[self.dataset_gesture_index].replace('_', ' ').title()
            self.main_window.update_status_banner(
                f"Recording: {g}  ({self.dataset_manager.frame_counter}/50)",
                bg=BG_FOCUS, fg=FG_PRIMARY
            )
        elif state == ApplicationState.TRAINING:
            self.main_window.update_status_banner(
                "Training model... please wait", bg=FG_WARN, fg=BG_CARD
            )
        elif state == ApplicationState.RECOGNITION:
            gest = (self.last_ml_gesture or "Waiting").replace('_', ' ').title()
            self.main_window.update_status_banner(
                f"Recognition active  —  {gest}", bg=BG_ALERT_SUCCESS, fg=FG_SUCCESS
            )
        elif state == ApplicationState.HELP:
            self.main_window.update_status_banner(
                "Help  —  Fist to go back, Open palm to stop speech", bg=BG_CARD, fg=FG_ACCENT
            )
        else:
            if nav_intent and nav_intent != "STOP_VOICE":
                self.main_window.update_status_banner(
                    f"Hand detected  —  {nav_intent}", bg=BG_ALERT_SUCCESS, fg=FG_SUCCESS
                )
            else:
                self.main_window.update_status_banner(
                    "Hand detected  —  Show a gesture", bg=BG_ALERT_SUCCESS, fg=FG_SUCCESS
                )

        # 2. Menu Highlight
        focused = self.navigation_controller.focus_manager.get_focused_item() or ""
        self.main_window.update_menu_highlight(focused)

        # 3. Description Card
        if state in (ApplicationState.MAIN_MENU, ApplicationState.GESTURE_SETUP):
            desc_info = self.navigation_controller.get_option_description(focused)
        elif state == ApplicationState.DATASET_COLLECTING:
            desc_info = {
                "title": "Dataset Collection",
                "description": "Hold the requested gesture in front of the camera.",
                "trigger": "Fist to cancel"
            }
        elif state == ApplicationState.RECOGNITION:
            desc_info = {
                "title": "Live Recognition",
                "description": "Show trained gestures to control the app.",
                "trigger": "Fist to exit"
            }
        elif state == ApplicationState.HELP:
            desc_info = {
                "title": "Help",
                "description": "1 finger = Next option\n2 fingers = Previous option\n3 fingers = Select\nFist = Back / Cancel\nOpen palm = Stop speech",
                "trigger": "Fist to go back"
            }
        else:
            desc_info = {"title": "", "description": "", "trigger": ""}

        self.main_window.update_description(
            desc_info.get("title", ""), desc_info.get("description", "")
        )

        # 4. Voice Bubble
        self.main_window.update_voice_bubble(
            self.speech_manager.last_spoken_text,
            is_interrupted=self.speech_manager.is_recently_interrupted
        )

        # 5. Camera Status & Video Rendering
        self.main_window.update_camera_status(self.is_camera_running, self.hand_present)
        self.main_window.render_camera(processed_frame)
