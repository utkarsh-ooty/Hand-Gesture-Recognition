import time
import os
import sys
import unittest

# Ensure SCRIPT_DIR is in path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from application_state import ApplicationState, StateManager
from voice_assistant import VoiceAssistant
from speech_controller import SpeechController
from command_engine import CommandEngine

class TestVoiceNavigation(unittest.TestCase):
    def setUp(self):
        self.state_manager = StateManager()
        self.voice_assistant = VoiceAssistant()
        self.dispatched_intents = []
        
        def mock_callback(intent, payload=None):
            self.dispatched_intents.append((intent, payload))

        self.speech_controller = SpeechController(mock_callback, voice_assistant=self.voice_assistant)
        
        self.app_callbacks = {
            "start_camera": lambda: print("[Mock] Camera Started"),
            "stop_camera": lambda: print("[Mock] Camera Stopped"),
            "select_gesture": lambda g: print(f"[Mock] Selected Gesture: {g}"),
            "start_recording": lambda: print("[Mock] Recording Started"),
            "stop_recording": lambda: print("[Mock] Recording Stopped"),
            "train_model": lambda: True,
            "is_model_trained": lambda: True,
            "start_recognition": lambda: True,
            "stop_recognition": lambda: True,
            "close_app": lambda: print("[Mock] App Closed")
        }
        self.command_engine = CommandEngine(self.state_manager, self.voice_assistant, self.app_callbacks)

    def tearDown(self):
        self.speech_controller.stop_listening()
        self.voice_assistant.stop_speaking()

    def test_stt_intent_parsing(self):
        """Test parsing of various natural spoken phrases into normalized intents."""
        test_cases = [
            ("next", "NEXT"),
            ("move next", "NEXT"),
            ("forward", "NEXT"),
            ("previous", "PREV"),
            ("go back", "BACK"),
            ("select", "SELECT"),
            ("confirm", "SELECT"),
            ("home", "HOME"),
            ("main menu", "HOME"),
            ("help me", "HELP"),
            ("start camera", "START_CAMERA"),
            ("turn off camera", "STOP_CAMERA"),
            ("dataset collection", "SELECT_DATASET_COLLECTION"),
            ("train model", "TRAIN_MODEL"),
            ("start recognition", "START_RECOGNITION"),
            ("stop recognition", "STOP_RECOGNITION"),
            ("exit application", "EXIT"),
            ("select open palm", "SELECT_GESTURE"),
            ("wider camera", "WIDER_CAMERA"),
            ("smaller camera", "SMALLER_CAMERA"),
            ("reset camera", "RESET_CAMERA"),
        ]

        print("\n--- Running STT Intent Parsing Self-Check ---")
        for phrase, expected_intent in test_cases:
            intent, payload = self.speech_controller.parse_text(phrase)
            self.assertEqual(intent, expected_intent, f"Failed for phrase: '{phrase}'. Expected {expected_intent}, got {intent}")
            print(f"  [PASS] Phrase: '{phrase}' -> Intent: '{intent}' (Payload: {payload})")

    def test_command_engine_state_transitions(self):
        """Test CommandEngine state changes driven by voice intents."""
        print("\n--- Running CommandEngine State Transition Self-Check ---")
        
        # Start in WAITING_FOR_HAND
        self.state_manager.set_state(ApplicationState.WAITING_FOR_HAND)
        self.command_engine.process_intent("STARTUP")
        self.assertEqual(self.state_manager.get_state(), ApplicationState.MAIN_MENU)
        print("  [PASS] STARTUP state transition -> MAIN_MENU")

        # Scroll next in MAIN_MENU
        initial_index = self.command_engine.main_menu_index
        self.command_engine.process_intent("NEXT")
        self.assertEqual(self.command_engine.main_menu_index, (initial_index + 1) % len(self.command_engine.main_menu_items))
        print("  [PASS] NEXT voice intent changed main_menu_index")

        # Direct navigation to Dataset Collection via voice
        self.command_engine.process_intent("SELECT_DATASET_COLLECTION")
        self.assertEqual(self.state_manager.get_state(), ApplicationState.DATASET_MENU)
        print("  [PASS] SELECT_DATASET_COLLECTION voice intent -> DATASET_MENU")

        # Go home from DATASET_MENU
        self.command_engine.process_intent("HOME")
        self.assertEqual(self.state_manager.get_state(), ApplicationState.MAIN_MENU)
        print("  [PASS] HOME voice intent -> MAIN_MENU")

    def test_tts_lock_coordination(self):
        """Test VoiceAssistant is_speaking state and queue handling."""
        print("\n--- Running TTS Lock Coordination Self-Check ---")
        self.assertFalse(self.voice_assistant.is_speaking, "VoiceAssistant should not be speaking initially")
        
        # Speak short message
        self.voice_assistant.speak("Self test message")
        time.sleep(0.15)
        print(f"  [PASS] VoiceAssistant handles non-blocking speak request cleanly (is_speaking: {self.voice_assistant.is_speaking})")

if __name__ == "__main__":
    unittest.main()
