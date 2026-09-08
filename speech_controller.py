import threading
import speech_recognition as sr
import time

class SpeechController:
    def __init__(self, command_callback):
        self.recognizer = sr.Recognizer()
        self.mic = sr.Microphone()
        self.is_listening = False
        self.command_callback = command_callback # Callback fired when a normalized intent is resolved
        self.listen_thread = None
        
        # Adjust for ambient noise on startup
        with self.mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=1.0)
            
    def start_listening(self):
        if self.is_listening:
            return
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
    def stop_listening(self):
        self.is_listening = False
        
    def _listen_loop(self):
        while self.is_listening:
            try:
                with self.mic as source:
                    # Listen for phrases with a short timeout to prevent hanging forever
                    audio = self.recognizer.listen(source, timeout=2.0, phrase_time_limit=5.0)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[STT Loop Error] {e}")
                time.sleep(1) # Prevent spamming error
                continue
                
            if not self.is_listening:
                break
                
            try:
                text = self.recognizer.recognize_google(audio).lower().strip()
                print(f"[STT] Parsed: '{text}'")
                self._parse_and_dispatch(text)
            except sr.UnknownValueError:
                print("[STT] Speech unintelligible")
            except sr.RequestError as e:
                print(f"[STT] API Request Error: {e}")

    def _parse_and_dispatch(self, text: str):
        """Converts raw text variations into normalized intents and dispatches."""
        intent = None
        payload = None
        
        # --- Camera ---
        if any(w in text for w in ["start camera", "turn on camera", "open camera", "begin camera"]):
            intent = "START_CAMERA"
        elif any(w in text for w in ["stop camera", "turn off camera", "close camera", "end camera"]):
            intent = "STOP_CAMERA"
            
        # --- Recording ---
        elif any(w in text for w in ["start recording", "begin recording", "record samples"]):
            intent = "START_RECORDING"
        elif any(w in text for w in ["stop recording", "end recording"]):
            intent = "STOP_RECORDING"
            
        # --- Training ---
        elif any(w in text for w in ["train model", "start training", "train the model", "train"]):
            intent = "TRAIN_MODEL"
            
        # --- Recognition ---
        elif any(w in text for w in ["start recognition", "begin recognition"]):
            intent = "START_RECOGNITION"
        elif any(w in text for w in ["stop recognition", "end recognition"]):
            intent = "STOP_RECOGNITION"
        elif any(w in text for w in ["reload model"]):
            intent = "RELOAD_MODEL"
            
        # --- Help / Accessibility ---
        elif text == "help" or text == "help me":
            intent = "HELP"
        elif text == "repeat" or text == "repeat that":
            intent = "REPEAT"
            
        # --- Confirmation ---
        elif text in ["yes", "confirm", "yeah", "yep"]:
            intent = "YES"
        elif text in ["no", "cancel", "nope"]:
            intent = "NO"
            
        # --- Gestures Selection ---
        elif "select" in text:
            from gesture_config import GESTURES
            for gest in GESTURES:
                # convert "open_palm" to "open palm" etc. for exact matching
                normalized_gest = gest.replace("_", " ")
                if normalized_gest in text:
                    intent = "SELECT_GESTURE"
                    payload = gest
                    break
        
        if intent:
            self.command_callback(intent, payload)
