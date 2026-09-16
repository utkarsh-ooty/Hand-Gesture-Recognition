import threading
import speech_recognition as sr
import time

class SpeechController:
    def __init__(self, command_callback, voice_assistant=None):
        self.recognizer = sr.Recognizer()
        self.mic = None
        self.use_sounddevice_fallback = False
        self.is_listening = False
        self.command_callback = command_callback # Callback fired when a normalized intent is resolved
        self.voice_assistant = voice_assistant
        self.listen_thread = None
        self.last_parsed_text = ""
        self.last_intent = ""
        
        # Microphones initialization safety
        try:
            self.mic = sr.Microphone()
            with self.mic as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.8)
            print("[SpeechController] PyAudio microphone initialized successfully.")
        except Exception as e:
            print(f"[SpeechController] PyAudio mic not available ({e}). Using sounddevice fallback.")
            self.mic = None
            try:
                import sounddevice as sd
                self.use_sounddevice_fallback = True
            except Exception as sd_e:
                print(f"[SpeechController] sounddevice fallback unavailable: {sd_e}")
                self.use_sounddevice_fallback = False

    def start_listening(self):
        if self.is_listening:
            return
        if self.mic is None and not self.use_sounddevice_fallback:
            return
        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
    def stop_listening(self):
        self.is_listening = False

    def _listen_loop(self):
        import sounddevice as sd

        while self.is_listening:
            # Self-Hearing Prevention: pause listening if VoiceAssistant is actively speaking
            if self.voice_assistant and self.voice_assistant.is_speaking:
                time.sleep(0.3)
                continue

            audio = None
            try:
                if self.mic is not None:
                    with self.mic as source:
                        audio = self.recognizer.listen(source, timeout=2.0, phrase_time_limit=4.0)
                elif self.use_sounddevice_fallback:
                    sample_rate = 16000
                    duration = 3.5
                    rec = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
                    sd.wait()
                    audio = sr.AudioData(rec.tobytes(), sample_rate, 2)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"[SpeechController Loop Error] {e}")
                time.sleep(1.0)
                continue
                
            if not self.is_listening or audio is None:
                break
                
            if self.voice_assistant and self.voice_assistant.is_speaking:
                continue

            try:
                text = self.recognizer.recognize_google(audio).lower().strip()
                self.last_parsed_text = text
                print(f"[SpeechController STT] Parsed: '{text}'")
                self._parse_and_dispatch(text)
            except sr.UnknownValueError:
                pass
            except sr.RequestError as e:
                print(f"[SpeechController STT] API Request Error: {e}")

    def parse_text(self, text: str):
        """Helper method for manual/testing intent parsing."""
        return self._parse_and_dispatch(text)

    def _parse_and_dispatch(self, text: str):
        """Converts raw text variations into normalized intents and dispatches."""
        intent = None
        payload = None
        text_clean = text.lower().strip()
        
        # --- App Control & Navigation ---
        if any(w in text_clean for w in ["start camera", "turn on camera", "open camera", "enable camera"]):
            intent = "START_CAMERA"
        elif any(w in text_clean for w in ["stop camera", "turn off camera", "close camera", "disable camera"]):
            intent = "STOP_CAMERA"
        elif any(w in text_clean for w in ["wider camera", "larger camera", "expand camera", "bigger camera", "widen camera"]):
            intent = "WIDER_CAMERA"
        elif any(w in text_clean for w in ["smaller camera", "narrower camera", "shrink camera"]):
            intent = "SMALLER_CAMERA"
        elif any(w in text_clean for w in ["reset camera", "default camera", "normal camera"]):
            intent = "RESET_CAMERA"

        # --- Specific Gestures Selection (Checked before generic 'select') ---
        elif "select" in text_clean:
            from gesture_config import GESTURES
            for gest in GESTURES:
                normalized_gest = gest.replace("_", " ")
                if normalized_gest in text_clean:
                    intent = "SELECT_GESTURE"
                    payload = gest
                    break

        # --- Menu Navigation Intents ---
        if intent is None:
            if any(w == text_clean or w in text_clean for w in ["next", "forward", "move next", "go next", "swipe right"]):
                intent = "NEXT"
            elif any(w == text_clean or w in text_clean for w in ["previous", "prev", "backwards", "move back", "swipe left"]):
                intent = "PREV"
            elif any(w == text_clean or w in text_clean for w in ["select", "choose", "confirm", "enter", "pick"]):
                intent = "SELECT"
            elif any(w == text_clean or w in text_clean for w in ["back", "go back", "closed fist", "cancel section"]):
                intent = "BACK"
            elif any(w == text_clean or w in text_clean for w in ["home", "main menu", "return home", "go home"]):
                intent = "HOME"
            elif any(w == text_clean or w in text_clean for w in ["help", "help me", "instructions"]):
                intent = "HELP"

        # --- Section Specific ---
        if intent is None:
            if any(w in text_clean for w in ["dataset collection", "collect dataset", "record dataset", "dataset menu"]):
                intent = "SELECT_DATASET_COLLECTION"
            elif any(w in text_clean for w in ["start recording", "begin recording", "record samples"]):
                intent = "START_RECORDING"
            elif any(w in text_clean for w in ["stop recording", "end recording", "cancel recording"]):
                intent = "STOP_RECORDING"
                
            elif any(w in text_clean for w in ["train model", "start training", "train the model"]):
                intent = "TRAIN_MODEL"
                
            elif any(w in text_clean for w in ["start recognition", "begin recognition", "gesture recognition"]):
                intent = "START_RECOGNITION"
            elif any(w in text_clean for w in ["stop recognition", "end recognition"]):
                intent = "STOP_RECOGNITION"
            elif "reload model" in text_clean:
                intent = "RELOAD_MODEL"
                
            elif any(w in text_clean for w in ["exit application", "close app", "quit app", "exit app"]):
                intent = "EXIT"
                
            # --- Confirmation / Answers ---
            elif text_clean in ["yes", "confirm", "yeah", "yep"]:
                intent = "YES"
            elif text_clean in ["no", "cancel", "nope"]:
                intent = "NO"
        
        if intent:
            self.last_intent = intent
            self.command_callback(intent, payload)
        return intent, payload


