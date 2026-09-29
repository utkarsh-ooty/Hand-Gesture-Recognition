import threading
import queue
import time
import os
import platform
import asyncio
import uuid
import tempfile

# Threshold: messages shorter than this use fast local SAPI; longer ones use Edge-TTS neural voice
_FAST_CHAR_LIMIT = 80

class VoiceAssistant:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self._is_speaking = False
        self._lock = threading.Lock()
        
        # Pre-init pyttsx3 engine once on the worker thread (avoids re-init cost)
        self._pyttsx3_engine = None
        
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def _set_speaking(self, val: bool):
        with self._lock:
            self._is_speaking = val

    def _init_pyttsx3(self):
        """Lazily initialize pyttsx3 on the worker thread (COM-safe)."""
        if self._pyttsx3_engine is None:
            try:
                if platform.system() == "Windows":
                    import pythoncom
                    pythoncom.CoInitialize()
                import pyttsx3
                self._pyttsx3_engine = pyttsx3.init()
                self._pyttsx3_engine.setProperty('rate', 210)  # Snappy pace
                self._pyttsx3_engine.setProperty('volume', 1.0)
            except Exception as e:
                print(f"[VoiceAssistant] pyttsx3 init failed: {e}")
                self._pyttsx3_engine = None

    def _speak_fast(self, text):
        """Use local SAPI/pyttsx3 for instant low-latency speech."""
        self._init_pyttsx3()
        if self._pyttsx3_engine:
            try:
                self._pyttsx3_engine.say(text)
                self._pyttsx3_engine.runAndWait()
                return True
            except Exception as e:
                print(f"[VoiceAssistant] Fast speech error: {e}")
        # Fallback to win32com SAPI
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Rate = 3  # Faster rate
            speaker.Speak(text)
            return True
        except Exception:
            pass
        return False

    def _play_mp3(self, file_path):
        if platform.system() == "Windows":
            import ctypes
            alias = f"myaudio_{uuid.uuid4().hex[:8]}"
            ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" type mpegvideo alias {alias}', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)

    def _speak_neural(self, text):
        """Use Edge-TTS neural voice for longer, richer narration."""
        try:
            import edge_tts
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            temp_dir = tempfile.gettempdir()
            filename = f"voice_{uuid.uuid4().hex[:8]}.mp3"
            temp_file = os.path.abspath(os.path.join(temp_dir, filename))
            
            communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural", rate="+15%")
            loop.run_until_complete(communicate.save(temp_file))
            
            if os.path.exists(temp_file):
                self._play_mp3(temp_file)
                
            # Cleanup
            try:
                os.remove(temp_file)
            except Exception:
                pass
            return True
        except Exception as e:
            print(f"[VoiceAssistant] Edge-TTS error: {e}")
            return False

    def _speech_worker(self):
        while True:
            text = self.speech_queue.get()
            if text is None:
                break
                
            # Drain queue — only speak the latest message if flooded
            while not self.speech_queue.empty():
                try:
                    text = self.speech_queue.get_nowait()
                except queue.Empty:
                    break
                    
            if not text or not text.strip():
                continue

            self._set_speaking(True)
            try:
                if len(text) <= _FAST_CHAR_LIMIT:
                    # Short confirmations: use instant local engine
                    if not self._speak_fast(text):
                        self._speak_neural(text)
                else:
                    # Long narrations: use neural voice, fallback to fast
                    if not self._speak_neural(text):
                        self._speak_fast(text)
            finally:
                self._set_speaking(False)

    def speak(self, text: str):
        """Adds text to the queue without blocking the main application."""
        self.speech_queue.put(text)

    def stop_speaking(self):
        """Clear the queue to prevent backlogs."""
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break
