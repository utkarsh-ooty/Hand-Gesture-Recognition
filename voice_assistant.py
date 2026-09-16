import threading
import queue
import time
import os
import platform
import asyncio
import uuid
import tempfile

class VoiceAssistant:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self._is_speaking = False
        self._lock = threading.Lock()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    @property
    def is_speaking(self) -> bool:
        with self._lock:
            return self._is_speaking

    def _set_speaking(self, val: bool):
        with self._lock:
            self._is_speaking = val

    def _play_mp3(self, file_path):
        if platform.system() == "Windows":
            import ctypes
            alias = f"myaudio_{uuid.uuid4().hex[:8]}"
            ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" type mpegvideo alias {alias}', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'play {alias} wait', None, 0, None)
            ctypes.windll.winmm.mciSendStringW(f'close {alias}', None, 0, None)

    def _speech_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        while True:
            text = self.speech_queue.get()
            if text is None: # Sentinel
                break
                
            # Drain queue to only play newest latest message if flooded
            while not self.speech_queue.empty():
                try:
                    text = self.speech_queue.get_nowait()
                except queue.Empty:
                    break
                    
            if not text or not text.strip():
                continue

            self._set_speaking(True)
            temp_file = None
            try:
                import edge_tts
                temp_dir = tempfile.gettempdir()
                filename = f"voice_{uuid.uuid4().hex[:8]}.mp3"
                temp_file = os.path.abspath(os.path.join(temp_dir, filename))
                
                communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
                loop.run_until_complete(communicate.save(temp_file))
                
                if os.path.exists(temp_file):
                    self._play_mp3(temp_file)
            except Exception as eval_e:
                print(f"[VoiceAssistant] Edge-TTS error: {eval_e}. Falling back to SAPI/pyttsx3.")
                self._fallback_sapi(text)
            finally:
                self._set_speaking(False)
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass

    def _fallback_sapi(self, text):
        try:
            if platform.system() == "Windows":
                import pythoncom
                pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
            return
        except Exception as e:
            print(f"[VoiceAssistant] SAPI Fallback failed: {e}")

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.say(text)
            engine.runAndWait()
        except Exception as e:
            print(f"[VoiceAssistant] pyttsx3 Fallback failed: {e}")

    def speak(self, text: str):
        """Adds text to the queue without blocking the main Application."""
        self.speech_queue.put(text)

    def stop_speaking(self):
        # Clear the queue to prevent backlogs
        while not self.speech_queue.empty():
            try:
                self.speech_queue.get_nowait()
            except queue.Empty:
                break

