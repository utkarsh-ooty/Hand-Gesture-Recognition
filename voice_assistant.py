import threading
import queue
import time
import os
import platform
import asyncio

class VoiceAssistant:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _play_mp3(self, file_path):
        import ctypes
        # Native Windows Media Foundation play to skip external python dependencies 
        ctypes.windll.winmm.mciSendStringW(f'open "{file_path}" type mpegvideo alias myaudio', None, 0, None)
        ctypes.windll.winmm.mciSendStringW('play myaudio wait', None, 0, None)
        ctypes.windll.winmm.mciSendStringW('close myaudio', None, 0, None)

    def _speech_worker(self):
        # We handle async execution here cleanly inside the dedicated thread
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
                    
            try:
                import edge_tts
                temp_file = os.path.abspath("temp_voice.mp3")
                communicate = edge_tts.Communicate(text, "en-IN-NeerjaNeural")
                loop.run_until_complete(communicate.save(temp_file))
                
                # Verify rendering worked
                if os.path.exists(temp_file):
                    self._play_mp3(temp_file)
            except Exception as eval_e:
                print(f"Edge-TTS saying error: {eval_e}. Falling back to default.")
                self._fallback_sapi(text)

    def _fallback_sapi(self, text):
        try:
            if platform.system() == "Windows":
                import pythoncom
                pythoncom.CoInitialize()
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            speaker.Speak(text)
        except Exception as e:
            print(f"SAPI Fallback failed: {e}")

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
