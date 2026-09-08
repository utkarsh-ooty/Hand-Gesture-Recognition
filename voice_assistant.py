import threading
import queue
import time
import platform

class VoiceAssistant:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        try:
            if platform.system() == "Windows":
                import pythoncom
                pythoncom.CoInitialize()
        except ImportError:
            pass
            
        try:
            import win32com.client
            speaker = win32com.client.Dispatch("SAPI.SpVoice")
            
            while True:
                text = self.speech_queue.get()
                if text is None: # Sentinel
                    break
                try:
                    speaker.Speak(text)
                except Exception as eval_e:
                    print(f"TTS saying error: {eval_e}")
                    
        except ImportError:
            print("win32com is not available. Please install pywin32 to use TTS on Windows.")
        except Exception as e:
            print(f"TTS Engine Init Error: {e}")

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
