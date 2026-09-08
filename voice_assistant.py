import pyttsx3
import threading
import queue

class VoiceAssistant:
    def __init__(self):
        self.speech_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._speech_worker, daemon=True)
        self.worker_thread.start()

    def _speech_worker(self):
        # We initialize pyttsx3 ONCE inside the dedicated thread to avoid 
        # COM thread apartment issues on Windows and SAPI5 deadlocks.
        try:
            import pythoncom
            pythoncom.CoInitialize()
        except ImportError:
            pass
            
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.setProperty('volume', 1.0)
            
            while True:
                text = self.speech_queue.get()
                if text is None: # Sentinel
                    break
                try:
                    engine.say(text)
                    engine.runAndWait()
                except Exception as eval_e:
                    print(f"TTS saying error: {eval_e}")
        except Exception as e:
            print(f"TTS Engine Init Error: {e}")

    def speak(self, text: str):
        """Adds text to the queue without blocking the main Application."""
        self.speech_queue.put(text)

    def stop_speaking(self):
        # Difficult to stop an in-progress SAPI5 speech cleanly across threads.
        pass
