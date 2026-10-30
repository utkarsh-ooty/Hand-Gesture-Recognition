"""
Speech engine abstractions for Hand Gesture AI.
Provides low-latency hardware speech synthesis with Windows SAPI and pyttsx3 fallback.
"""

import platform
import time
from typing import Optional


class BaseSpeechEngine:
    """Interface for text-to-speech engines."""

    def speak(self, text: str) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def wait_until_done(self, timeout_ms: int = 40) -> bool:
        """Returns True if finished speaking, False if still speaking."""
        return True

    def close(self) -> None:
        pass


class SapiSpeechEngine(BaseSpeechEngine):
    """Windows SAPI speech engine using win32com."""

    def __init__(self, rate: int = 2):
        self._rate = rate
        self._voice = None
        self._init_engine()

    def _init_engine(self):
        if platform.system() != "Windows":
            raise RuntimeError("SAPI is only available on Windows.")
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        self._voice = win32com.client.Dispatch("SAPI.SpVoice")
        self._voice.Rate = self._rate

    def speak(self, text: str) -> None:
        if self._voice:
            # 1 = SVSFlagsAsync
            self._voice.Speak(text, 1)

    def stop(self) -> None:
        if self._voice:
            # 2 = SVSFPurgeBeforeSpeak (immediately purges hardware buffer)
            try:
                self._voice.Speak("", 2)
            except Exception:
                pass

    def wait_until_done(self, timeout_ms: int = 40) -> bool:
        if self._voice:
            return bool(self._voice.WaitUntilDone(timeout_ms))
        return True

    def close(self) -> None:
        self.stop()
        self._voice = None


class Pyttsx3SpeechEngine(BaseSpeechEngine):
    """Cross-platform fallback speech engine using pyttsx3."""

    def __init__(self, rate: int = 200, volume: float = 1.0):
        import pyttsx3
        self._engine = pyttsx3.init()
        self._engine.setProperty("rate", rate)
        self._engine.setProperty("volume", volume)

    def speak(self, text: str) -> None:
        if self._engine:
            self._engine.say(text)
            self._engine.runAndWait()

    def stop(self) -> None:
        if self._engine:
            try:
                self._engine.stop()
            except Exception:
                pass

    def close(self) -> None:
        self.stop()
        self._engine = None


class SilentSpeechEngine(BaseSpeechEngine):
    """Fallback dummy engine for environments with no audio devices."""

    def __init__(self):
        self._active = False

    def speak(self, text: str) -> None:
        self._active = True
        time.sleep(0.05)
        self._active = False

    def stop(self) -> None:
        self._active = False


def create_speech_engine(sapi_rate: int = 2, pyttsx3_rate: int = 200) -> BaseSpeechEngine:
    """Factory creating the best available speech engine."""
    if platform.system() == "Windows":
        try:
            return SapiSpeechEngine(rate=sapi_rate)
        except Exception as e:
            print(f"[SpeechEngine] SAPI init failed ({e}), falling back to pyttsx3.")

    try:
        return Pyttsx3SpeechEngine(rate=pyttsx3_rate)
    except Exception as e:
        print(f"[SpeechEngine] pyttsx3 init failed ({e}), using silent fallback.")
        return SilentSpeechEngine()
