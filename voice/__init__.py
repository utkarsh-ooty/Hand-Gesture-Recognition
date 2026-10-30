"""
Voice output package for Hand Gesture AI.
Provides accessible, interruptible text-to-speech output off the UI thread.
"""

from voice.speech_manager import SpeechManager
from voice.speech_queue import SpeechQueue
from voice.interruption import SpeechInterruptionHandler
from voice.speech_engine import BaseSpeechEngine, create_speech_engine

__all__ = [
    "SpeechManager",
    "SpeechQueue",
    "SpeechInterruptionHandler",
    "BaseSpeechEngine",
    "create_speech_engine",
]
