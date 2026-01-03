"""
STT (Speech-to-Text) Providers for Knomly.

Provides multiple speech-to-text implementations:
- GeminiSTTProvider: Google Gemini with transcription + translation
- DeepgramSTTProvider: Deepgram nova-2 with high accuracy
- WhisperSTTProvider: OpenAI Whisper with translation support
"""

from .base import BaseSTTProvider, STTProvider, TranscriptionResult
from .deepgram import DeepgramStreamingSTTProvider, DeepgramSTTProvider
from .gemini import GeminiSTTProvider
from .whisper import WhisperSTTProvider

__all__ = [
    "BaseSTTProvider",
    "DeepgramSTTProvider",
    "DeepgramStreamingSTTProvider",
    # Implementations
    "GeminiSTTProvider",
    # Protocol and base
    "STTProvider",
    "TranscriptionResult",
    "WhisperSTTProvider",
]
