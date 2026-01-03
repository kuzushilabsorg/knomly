"""
STT (Speech-to-Text) Providers for Knomly.

Provides multiple speech-to-text implementations:
- GeminiSTTProvider: Google Gemini with transcription + translation
- DeepgramSTTProvider: Deepgram nova-2 with high accuracy
- WhisperSTTProvider: OpenAI Whisper with translation support
"""

from .base import BaseSTTProvider, STTProvider, TranscriptionResult
from .gemini import GeminiSTTProvider
from .deepgram import DeepgramSTTProvider, DeepgramStreamingSTTProvider
from .whisper import WhisperSTTProvider

__all__ = [
    # Protocol and base
    "STTProvider",
    "BaseSTTProvider",
    "TranscriptionResult",
    # Implementations
    "GeminiSTTProvider",
    "DeepgramSTTProvider",
    "DeepgramStreamingSTTProvider",
    "WhisperSTTProvider",
]
