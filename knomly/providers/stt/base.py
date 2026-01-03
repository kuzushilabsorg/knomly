"""
STT (Speech-to-Text) Provider Protocol for Knomly.

Defines the interface for speech transcription providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


@dataclass
class TranscriptionResult:
    """
    Result from STT transcription.

    Attributes:
        original_text: Transcribed text in detected language
        english_text: Translated text in English (same if already English)
        detected_language: ISO 639-1 language code (e.g., "hi", "en")
        language_name: Human-readable language name (e.g., "Hindi")
        confidence: Transcription confidence (0.0 to 1.0)
        words: Optional word-level timestamps
        duration_ms: Audio duration in milliseconds
        provider: Name of the provider used
    """

    original_text: str
    english_text: str
    detected_language: str = "en"
    language_name: str = "English"
    confidence: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    duration_ms: Optional[int] = None
    provider: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_text": self.original_text,
            "english_text": self.english_text,
            "detected_language": self.detected_language,
            "language_name": self.language_name,
            "confidence": self.confidence,
            "duration_ms": self.duration_ms,
            "provider": self.provider,
        }


@runtime_checkable
class STTProvider(Protocol):
    """
    Protocol for Speech-to-Text providers.

    Implementations must provide:
    - transcribe(): Convert audio to text
    - name: Provider identifier

    Supported providers:
    - Gemini (gemini-2.0-flash-exp with audio support)
    - Deepgram (nova-2 model)
    - OpenAI Whisper (whisper-1)
    """

    @property
    def name(self) -> str:
        """Provider name for logging and configuration."""
        ...

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio to text.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio (audio/ogg, audio/mpeg, etc.)
            language_hint: Optional language hint for transcription

        Returns:
            TranscriptionResult with transcribed and translated text
        """
        ...


class BaseSTTProvider(ABC):
    """
    Base class for STT provider implementations.

    Provides common functionality and enforces interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: Optional[str] = None,
    ) -> TranscriptionResult:
        """Transcribe audio to text."""
        pass

    async def transcribe_with_translation(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        target_language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe audio and translate to target language.

        Default implementation calls transcribe() which handles translation.
        Override if provider has separate translation capability.
        """
        return await self.transcribe(audio_bytes, mime_type)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
