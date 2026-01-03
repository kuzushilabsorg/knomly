"""
Whisper STT Provider for Knomly.

Uses OpenAI's Whisper API for speech transcription and translation.
"""

from __future__ import annotations

import io
import logging
from typing import Any

from .base import BaseSTTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

# Whisper supported languages (ISO 639-1 codes)
WHISPER_LANGUAGES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh",
}


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    return WHISPER_LANGUAGES.get(code, code.title())


class WhisperSTTProvider(BaseSTTProvider):
    """
    OpenAI Whisper Speech-to-Text provider.

    Uses OpenAI's Whisper API for:
    - High-accuracy transcription (whisper-1 model)
    - Automatic language detection
    - Built-in translation to English
    - Support for multiple audio formats

    Requirements:
    - openai package (v1+)
    - OPENAI_API_KEY environment variable

    Features:
    - 50+ language support
    - Automatic language detection
    - Translation to English built-in
    - Word-level timestamps (verbose_json)
    - Multiple response formats
    """

    def __init__(
        self,
        api_key: str,
        model: str = "whisper-1",
        response_format: str = "verbose_json",
        temperature: float = 0.0,
    ):
        """
        Initialize Whisper STT provider.

        Args:
            api_key: OpenAI API key
            model: Model to use (whisper-1)
            response_format: Response format (json, text, verbose_json, srt, vtt)
            temperature: Sampling temperature (0.0 for deterministic)
        """
        self._api_key = api_key
        self._model = model
        self._response_format = response_format
        self._temperature = temperature
        self._client = None  # Lazy initialization

    @property
    def name(self) -> str:
        return "whisper"

    def _get_client(self):
        """Lazy initialization of OpenAI async client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "openai package is required for Whisper STT. "
                    "Install with: pip install openai"
                )
        return self._client

    def _get_file_extension(self, mime_type: str) -> str:
        """Convert MIME type to file extension."""
        mime_to_ext = {
            "audio/ogg": "ogg",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/wav": "wav",
            "audio/webm": "webm",
            "audio/flac": "flac",
            "audio/mp4": "mp4",
            "audio/m4a": "m4a",
            "audio/x-m4a": "m4a",
        }
        return mime_to_ext.get(mime_type.lower(), "mp3")

    def _extract_words(self, response: Any) -> list[dict[str, Any]] | None:
        """Extract word-level data from verbose_json response."""
        try:
            if hasattr(response, "words") and response.words:
                return [
                    {
                        "word": w.word,
                        "start": w.start,
                        "end": w.end,
                    }
                    for w in response.words
                ]
            # Try segments if words not available
            if hasattr(response, "segments") and response.segments:
                words = []
                for segment in response.segments:
                    if hasattr(segment, "words") and segment.words:
                        for w in segment.words:
                            words.append(
                                {
                                    "word": w.word,
                                    "start": w.start,
                                    "end": w.end,
                                }
                            )
                return words if words else None
        except (AttributeError, TypeError):
            pass
        return None

    def _extract_duration(self, response: Any) -> int | None:
        """Extract audio duration from response."""
        try:
            if hasattr(response, "duration"):
                return int(response.duration * 1000)  # Convert to ms
        except (AttributeError, TypeError):
            pass
        return None

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Whisper.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio
            language_hint: Optional language hint (ISO 639-1 code)

        Returns:
            TranscriptionResult with transcription
        """
        try:
            client = self._get_client()

            # Prepare audio file
            ext = self._get_file_extension(mime_type)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{ext}"

            # Prepare transcription parameters
            params = {
                "model": self._model,
                "file": audio_file,
                "response_format": self._response_format,
                "temperature": self._temperature,
            }

            if language_hint:
                params["language"] = language_hint

            # Perform transcription
            logger.debug(f"Sending {len(audio_bytes)} bytes to Whisper ({mime_type})")
            response = await client.audio.transcriptions.create(**params)

            # Extract text based on response format
            if self._response_format == "verbose_json":
                text = response.text
                detected_language = getattr(response, "language", "en")
                words = self._extract_words(response)
                duration_ms = self._extract_duration(response)
            elif self._response_format == "json":
                text = response.text
                detected_language = "en"  # Not available in json format
                words = None
                duration_ms = None
            else:
                # text, srt, vtt formats return string
                text = response if isinstance(response, str) else str(response)
                detected_language = "en"
                words = None
                duration_ms = None

            logger.debug(
                f"Whisper transcription: {len(text)} chars, " f"language={detected_language}"
            )

            return TranscriptionResult(
                original_text=text,
                english_text=text,  # Transcription is in original language
                detected_language=detected_language,
                language_name=get_language_name(detected_language),
                confidence=0.95,  # Whisper doesn't provide confidence scores
                words=words,
                duration_ms=duration_ms,
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"Whisper transcription error: {e}", exc_info=True)
            raise

    async def transcribe_with_translation(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        target_language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe audio and translate to English.

        Whisper has a built-in translation endpoint that transcribes
        any language audio directly to English text.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio
            target_language: Target language (only "en" supported)

        Returns:
            TranscriptionResult with English translation
        """
        if target_language != "en":
            logger.warning(
                f"Whisper only supports translation to English. "
                f"Ignoring target_language={target_language}"
            )

        try:
            client = self._get_client()

            # Prepare audio file
            ext = self._get_file_extension(mime_type)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{ext}"

            # Prepare translation parameters
            params = {
                "model": self._model,
                "file": audio_file,
                "response_format": self._response_format,
                "temperature": self._temperature,
            }

            # Use translation endpoint
            logger.debug(f"Sending {len(audio_bytes)} bytes to Whisper translation")
            response = await client.audio.translations.create(**params)

            # Extract text
            if self._response_format == "verbose_json":
                english_text = response.text
                words = self._extract_words(response)
                duration_ms = self._extract_duration(response)
            elif self._response_format == "json":
                english_text = response.text
                words = None
                duration_ms = None
            else:
                english_text = response if isinstance(response, str) else str(response)
                words = None
                duration_ms = None

            # First get transcription in original language
            original_result = await self.transcribe(audio_bytes, mime_type)

            return TranscriptionResult(
                original_text=original_result.original_text,
                english_text=english_text,
                detected_language=original_result.detected_language,
                language_name=original_result.language_name,
                confidence=0.95,
                words=words,
                duration_ms=duration_ms,
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"Whisper translation error: {e}", exc_info=True)
            raise

    async def translate_only(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
    ) -> str:
        """
        Translate audio directly to English text.

        This is more efficient than transcribe_with_translation
        when you only need the English translation.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio

        Returns:
            English text translation
        """
        try:
            client = self._get_client()

            ext = self._get_file_extension(mime_type)
            audio_file = io.BytesIO(audio_bytes)
            audio_file.name = f"audio.{ext}"

            response = await client.audio.translations.create(
                model=self._model,
                file=audio_file,
                response_format="text",
                temperature=self._temperature,
            )

            return response if isinstance(response, str) else response.text

        except Exception as e:
            logger.error(f"Whisper translation error: {e}", exc_info=True)
            raise


__all__ = [
    "WHISPER_LANGUAGES",
    "WhisperSTTProvider",
    "get_language_name",
]
