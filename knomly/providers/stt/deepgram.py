"""
Deepgram STT Provider for Knomly.

Uses Deepgram's nova-2 model for accurate speech transcription
with language detection and word-level timestamps.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseSTTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

# Language code to name mapping for common languages
LANGUAGE_NAMES: dict[str, str] = {
    "en": "English",
    "en-US": "English (US)",
    "en-GB": "English (UK)",
    "en-AU": "English (Australia)",
    "en-IN": "English (India)",
    "hi": "Hindi",
    "hi-Latn": "Hindi (Latin script)",
    "es": "Spanish",
    "es-419": "Spanish (Latin America)",
    "fr": "French",
    "fr-CA": "French (Canadian)",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pt-BR": "Portuguese (Brazil)",
    "ja": "Japanese",
    "ko": "Korean",
    "zh": "Chinese",
    "zh-CN": "Chinese (Simplified)",
    "zh-TW": "Chinese (Traditional)",
    "ar": "Arabic",
    "ru": "Russian",
    "nl": "Dutch",
    "tr": "Turkish",
    "pl": "Polish",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "fi": "Finnish",
    "id": "Indonesian",
    "ms": "Malay",
    "th": "Thai",
    "vi": "Vietnamese",
    "uk": "Ukrainian",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
}


def get_language_name(code: str) -> str:
    """Get human-readable language name from code."""
    if code in LANGUAGE_NAMES:
        return LANGUAGE_NAMES[code]
    # Try base language code
    base_code = code.split("-")[0] if "-" in code else code
    return LANGUAGE_NAMES.get(base_code, code.title())


class DeepgramSTTProvider(BaseSTTProvider):
    """
    Deepgram Speech-to-Text provider.

    Uses Deepgram's nova-2 model for:
    - High-accuracy transcription
    - Automatic language detection
    - Word-level timestamps
    - Punctuation and formatting

    Requirements:
    - deepgram-sdk package (v3+)
    - DEEPGRAM_API_KEY environment variable

    Features:
    - Smart formatting (punctuation, capitalization)
    - Multi-language support (30+ languages)
    - Word-level confidence scores
    - Audio duration reporting
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        smart_format: bool = True,
        punctuate: bool = True,
        diarize: bool = False,
        utterances: bool = False,
    ):
        """
        Initialize Deepgram STT provider.

        Args:
            api_key: Deepgram API key
            model: Model to use (nova-2, nova, base, enhanced)
            smart_format: Enable smart formatting
            punctuate: Enable punctuation
            diarize: Enable speaker diarization
            utterances: Enable utterance detection
        """
        self._api_key = api_key
        self._model = model
        self._smart_format = smart_format
        self._punctuate = punctuate
        self._diarize = diarize
        self._utterances = utterances
        self._client = None  # Lazy initialization

    @property
    def name(self) -> str:
        return "deepgram"

    def _get_client(self):
        """Lazy initialization of Deepgram client."""
        if self._client is None:
            try:
                from deepgram import DeepgramClient

                self._client = DeepgramClient(self._api_key)
            except ImportError:
                raise ImportError(
                    "deepgram-sdk package is required for Deepgram STT. "
                    "Install with: pip install deepgram-sdk"
                )
        return self._client

    def _get_mime_type_for_deepgram(self, mime_type: str) -> str:
        """Convert MIME type to Deepgram-compatible format."""
        mime_map = {
            "audio/ogg": "audio/ogg",
            "audio/mpeg": "audio/mpeg",
            "audio/mp3": "audio/mpeg",
            "audio/wav": "audio/wav",
            "audio/webm": "audio/webm",
            "audio/flac": "audio/flac",
            "audio/mp4": "audio/mp4",
            "audio/m4a": "audio/mp4",
            "audio/aac": "audio/aac",
        }
        return mime_map.get(mime_type.lower(), mime_type)

    def _extract_words(self, response: Any) -> list[dict[str, Any]] | None:
        """Extract word-level data from Deepgram response."""
        try:
            words = response.results.channels[0].alternatives[0].words
            return [
                {
                    "word": w.word,
                    "start": w.start,
                    "end": w.end,
                    "confidence": w.confidence,
                }
                for w in words
            ]
        except (AttributeError, IndexError):
            return None

    def _extract_duration(self, response: Any) -> int | None:
        """Extract audio duration from Deepgram response."""
        try:
            duration_seconds = response.metadata.duration
            return int(duration_seconds * 1000)  # Convert to ms
        except AttributeError:
            return None

    def _extract_confidence(self, response: Any) -> float:
        """Extract overall confidence from Deepgram response."""
        try:
            return response.results.channels[0].alternatives[0].confidence
        except (AttributeError, IndexError):
            return 0.0

    def _extract_detected_language(self, response: Any) -> str:
        """Extract detected language from Deepgram response."""
        try:
            return response.results.channels[0].detected_language or "en"
        except (AttributeError, IndexError):
            return "en"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Deepgram.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio
            language_hint: Optional language hint (BCP-47 code)

        Returns:
            TranscriptionResult with transcription and metadata
        """
        try:
            from deepgram import PrerecordedOptions

            client = self._get_client()

            # Configure transcription options
            options = PrerecordedOptions(
                model=self._model,
                smart_format=self._smart_format,
                punctuate=self._punctuate,
                diarize=self._diarize,
                utterances=self._utterances,
                detect_language=True,
            )

            # Add language hint if provided
            if language_hint:
                options.language = language_hint

            # Prepare audio source
            source = {
                "buffer": audio_bytes,
                "mimetype": self._get_mime_type_for_deepgram(mime_type),
            }

            # Perform transcription
            logger.debug(f"Sending {len(audio_bytes)} bytes to Deepgram ({mime_type})")
            response = await client.listen.asyncrest.v("1").transcribe_file(source, options)

            # Extract transcript
            try:
                transcript = response.results.channels[0].alternatives[0].transcript
            except (AttributeError, IndexError):
                logger.warning("No transcript found in Deepgram response")
                transcript = ""

            # Extract metadata
            detected_language = self._extract_detected_language(response)
            confidence = self._extract_confidence(response)
            words = self._extract_words(response)
            duration_ms = self._extract_duration(response)

            logger.debug(
                f"Deepgram transcription: {len(transcript)} chars, "
                f"language={detected_language}, confidence={confidence:.2f}"
            )

            return TranscriptionResult(
                original_text=transcript,
                english_text=transcript,  # Deepgram doesn't translate by default
                detected_language=detected_language,
                language_name=get_language_name(detected_language),
                confidence=confidence,
                words=words,
                duration_ms=duration_ms,
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"Deepgram transcription error: {e}", exc_info=True)
            raise

    async def transcribe_with_translation(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        target_language: str = "en",
    ) -> TranscriptionResult:
        """
        Transcribe audio with translation.

        Note: Deepgram doesn't provide built-in translation.
        This method returns the transcription as-is.
        For translation, use an LLM provider.
        """
        return await self.transcribe(audio_bytes, mime_type)


class DeepgramStreamingSTTProvider(BaseSTTProvider):
    """
    Deepgram Streaming Speech-to-Text provider.

    For real-time transcription of audio streams.
    Uses WebSocket connection for low-latency results.

    Note: This is a specialized provider for streaming use cases.
    For batch transcription, use DeepgramSTTProvider.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "nova-2",
        smart_format: bool = True,
        interim_results: bool = True,
        endpointing: int = 300,
    ):
        """
        Initialize Deepgram Streaming STT provider.

        Args:
            api_key: Deepgram API key
            model: Model to use
            smart_format: Enable smart formatting
            interim_results: Return interim (partial) results
            endpointing: Silence duration (ms) to end utterance
        """
        self._api_key = api_key
        self._model = model
        self._smart_format = smart_format
        self._interim_results = interim_results
        self._endpointing = endpointing
        self._client = None

    @property
    def name(self) -> str:
        return "deepgram-streaming"

    def _get_client(self):
        """Lazy initialization of Deepgram client."""
        if self._client is None:
            try:
                from deepgram import DeepgramClient

                self._client = DeepgramClient(self._api_key)
            except ImportError:
                raise ImportError(
                    "deepgram-sdk package is required for Deepgram STT. "
                    "Install with: pip install deepgram-sdk"
                )
        return self._client

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using streaming API.

        For streaming, this processes the entire audio buffer
        through the streaming connection for consistency.
        """
        # For single-shot transcription, delegate to batch provider
        batch_provider = DeepgramSTTProvider(
            api_key=self._api_key,
            model=self._model,
            smart_format=self._smart_format,
        )
        return await batch_provider.transcribe(audio_bytes, mime_type, language_hint)

    async def create_streaming_connection(self):
        """
        Create a streaming WebSocket connection.

        Returns a connection object for real-time transcription.
        Use this for live audio streams.

        Example:
            async with provider.create_streaming_connection() as conn:
                await conn.send(audio_chunk)
                async for result in conn.results():
                    print(result.transcript)
        """
        try:
            from deepgram import LiveOptions

            client = self._get_client()

            options = LiveOptions(
                model=self._model,
                smart_format=self._smart_format,
                interim_results=self._interim_results,
                endpointing=self._endpointing,
                detect_language=True,
            )

            return await client.listen.asynclive.v("1").transcribe_live(options)

        except Exception as e:
            logger.error(f"Failed to create streaming connection: {e}")
            raise


__all__ = [
    "LANGUAGE_NAMES",
    "DeepgramSTTProvider",
    "DeepgramStreamingSTTProvider",
    "get_language_name",
]
