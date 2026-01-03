"""
Gemini STT Provider for Knomly.

Uses Google's Gemini 2.0 Flash for audio transcription and translation.
"""

from __future__ import annotations

import io
import json
import logging

from .base import BaseSTTProvider, TranscriptionResult

logger = logging.getLogger(__name__)

# System prompt for transcription and translation
TRANSCRIPTION_PROMPT = """
Your task is to process audio data. First, transcribe the audio to its original language. Then, translate the transcribed text to English. Finally, identify the detected language.

Return a JSON object with the following structure:
{
    "original_text": "<The transcribed text in its original language>",
    "english_text": "<The English translation of the transcribed text>",
    "detected_language": "<The ISO 639-1 code of the detected language>",
    "language_name": "<The full name of the detected language>",
    "confidence": <A float number between 0.0 and 1.0 representing the confidence of the language detection>
}

If you cannot determine the language, use "unknown" for the language code and name, and 0.0 for confidence.
"""


class GeminiSTTProvider(BaseSTTProvider):
    """
    Gemini-based Speech-to-Text provider.

    Uses Gemini 2.0 Flash with audio modality for:
    - Transcription in original language
    - Translation to English
    - Language detection

    Requirements:
    - google-generativeai package
    - GEMINI_API_KEY environment variable
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
    ):
        """
        Initialize Gemini STT provider.

        Args:
            api_key: Google AI API key
            model: Model to use (default: gemini-2.0-flash-exp)
        """
        self._api_key = api_key
        self._model_name = model
        self._model = None  # Lazy initialization

    @property
    def name(self) -> str:
        return "gemini"

    def _get_model(self):
        """Lazy initialization of Gemini model."""
        if self._model is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._model = genai.GenerativeModel(
                    self._model_name,
                    system_instruction=[TRANSCRIPTION_PROMPT],
                )
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini STT. "
                    "Install with: pip install google-generativeai"
                )
        return self._model

    def _convert_audio_if_needed(
        self,
        audio_bytes: bytes,
        mime_type: str,
    ) -> tuple[bytes, str]:
        """
        Convert audio to MP3 if not in a supported format.

        Gemini supports: FLAC, MP3, MPEG, OGG, WAV, AMR, WEBM
        We convert to MP3 for broad compatibility.
        """
        supported_formats = ["mpeg", "mp3", "wav", "flac", "ogg", "webm"]
        if any(f in mime_type.lower() for f in supported_formats):
            return audio_bytes, mime_type

        try:
            from pydub import AudioSegment

            logger.info(f"Converting audio from {mime_type} to MP3")
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            buffer = io.BytesIO()
            audio_segment.export(buffer, format="mp3")
            return buffer.getvalue(), "audio/mpeg"
        except ImportError:
            raise ImportError(
                "pydub package is required for audio conversion. Install with: pip install pydub"
            )
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            # Try with original format anyway
            return audio_bytes, mime_type

    def _parse_response(self, response_text: str) -> dict:
        """Parse JSON from Gemini response."""
        # Clean markdown code blocks
        cleaned = response_text.strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0]
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0]
        return json.loads(cleaned.strip())

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio using Gemini.

        Args:
            audio_bytes: Raw audio data
            mime_type: MIME type of audio
            language_hint: Optional language hint (not used by Gemini)

        Returns:
            TranscriptionResult with transcription and translation
        """
        try:
            # Convert audio if needed
            audio_bytes, mime_type = self._convert_audio_if_needed(audio_bytes, mime_type)

            # Get or initialize model
            model = self._get_model()

            # Generate transcription
            response = model.generate_content([{"mime_type": mime_type, "data": audio_bytes}])

            # Parse response
            result = self._parse_response(response.text)

            return TranscriptionResult(
                original_text=result.get("original_text", ""),
                english_text=result.get("english_text", result.get("original_text", "")),
                detected_language=result.get("detected_language", "en"),
                language_name=result.get("language_name", "English"),
                confidence=float(result.get("confidence", 0.0)),
                provider=self.name,
            )

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response: {e}")
            return TranscriptionResult(
                original_text="",
                english_text="",
                detected_language="unknown",
                language_name="Unknown",
                confidence=0.0,
                provider=self.name,
            )
        except Exception as e:
            logger.error(f"Gemini transcription error: {e}", exc_info=True)
            raise
