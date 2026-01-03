"""
Transcription Processor for Knomly.

Converts audio to text using STT providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..processor import Processor

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import Frame

logger = logging.getLogger(__name__)


class TranscriptionProcessor(Processor):
    """
    Transcribes audio using configured STT provider.

    Input: AudioInputFrame with audio_data
    Output: TranscriptionFrame with transcription and translation

    Uses STT provider from ctx.providers.
    """

    def __init__(self, provider_name: str | None = None):
        """
        Args:
            provider_name: Specific STT provider to use (or default)
        """
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return "transcription"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | None:
        from ..frames import AudioInputFrame, TranscriptionFrame

        if not isinstance(frame, AudioInputFrame):
            return frame

        if not frame.has_audio:
            raise ValueError("AudioInputFrame has no audio_data")

        if ctx.providers is None:
            raise RuntimeError("No providers configured in context")

        stt = ctx.providers.get_stt(self._provider_name)

        logger.info(f"Transcribing {len(frame.audio_data)} bytes with {stt.name}")

        result = await stt.transcribe(
            audio_data=frame.audio_data,
            mime_type=frame.mime_type,
        )

        logger.info(
            f"Transcription: {len(result.english_text)} chars, "
            f"lang={result.detected_language}, conf={result.confidence:.2f}"
        )

        return TranscriptionFrame(
            original_text=result.original_text,
            english_text=result.english_text,
            detected_language=result.detected_language,
            language_name=result.language_name,
            confidence=result.confidence,
            sender_phone=frame.sender_phone,
            provider=result.provider,
            source_frame_id=frame.id,
        )
