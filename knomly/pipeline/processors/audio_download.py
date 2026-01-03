"""
Audio Download Processor for Knomly.

Downloads audio from Twilio media URLs.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import httpx

from ..processor import Processor

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import AudioInputFrame, Frame

logger = logging.getLogger(__name__)


class AudioDownloadProcessor(Processor):
    """
    Downloads audio from Twilio media URLs.

    Input: AudioInputFrame with media_url
    Output: AudioInputFrame with audio_data populated

    If frame already has audio_data, passes through unchanged.
    """

    def __init__(
        self,
        twilio_account_sid: str,
        twilio_auth_token: str,
        timeout_seconds: float = 30.0,
    ):
        self._account_sid = twilio_account_sid
        self._auth_token = twilio_auth_token
        self._timeout = timeout_seconds

    @property
    def name(self) -> str:
        return "audio_download"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | None":
        # Import here to avoid circular imports
        from ..frames import AudioInputFrame

        if not isinstance(frame, AudioInputFrame):
            return frame

        # Already has audio data - pass through
        if frame.has_audio:
            logger.debug("Frame already has audio data")
            return frame

        # No URL to download from
        if not frame.media_url:
            raise ValueError("AudioInputFrame has no media_url")

        logger.info(f"Downloading audio from Twilio: {frame.media_url[:60]}...")

        async with httpx.AsyncClient(
            auth=(self._account_sid, self._auth_token),
            timeout=self._timeout,
        ) as client:
            response = await client.get(frame.media_url)
            response.raise_for_status()

            audio_data = response.content
            content_type = response.headers.get("content-type", frame.mime_type)

        logger.info(f"Downloaded {len(audio_data)} bytes, type={content_type}")

        return frame.with_audio(audio_data=audio_data, mime_type=content_type)
