"""
Input frames for Knomly pipeline.

These frames represent the initial data entering the pipeline,
typically from external sources like WhatsApp/Twilio webhooks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Frame


@dataclass(frozen=True, kw_only=True, slots=True)
class AudioInputFrame(Frame):
    """
    Frame containing audio data for processing.

    Can be initialized with either:
    - media_url: URL to download audio from (Twilio media URL)
    - audio_data: Raw audio bytes already downloaded

    Attributes:
        media_url: URL where audio can be downloaded
        audio_data: Raw audio bytes (None if not yet downloaded)
        mime_type: MIME type of audio (e.g., "audio/ogg", "audio/mpeg")
        sender_phone: Phone number of the sender (E.164 format without +)
        profile_name: WhatsApp profile name of sender
        channel_id: Transport channel identifier (e.g., "twilio", "telegram")
    """

    media_url: str | None = None
    audio_data: bytes | None = None
    mime_type: str = "audio/ogg"
    sender_phone: str = ""
    profile_name: str = ""
    channel_id: str = ""

    @property
    def has_audio(self) -> bool:
        """Check if audio bytes are available."""
        return self.audio_data is not None and len(self.audio_data) > 0

    @property
    def needs_download(self) -> bool:
        """Check if audio needs to be downloaded from URL."""
        return not self.has_audio and self.media_url is not None

    def with_audio(self, audio_data: bytes, mime_type: str | None = None) -> AudioInputFrame:
        """Create new frame with downloaded audio bytes."""
        return self.derive(
            audio_data=audio_data,
            mime_type=mime_type or self.mime_type,
        )

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update({
            "media_url": self.media_url[:50] + "..." if self.media_url and len(self.media_url) > 50 else self.media_url,
            "has_audio": self.has_audio,
            "audio_size": len(self.audio_data) if self.audio_data else 0,
            "mime_type": self.mime_type,
            "sender_phone": self.sender_phone,
            "profile_name": self.profile_name,
            "channel_id": self.channel_id,
        })
        return base


@dataclass(frozen=True, kw_only=True, slots=True)
class TextInputFrame(Frame):
    """
    Frame containing text input for processing.

    Used when user sends a text message instead of voice note.

    Attributes:
        text: The raw text content
        sender_phone: Phone number of the sender
        profile_name: WhatsApp profile name of sender
        channel_id: Transport channel identifier (e.g., "twilio", "telegram")
    """

    text: str = ""
    sender_phone: str = ""
    profile_name: str = ""
    channel_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update({
            "text": self.text[:100] + "..." if len(self.text) > 100 else self.text,
            "text_length": len(self.text),
            "sender_phone": self.sender_phone,
        })
        return base
