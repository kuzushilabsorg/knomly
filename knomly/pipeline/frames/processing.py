"""
Processing frames for Knomly pipeline.

These frames represent intermediate processing results.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .base import Frame


@dataclass(frozen=True, kw_only=True, slots=True)
class TranscriptionFrame(Frame):
    """
    Frame containing transcription results from STT processing.

    Attributes:
        original_text: Transcribed text in original language
        english_text: Translated text in English (same as original if already English)
        detected_language: ISO 639-1 language code (e.g., "hi", "en", "es")
        language_name: Human-readable language name (e.g., "Hindi", "English")
        confidence: Transcription confidence score (0.0 to 1.0)
        sender_phone: Phone number of the sender (passed through pipeline)
        provider: Name of the STT provider used
    """

    original_text: str = ""
    english_text: str = ""
    detected_language: str = "en"
    language_name: str = "English"
    confidence: float = 0.0
    sender_phone: str = ""
    provider: str = ""

    @property
    def text(self) -> str:
        """Convenience property returning English text for processing."""
        return self.english_text or self.original_text

    @property
    def is_translated(self) -> bool:
        """Check if the text was translated from another language."""
        return self.detected_language.lower() != "en"

    @property
    def is_high_confidence(self) -> bool:
        """Check if transcription meets quality threshold."""
        return self.confidence >= 0.7

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update({
            "original_text": self.original_text[:200] + "..." if len(self.original_text) > 200 else self.original_text,
            "english_text": self.english_text[:200] + "..." if len(self.english_text) > 200 else self.english_text,
            "detected_language": self.detected_language,
            "language_name": self.language_name,
            "confidence": self.confidence,
            "is_translated": self.is_translated,
            "sender_phone": self.sender_phone,
            "provider": self.provider,
        })
        return base


@dataclass(frozen=True, kw_only=True, slots=True)
class ExtractionFrame(Frame):
    """
    Frame containing extracted standup information from transcription.

    Attributes:
        today_items: List of tasks/items for today
        yesterday_items: List of tasks completed yesterday
        blockers: List of blockers or issues
        summary: Brief summary of the standup
        sender_phone: Phone number of the sender
        user_name: Display name of the user
        zulip_stream: Target Zulip stream (e.g., "standup")
        zulip_topic: Target Zulip topic (e.g., "arunank-updates")
    """

    today_items: tuple[str, ...] = ()
    yesterday_items: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    summary: str = ""
    sender_phone: str = ""
    user_name: str = ""
    zulip_stream: str = "standup"
    zulip_topic: str = ""

    @property
    def has_blockers(self) -> bool:
        """Check if any blockers were reported."""
        return len(self.blockers) > 0

    @property
    def has_items(self) -> bool:
        """Check if any work items were reported."""
        return len(self.today_items) > 0 or len(self.yesterday_items) > 0

    def format_zulip_message(self) -> str:
        """
        Format the standup for Zulip posting.

        Returns a markdown-formatted message suitable for Zulip.
        """
        from datetime import datetime, timezone

        date_str = datetime.now(timezone.utc).strftime("%B %d, %Y")
        lines = [f"**Morning Standup - {date_str}**", ""]

        if self.yesterday_items:
            lines.append("**Yesterday:**")
            for item in self.yesterday_items:
                lines.append(f"- [x] {item}")
            lines.append("")

        if self.today_items:
            lines.append("**Today's Focus:**")
            for item in self.today_items:
                lines.append(f"- [ ] {item}")
            lines.append("")

        if self.blockers:
            lines.append("**Blockers:**")
            for blocker in self.blockers:
                lines.append(f"- :warning: {blocker}")
        else:
            lines.append("**Blockers:** None")

        if self.summary:
            lines.append("")
            lines.append(f"*{self.summary}*")

        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update({
            "today_items": list(self.today_items),
            "yesterday_items": list(self.yesterday_items),
            "blockers": list(self.blockers),
            "summary": self.summary,
            "user_name": self.user_name,
            "zulip_stream": self.zulip_stream,
            "zulip_topic": self.zulip_topic,
            "sender_phone": self.sender_phone,
            "has_blockers": self.has_blockers,
            "has_items": self.has_items,
        })
        return base
