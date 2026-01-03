"""
Transcription frames for Knomly pipeline.

These frames contain the result of speech-to-text processing.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from .base import Frame


@dataclass(frozen=True)
class TranscriptionFrame(Frame):
    """
    Frame containing transcription results from STT processing.

    Attributes:
        original_text: Transcribed text in original language
        english_text: Translated text in English (same as original if already English)
        detected_language: ISO 639-1 language code (e.g., "hi", "en", "es")
        language_name: Human-readable language name (e.g., "Hindi", "English")
        confidence: Transcription confidence score (0.0 to 1.0)
        words: Optional list of word-level timestamps
        sender_phone: Phone number of the sender (passed through pipeline)
    """

    original_text: str = ""
    english_text: str = ""
    detected_language: str = "en"
    language_name: str = "English"
    confidence: float = 0.0
    words: Optional[List[Dict[str, Any]]] = None
    sender_phone: Optional[str] = None

    _id: uuid.UUID = field(default_factory=uuid.uuid4, repr=False)
    _created_at: datetime = field(default_factory=datetime.utcnow, repr=False)
    _metadata: Dict[str, Any] = field(default_factory=dict, repr=False)
    _source_frame_id: Optional[uuid.UUID] = field(default=None, repr=False)

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def metadata(self) -> Dict[str, Any]:
        return self._metadata.copy()

    @property
    def source_frame_id(self) -> Optional[uuid.UUID]:
        return self._source_frame_id

    @property
    def frame_type(self) -> str:
        return "TranscriptionFrame"

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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": str(self._id),
            "frame_type": self.frame_type,
            "created_at": self._created_at.isoformat(),
            "source_frame_id": str(self._source_frame_id) if self._source_frame_id else None,
            "metadata": self._metadata,
            "original_text": self.original_text[:200] + "..." if len(self.original_text) > 200 else self.original_text,
            "english_text": self.english_text[:200] + "..." if len(self.english_text) > 200 else self.english_text,
            "detected_language": self.detected_language,
            "language_name": self.language_name,
            "confidence": self.confidence,
            "is_translated": self.is_translated,
            "sender_phone": self.sender_phone,
        }

    @classmethod
    def from_stt_result(
        cls,
        result: Dict[str, Any],
        source_frame_id: Optional[uuid.UUID] = None,
        sender_phone: Optional[str] = None,
    ) -> "TranscriptionFrame":
        """
        Factory method to create TranscriptionFrame from STT provider result.

        Expected result format:
        {
            "original_text": "...",
            "english_text": "...",
            "detected_language": "hi",
            "language_name": "Hindi",
            "confidence": 0.95
        }
        """
        return cls(
            original_text=result.get("original_text", ""),
            english_text=result.get("english_text", result.get("original_text", "")),
            detected_language=result.get("detected_language", "en"),
            language_name=result.get("language_name", "English"),
            confidence=result.get("confidence", 0.0),
            words=result.get("words"),
            sender_phone=sender_phone,
            _source_frame_id=source_frame_id,
        )
