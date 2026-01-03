"""
Tests for Knomly pipeline frames.

Tests frame immutability, derivation, and serialization.
"""
import pytest
from uuid import UUID
from datetime import datetime

from knomly.pipeline.frames import (
    AudioInputFrame,
    ConfirmationFrame,
    ErrorFrame,
    ExtractionFrame,
    Frame,
    TextInputFrame,
    TranscriptionFrame,
    ZulipMessageFrame,
)


class TestFrameBase:
    """Tests for base Frame class."""

    def test_frame_has_auto_generated_id(self):
        """Frame should have a UUID id auto-generated."""
        frame = AudioInputFrame()
        assert isinstance(frame.id, UUID)

    def test_frame_has_timestamp(self):
        """Frame should have a created_at timestamp."""
        frame = AudioInputFrame()
        assert isinstance(frame.created_at, datetime)

    def test_frame_is_immutable(self):
        """Frame should be frozen (immutable)."""
        frame = AudioInputFrame()
        with pytest.raises(AttributeError):
            frame.sender_phone = "123456"

    def test_frame_derive_creates_new_frame(self):
        """derive() should create a new frame with new ID."""
        original = AudioInputFrame(sender_phone="123456")
        derived = original.derive(sender_phone="789012")

        assert derived.id != original.id
        assert derived.sender_phone == "789012"
        assert derived.source_frame_id == original.id

    def test_frame_to_dict_serializes(self):
        """to_dict() should return serializable dict."""
        frame = AudioInputFrame(sender_phone="123456")
        data = frame.to_dict()

        assert "id" in data
        assert "frame_type" in data
        assert "created_at" in data
        assert data["frame_type"] == "AudioInputFrame"


class TestAudioInputFrame:
    """Tests for AudioInputFrame."""

    def test_audio_input_with_url(self):
        """AudioInputFrame should store media URL."""
        frame = AudioInputFrame(
            media_url="https://example.com/audio.ogg",
            mime_type="audio/ogg",
            sender_phone="919876543210",
        )

        assert frame.media_url == "https://example.com/audio.ogg"
        assert frame.mime_type == "audio/ogg"
        assert frame.has_audio is False
        assert frame.needs_download is True

    def test_audio_input_with_bytes(self):
        """AudioInputFrame should store audio bytes."""
        audio_bytes = b"fake audio data"
        frame = AudioInputFrame(
            audio_data=audio_bytes,
            mime_type="audio/mpeg",
        )

        assert frame.has_audio is True
        assert frame.needs_download is False
        assert len(frame.audio_data) == len(audio_bytes)

    def test_with_audio_creates_new_frame(self):
        """with_audio() should create new frame with audio data."""
        original = AudioInputFrame(
            media_url="https://example.com/audio.ogg",
            sender_phone="123456",
        )

        audio_bytes = b"downloaded audio"
        new_frame = original.with_audio(audio_bytes, "audio/mpeg")

        assert new_frame.has_audio is True
        assert new_frame.mime_type == "audio/mpeg"
        assert new_frame.sender_phone == "123456"  # Preserved
        assert new_frame.source_frame_id == original.id


class TestTranscriptionFrame:
    """Tests for TranscriptionFrame."""

    def test_transcription_text_property(self):
        """text property should return english_text or original_text."""
        # When both present, prefer english
        frame = TranscriptionFrame(
            original_text="मैं आज काम करूंगा",
            english_text="I will work today",
            detected_language="hi",
        )
        assert frame.text == "I will work today"

        # When only original present
        frame2 = TranscriptionFrame(
            original_text="Hello world",
            detected_language="en",
        )
        assert frame2.text == "Hello world"

    def test_transcription_is_translated(self):
        """is_translated should detect non-English source."""
        hindi_frame = TranscriptionFrame(detected_language="hi")
        assert hindi_frame.is_translated is True

        english_frame = TranscriptionFrame(detected_language="en")
        assert english_frame.is_translated is False

    def test_transcription_confidence_check(self):
        """is_high_confidence should check threshold."""
        high_conf = TranscriptionFrame(confidence=0.85)
        assert high_conf.is_high_confidence is True

        low_conf = TranscriptionFrame(confidence=0.5)
        assert low_conf.is_high_confidence is False


class TestExtractionFrame:
    """Tests for ExtractionFrame."""

    def test_extraction_has_items(self):
        """has_items should check for any work items."""
        empty = ExtractionFrame()
        assert empty.has_items is False

        with_today = ExtractionFrame(today_items=("Task 1",))
        assert with_today.has_items is True

        with_yesterday = ExtractionFrame(yesterday_items=("Done task",))
        assert with_yesterday.has_items is True

    def test_extraction_has_blockers(self):
        """has_blockers should check for blockers."""
        no_blockers = ExtractionFrame()
        assert no_blockers.has_blockers is False

        with_blockers = ExtractionFrame(blockers=("Waiting for API access",))
        assert with_blockers.has_blockers is True

    def test_format_zulip_message(self):
        """format_zulip_message should produce markdown."""
        frame = ExtractionFrame(
            today_items=("Code review", "Deploy feature"),
            yesterday_items=("Fixed bug",),
            blockers=("Waiting on design",),
            summary="Progressing well",
        )

        message = frame.format_zulip_message()

        assert "**Morning Standup" in message
        assert "**Yesterday:**" in message
        assert "- [x] Fixed bug" in message
        assert "**Today's Focus:**" in message
        assert "- [ ] Code review" in message
        assert "**Blockers:**" in message
        assert "Waiting on design" in message


class TestZulipMessageFrame:
    """Tests for ZulipMessageFrame."""

    def test_format_confirmation_success(self):
        """format_confirmation should report success."""
        frame = ZulipMessageFrame(
            stream="standup",
            topic="user-updates",
            success=True,
            message_id=12345,
        )

        msg = frame.format_confirmation()
        assert "posted" in msg.lower()
        assert "standup" in msg
        assert "user-updates" in msg

    def test_format_confirmation_failure(self):
        """format_confirmation should report failure."""
        frame = ZulipMessageFrame(
            success=False,
            error="API timeout",
        )

        msg = frame.format_confirmation()
        assert "failed" in msg.lower()
        assert "API timeout" in msg


class TestErrorFrame:
    """Tests for ErrorFrame."""

    def test_error_frame_from_exception(self):
        """from_exception should capture exception details."""
        source = AudioInputFrame(sender_phone="123456")

        try:
            raise ValueError("Invalid audio format")
        except Exception as e:
            error_frame = ErrorFrame.from_exception(
                exc=e,
                processor_name="audio_download",
                source_frame=source,
            )

        assert error_frame.error_message == "Invalid audio format"
        assert error_frame.processor_name == "audio_download"
        assert error_frame.exception_class == "ValueError"
        assert error_frame.source_frame_id == source.id
        assert error_frame.sender_phone == "123456"

    def test_error_classification(self):
        """from_exception should classify error types."""
        timeout_frame = ErrorFrame.from_exception(
            exc=TimeoutError("Connection timed out"),
            processor_name="test",
        )
        assert timeout_frame.error_type == "timeout"

    def test_format_user_message(self):
        """format_user_message should return user-friendly message."""
        frame = ErrorFrame(error_type="network")
        msg = frame.format_user_message()
        assert "try again" in msg.lower()

        frame2 = ErrorFrame(error_type="validation", error_message="Phone required")
        msg2 = frame2.format_user_message()
        assert "invalid" in msg2.lower() or "Phone required" in msg2
