"""
Tests for Knomly Pipeline Processors.

Tests each processor in isolation with mock providers.
"""
import asyncio
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knomly.pipeline.context import PipelineContext
from knomly.pipeline.frames import (
    AudioInputFrame,
    ConfirmationFrame,
    ErrorFrame,
    ExtractionFrame,
    Frame,
    TranscriptionFrame,
    ZulipMessageFrame,
)
from knomly.pipeline.processor import PassthroughProcessor, Processor
from knomly.pipeline.processors import (
    AudioDownloadProcessor,
    ConfirmationProcessor,
    ExtractionProcessor,
    TranscriptionProcessor,
    ZulipProcessor,
)
from knomly.providers import TranscriptionResult
from knomly.providers.chat import MessageResult
from knomly.providers.llm import LLMResponse


# =============================================================================
# Mock Providers
# =============================================================================


class MockSTTProvider:
    """Mock STT provider for testing."""

    def __init__(self, name: str = "mock_stt"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def transcribe(
        self,
        audio_data: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            original_text="Hola mundo",
            english_text="Hello world",
            detected_language="es",
            language_name="Spanish",
            confidence=0.95,
            provider=self.name,
        )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str = "mock_llm"):
        self._name = name
        self.default_model = "mock-model"

    @property
    def name(self) -> str:
        return self._name

    async def complete(self, messages: list, config: Any = None) -> LLMResponse:
        return LLMResponse(
            content='{"today_items": ["Task 1", "Task 2"], "yesterday_items": [], "blockers": ["Issue 1"], "summary": "Working on tasks"}',
            model="mock-model",
            provider=self.name,
        )


class MockChatProvider:
    """Mock Chat provider for testing."""

    def __init__(self, name: str = "mock_chat"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        return MessageResult(
            success=True,
            message_id=12345,
        )


class MockProviderRegistry:
    """Mock registry with all providers."""

    def __init__(self):
        self.stt = MockSTTProvider()
        self.llm = MockLLMProvider()
        self.chat = MockChatProvider()

    def get_stt(self, name: str | None = None) -> MockSTTProvider:
        return self.stt

    def get_llm(self, name: str | None = None) -> MockLLMProvider:
        return self.llm

    def get_chat(self, name: str | None = None) -> MockChatProvider:
        return self.chat


# =============================================================================
# Base Processor Tests
# =============================================================================


class TestBaseProcessor:
    """Tests for Processor base class."""

    def test_passthrough_processor_name(self):
        proc = PassthroughProcessor()
        assert proc.name == "passthrough"

    @pytest.mark.asyncio
    async def test_passthrough_returns_same_frame(self):
        proc = PassthroughProcessor()
        frame = Frame()
        ctx = PipelineContext()

        result = await proc.process(frame, ctx)

        assert result is frame

    @pytest.mark.asyncio
    async def test_process_frame_yields_single_result(self):
        proc = PassthroughProcessor()
        frame = Frame()
        ctx = PipelineContext()

        results = []
        async for output in proc.process_frame(frame, ctx):
            results.append(output)

        assert len(results) == 1
        assert results[0] is frame

    @pytest.mark.asyncio
    async def test_process_frame_handles_none(self):
        """Test that None result stops propagation."""

        class StopProcessor(Processor):
            @property
            def name(self) -> str:
                return "stop"

            async def process(self, frame, ctx):
                return None

        proc = StopProcessor()
        frame = Frame()
        ctx = PipelineContext()

        results = []
        async for output in proc.process_frame(frame, ctx):
            results.append(output)

        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_process_frame_handles_sequence(self):
        """Test that sequence of frames is yielded."""

        class FanOutProcessor(Processor):
            @property
            def name(self) -> str:
                return "fanout"

            async def process(self, frame, ctx):
                return [Frame(), Frame(), Frame()]

        proc = FanOutProcessor()
        frame = Frame()
        ctx = PipelineContext()

        results = []
        async for output in proc.process_frame(frame, ctx):
            results.append(output)

        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_process_frame_wraps_exception_in_error_frame(self):
        """Test that exceptions become ErrorFrames."""

        class FailingProcessor(Processor):
            @property
            def name(self) -> str:
                return "failing"

            async def process(self, frame, ctx):
                raise ValueError("Test error")

        proc = FailingProcessor()
        frame = Frame()
        ctx = PipelineContext()

        results = []
        async for output in proc.process_frame(frame, ctx):
            results.append(output)

        assert len(results) == 1
        assert isinstance(results[0], ErrorFrame)
        assert "Test error" in results[0].error_message
        assert results[0].processor_name == "failing"

    @pytest.mark.asyncio
    async def test_initialize_and_cleanup_are_optional(self):
        """Test default initialize and cleanup are no-ops."""
        proc = PassthroughProcessor()
        ctx = PipelineContext()

        # Should not raise
        await proc.initialize(ctx)
        await proc.cleanup()


# =============================================================================
# AudioDownloadProcessor Tests
# =============================================================================


class TestAudioDownloadProcessor:
    """Tests for AudioDownloadProcessor."""

    @pytest.fixture
    def processor(self):
        return AudioDownloadProcessor(
            twilio_account_sid="test_sid",
            twilio_auth_token="test_token",
        )

    def test_processor_name(self, processor):
        assert processor.name == "audio_download"

    @pytest.mark.asyncio
    async def test_passes_through_non_audio_frame(self, processor):
        frame = Frame()
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert result is frame

    @pytest.mark.asyncio
    async def test_passes_through_frame_with_audio(self, processor):
        frame = AudioInputFrame(
            audio_data=b"existing audio",
            mime_type="audio/ogg",
        )
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert result is frame

    @pytest.mark.asyncio
    async def test_raises_when_no_url(self, processor):
        frame = AudioInputFrame()  # No URL, no audio
        ctx = PipelineContext()

        with pytest.raises(ValueError, match="no media_url"):
            await processor.process(frame, ctx)

    @pytest.mark.asyncio
    async def test_downloads_audio_from_url(self, processor):
        frame = AudioInputFrame(
            media_url="https://api.twilio.com/audio/test.ogg",
        )
        ctx = PipelineContext()

        with patch("httpx.AsyncClient") as mock_client_class:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.content = b"downloaded audio data"
            mock_response.headers = {"content-type": "audio/ogg"}
            mock_response.raise_for_status = MagicMock()
            mock_client.get.return_value = mock_response
            mock_client.__aenter__.return_value = mock_client
            mock_client.__aexit__.return_value = None
            mock_client_class.return_value = mock_client

            result = await processor.process(frame, ctx)

            assert isinstance(result, AudioInputFrame)
            assert result.audio_data == b"downloaded audio data"
            assert result.has_audio


# =============================================================================
# TranscriptionProcessor Tests
# =============================================================================


class TestTranscriptionProcessor:
    """Tests for TranscriptionProcessor."""

    @pytest.fixture
    def processor(self):
        return TranscriptionProcessor()

    @pytest.fixture
    def ctx_with_providers(self):
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry()
        return ctx

    def test_processor_name(self, processor):
        assert processor.name == "transcription"

    @pytest.mark.asyncio
    async def test_passes_through_non_audio_frame(self, processor, ctx_with_providers):
        frame = Frame()

        result = await processor.process(frame, ctx_with_providers)

        assert result is frame

    @pytest.mark.asyncio
    async def test_raises_when_no_audio_data(self, processor, ctx_with_providers):
        frame = AudioInputFrame()  # No audio data

        with pytest.raises(ValueError, match="no audio_data"):
            await processor.process(frame, ctx_with_providers)

    @pytest.mark.asyncio
    async def test_raises_when_no_providers(self, processor):
        frame = AudioInputFrame(audio_data=b"audio")
        ctx = PipelineContext()  # No providers

        with pytest.raises(RuntimeError, match="No providers"):
            await processor.process(frame, ctx)

    @pytest.mark.asyncio
    async def test_transcribes_audio(self, processor, ctx_with_providers):
        frame = AudioInputFrame(
            audio_data=b"audio data",
            mime_type="audio/ogg",
            sender_phone="+1234567890",
        )

        result = await processor.process(frame, ctx_with_providers)

        assert isinstance(result, TranscriptionFrame)
        assert result.original_text == "Hola mundo"
        assert result.english_text == "Hello world"
        assert result.detected_language == "es"
        assert result.sender_phone == "+1234567890"


# =============================================================================
# ExtractionProcessor Tests
# =============================================================================


class TestExtractionProcessor:
    """Tests for ExtractionProcessor."""

    @pytest.fixture
    def processor(self):
        return ExtractionProcessor()

    @pytest.fixture
    def ctx_with_providers(self):
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry()
        ctx.user_name = "Test User"
        ctx.zulip_stream = "standup"
        ctx.zulip_topic = "test-updates"
        return ctx

    def test_processor_name(self, processor):
        assert processor.name == "extraction"

    @pytest.mark.asyncio
    async def test_passes_through_non_transcription_frame(
        self, processor, ctx_with_providers
    ):
        frame = Frame()

        result = await processor.process(frame, ctx_with_providers)

        assert result is frame

    @pytest.mark.asyncio
    async def test_raises_when_empty_text(self, processor, ctx_with_providers):
        frame = TranscriptionFrame(
            original_text="",
            english_text="",
        )

        with pytest.raises(ValueError, match="empty text"):
            await processor.process(frame, ctx_with_providers)

    @pytest.mark.asyncio
    async def test_raises_when_no_providers(self, processor):
        frame = TranscriptionFrame(original_text="Hello", english_text="Hello")
        ctx = PipelineContext()

        with pytest.raises(RuntimeError, match="No providers"):
            await processor.process(frame, ctx)

    @pytest.mark.asyncio
    async def test_extracts_standup_items(self, processor, ctx_with_providers):
        frame = TranscriptionFrame(
            original_text="I'm working on Task 1 and Task 2. Issue 1 is blocking me.",
            english_text="I'm working on Task 1 and Task 2. Issue 1 is blocking me.",
            sender_phone="+1234567890",
        )

        result = await processor.process(frame, ctx_with_providers)

        assert isinstance(result, ExtractionFrame)
        assert result.today_items == ("Task 1", "Task 2")
        assert result.blockers == ("Issue 1",)
        assert result.summary == "Working on tasks"
        assert result.zulip_stream == "standup"
        assert result.zulip_topic == "test-updates"

    @pytest.mark.asyncio
    async def test_parses_json_with_markdown(self, processor, ctx_with_providers):
        """Test that processor handles LLM response with markdown code blocks."""
        # Mock LLM to return markdown-wrapped JSON
        ctx_with_providers.providers.llm.complete = AsyncMock(
            return_value=LLMResponse(
                content='```json\n{"today_items": ["Task"], "blockers": [], "summary": "Test"}\n```',
                model="test",
                provider="test",
            )
        )

        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )

        result = await processor.process(frame, ctx_with_providers)

        assert isinstance(result, ExtractionFrame)
        assert result.today_items == ("Task",)


# =============================================================================
# ZulipProcessor Tests
# =============================================================================


class TestZulipProcessor:
    """Tests for ZulipProcessor."""

    @pytest.fixture
    def processor(self):
        return ZulipProcessor()

    @pytest.fixture
    def ctx_with_providers(self):
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry()
        return ctx

    def test_processor_name(self, processor):
        assert processor.name == "zulip"

    @pytest.mark.asyncio
    async def test_passes_through_non_extraction_frame(
        self, processor, ctx_with_providers
    ):
        frame = Frame()

        result = await processor.process(frame, ctx_with_providers)

        assert result is frame

    @pytest.mark.asyncio
    async def test_returns_failure_when_no_items(self, processor, ctx_with_providers):
        frame = ExtractionFrame(
            today_items=(),
            zulip_stream="standup",
            zulip_topic="test",
        )

        result = await processor.process(frame, ctx_with_providers)

        assert isinstance(result, ZulipMessageFrame)
        assert result.success is False
        assert "No standup items" in result.error

    @pytest.mark.asyncio
    async def test_raises_when_no_providers(self, processor):
        frame = ExtractionFrame(
            today_items=("Task 1",),
            zulip_stream="standup",
            zulip_topic="test",
        )
        ctx = PipelineContext()

        with pytest.raises(RuntimeError, match="No providers"):
            await processor.process(frame, ctx)

    @pytest.mark.asyncio
    async def test_posts_message_to_zulip(self, processor, ctx_with_providers):
        frame = ExtractionFrame(
            today_items=("Task 1", "Task 2"),
            blockers=("Blocker 1",),
            summary="Working on tasks",
            zulip_stream="standup",
            zulip_topic="test-updates",
            sender_phone="+1234567890",
        )

        result = await processor.process(frame, ctx_with_providers)

        assert isinstance(result, ZulipMessageFrame)
        assert result.success is True
        assert result.message_id == 12345
        assert result.stream == "standup"
        assert result.topic == "test-updates"
        assert "Task 1" in result.content
        assert "Blocker 1" in result.content


# =============================================================================
# ConfirmationProcessor Tests
# =============================================================================


class TestConfirmationProcessor:
    """Tests for ConfirmationProcessor."""

    @pytest.fixture
    def processor(self):
        return ConfirmationProcessor(
            twilio_account_sid="test_sid",
            twilio_auth_token="test_token",
            twilio_from_number="whatsapp:+14155238886",
        )

    def test_processor_name(self, processor):
        assert processor.name == "confirmation"

    @pytest.mark.asyncio
    async def test_passes_through_non_zulip_frame(self, processor):
        frame = Frame()
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert result is frame

    @pytest.mark.asyncio
    async def test_returns_failure_when_no_recipient(self, processor):
        frame = ZulipMessageFrame(
            stream="standup",
            topic="test",
            content="Test message",
            success=True,
            sender_phone="",  # No recipient
        )
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert isinstance(result, ConfirmationFrame)
        assert result.success is False
        assert "No recipient" in result.error

    @pytest.mark.asyncio
    async def test_sends_confirmation_for_success(self, processor):
        from knomly.pipeline.transports import SendResult

        frame = ZulipMessageFrame(
            stream="standup",
            topic="test-updates",
            content="Test message",
            message_id=12345,
            success=True,
            sender_phone="+1234567890",
        )
        ctx = PipelineContext()

        with patch.object(processor, "_send_message") as mock_send:
            mock_send.return_value = SendResult(success=True, message_id="SM123")

            result = await processor.process(frame, ctx)

            assert isinstance(result, ConfirmationFrame)
            assert result.success is True
            assert result.message_sid == "SM123"
            mock_send.assert_called_once()

    @pytest.mark.asyncio
    async def test_handles_error_frame(self, processor):
        from knomly.pipeline.transports import SendResult

        frame = ErrorFrame(
            error_type="network",
            error_message="Connection failed",
            sender_phone="+1234567890",
        )
        ctx = PipelineContext()

        with patch.object(processor, "_send_message") as mock_send:
            mock_send.return_value = SendResult(success=True, message_id="SM456")

            result = await processor.process(frame, ctx)

            assert isinstance(result, ConfirmationFrame)
            assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestProcessorIntegration:
    """Integration tests for processor chain."""

    @pytest.mark.asyncio
    async def test_transcription_to_extraction_flow(self):
        """Test TranscriptionFrame flows correctly through ExtractionProcessor."""
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry()
        ctx.user_name = "Test User"
        ctx.zulip_stream = "standup"
        ctx.zulip_topic = "test-updates"

        transcription = TranscriptionFrame(
            original_text="Working on feature X today",
            english_text="Working on feature X today",
            sender_phone="+1234567890",
        )

        processor = ExtractionProcessor()
        results = []
        async for output in processor.process_frame(transcription, ctx):
            results.append(output)

        assert len(results) == 1
        assert isinstance(results[0], ExtractionFrame)
        assert results[0].zulip_stream == "standup"

    @pytest.mark.asyncio
    async def test_extraction_to_zulip_flow(self):
        """Test ExtractionFrame flows correctly through ZulipProcessor."""
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry()

        extraction = ExtractionFrame(
            today_items=("Task 1", "Task 2"),
            blockers=(),
            summary="Working on tasks",
            zulip_stream="standup",
            zulip_topic="test-updates",
            sender_phone="+1234567890",
        )

        processor = ZulipProcessor()
        results = []
        async for output in processor.process_frame(extraction, ctx):
            results.append(output)

        assert len(results) == 1
        assert isinstance(results[0], ZulipMessageFrame)
        assert results[0].success is True
        assert results[0].message_id == 12345

    @pytest.mark.asyncio
    async def test_error_handling_in_chain(self):
        """Test that errors are properly wrapped in ErrorFrames."""
        ctx = PipelineContext()
        # No providers - should cause error

        transcription = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )

        processor = ExtractionProcessor()
        results = []
        async for output in processor.process_frame(transcription, ctx):
            results.append(output)

        assert len(results) == 1
        assert isinstance(results[0], ErrorFrame)
        assert "No providers" in results[0].error_message
