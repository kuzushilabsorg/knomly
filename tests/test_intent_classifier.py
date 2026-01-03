"""
Tests for IntentClassifierProcessor.
"""

import pytest

from knomly.pipeline.context import PipelineContext
from knomly.pipeline.frames import Frame, TranscriptionFrame
from knomly.pipeline.processors import (
    Intent,
    IntentClassifierProcessor,
    get_intent,
)
from knomly.providers.llm import LLMResponse


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, response: str = '{"intent": "standup", "confidence": 0.9}'):
        self._response = response

    @property
    def name(self) -> str:
        return "mock_llm"

    async def complete(self, messages: list, config=None) -> LLMResponse:
        return LLMResponse(
            content=self._response,
            model="mock-model",
            provider=self.name,
        )


class MockProviderRegistry:
    """Mock registry for testing."""

    def __init__(self, llm_response: str = '{"intent": "standup", "confidence": 0.9}'):
        self.llm = MockLLMProvider(llm_response)

    def get_llm(self, name=None):
        return self.llm


class TestIntent:
    """Tests for Intent enum."""

    def test_intent_values(self):
        assert Intent.STANDUP.value == "standup"
        assert Intent.TASK.value == "task"
        assert Intent.REMINDER.value == "reminder"
        assert Intent.QUERY.value == "query"
        assert Intent.UNKNOWN.value == "unknown"


class TestIntentClassifierProcessor:
    """Tests for IntentClassifierProcessor."""

    @pytest.fixture
    def processor(self):
        return IntentClassifierProcessor()

    def test_processor_name(self, processor):
        assert processor.name == "intent_classifier"

    @pytest.mark.asyncio
    async def test_passes_through_non_transcription_frame(self, processor):
        frame = Frame()
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert result is frame

    @pytest.mark.asyncio
    async def test_empty_text_returns_unknown_intent(self, processor):
        frame = TranscriptionFrame(original_text="", english_text="")
        ctx = PipelineContext()

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == Intent.UNKNOWN.value
        assert result.metadata.get("intent_confidence") == 0.0

    @pytest.mark.asyncio
    async def test_raises_when_no_providers(self, processor):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()

        with pytest.raises(RuntimeError, match="No providers"):
            await processor.process(frame, ctx)

    @pytest.mark.asyncio
    async def test_classifies_standup_intent(self, processor):
        frame = TranscriptionFrame(
            original_text="Today I'm working on task X",
            english_text="Today I'm working on task X",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('{"intent": "standup", "confidence": 0.95}')

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == "standup"
        assert result.metadata.get("intent_confidence") == 0.95

    @pytest.mark.asyncio
    async def test_classifies_task_intent(self, processor):
        frame = TranscriptionFrame(
            original_text="Create a task to review the code",
            english_text="Create a task to review the code",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('{"intent": "task", "confidence": 0.85}')

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == "task"

    @pytest.mark.asyncio
    async def test_low_confidence_returns_unknown(self, processor):
        frame = TranscriptionFrame(
            original_text="Hmm not sure what this is",
            english_text="Hmm not sure what this is",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('{"intent": "standup", "confidence": 0.4}')

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == Intent.UNKNOWN.value

    @pytest.mark.asyncio
    async def test_custom_confidence_threshold(self):
        processor = IntentClassifierProcessor(confidence_threshold=0.8)
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('{"intent": "standup", "confidence": 0.75}')

        result = await processor.process(frame, ctx)

        # 0.75 < 0.8 threshold, so should be unknown
        assert result.metadata.get("intent") == Intent.UNKNOWN.value

    @pytest.mark.asyncio
    async def test_handles_markdown_wrapped_json(self, processor):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('```json\n{"intent": "task", "confidence": 0.9}\n```')

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == "task"

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self, processor):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry("not valid json")

        result = await processor.process(frame, ctx)

        # Unparseable LLM response returns unknown (LLM completed but returned garbage)
        # Note: STANDUP fallback is only for actual exceptions (network errors, etc.)
        assert result.metadata.get("intent") == Intent.UNKNOWN.value
        assert result.metadata.get("intent_confidence") == 0.0

    @pytest.mark.asyncio
    async def test_handles_unknown_intent_value(self, processor):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()
        ctx.providers = MockProviderRegistry('{"intent": "invalid_intent", "confidence": 0.9}')

        result = await processor.process(frame, ctx)

        assert result.metadata.get("intent") == Intent.UNKNOWN.value


class TestGetIntent:
    """Tests for get_intent helper function."""

    def test_extracts_intent_from_metadata(self):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        ).with_metadata(intent="standup")
        ctx = PipelineContext()

        intent = get_intent(frame, ctx)

        assert intent == "standup"

    def test_returns_unknown_when_no_intent(self):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
        )
        ctx = PipelineContext()

        intent = get_intent(frame, ctx)

        assert intent == Intent.UNKNOWN.value
