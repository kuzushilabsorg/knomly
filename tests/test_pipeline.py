"""
Tests for Knomly pipeline execution.

Tests Pipeline, PipelineBuilder, and processor integration.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knomly.pipeline import Pipeline, PipelineBuilder, PipelineContext, PipelineResult
from knomly.pipeline.processor import Processor
from knomly.pipeline.frames import (
    AudioInputFrame,
    ErrorFrame,
    Frame,
    TranscriptionFrame,
)


class MockPassthroughProcessor(Processor):
    """Test processor that passes frames through unchanged."""

    @property
    def name(self) -> str:
        return "mock_passthrough"

    async def process(self, frame: Frame, ctx: PipelineContext):
        return frame


class MockTransformProcessor(Processor):
    """Test processor that transforms AudioInput to Transcription."""

    @property
    def name(self) -> str:
        return "mock_transform"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if isinstance(frame, AudioInputFrame):
            return TranscriptionFrame(
                original_text="Mocked transcription",
                english_text="Mocked transcription",
                sender_phone=frame.sender_phone,
                source_frame_id=frame.id,
            )
        return frame


class MockFilterProcessor(Processor):
    """Test processor that filters out certain frames."""

    @property
    def name(self) -> str:
        return "mock_filter"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if isinstance(frame, AudioInputFrame) and not frame.has_audio:
            return None  # Filter out frames without audio
        return frame


class MockFanOutProcessor(Processor):
    """Test processor that produces multiple output frames."""

    @property
    def name(self) -> str:
        return "mock_fanout"

    async def process(self, frame: Frame, ctx: PipelineContext):
        return [
            frame,
            frame.derive(metadata={"copy": True}),
        ]


class MockErrorProcessor(Processor):
    """Test processor that raises an error."""

    @property
    def name(self) -> str:
        return "mock_error"

    async def process(self, frame: Frame, ctx: PipelineContext):
        raise ValueError("Test error")


class TestPipeline:
    """Tests for Pipeline execution."""

    @pytest.mark.asyncio
    async def test_pipeline_executes_processors_in_order(self):
        """Pipeline should execute processors sequentially."""
        execution_order = []

        class OrderTrackingProcessor(Processor):
            def __init__(self, name: str):
                self._name = name

            @property
            def name(self) -> str:
                return self._name

            async def process(self, frame: Frame, ctx: PipelineContext):
                execution_order.append(self._name)
                return frame

        pipeline = Pipeline([
            OrderTrackingProcessor("first"),
            OrderTrackingProcessor("second"),
            OrderTrackingProcessor("third"),
        ])

        frame = AudioInputFrame()
        await pipeline.execute(frame)

        assert execution_order == ["first", "second", "third"]

    @pytest.mark.asyncio
    async def test_pipeline_passthrough(self):
        """Pipeline with passthrough processors should preserve frame."""
        pipeline = Pipeline([
            MockPassthroughProcessor(),
            MockPassthroughProcessor(),
        ])

        input_frame = AudioInputFrame(sender_phone="123456")
        result = await pipeline.execute(input_frame)

        assert result.success is True
        assert len(result.output_frames) == 1
        assert result.output_frames[0].sender_phone == "123456"

    @pytest.mark.asyncio
    async def test_pipeline_transform(self):
        """Pipeline should handle frame transformations."""
        pipeline = Pipeline([
            MockTransformProcessor(),
        ])

        input_frame = AudioInputFrame(sender_phone="123456")
        result = await pipeline.execute(input_frame)

        assert result.success is True
        assert len(result.output_frames) == 1
        assert isinstance(result.output_frames[0], TranscriptionFrame)
        assert result.output_frames[0].sender_phone == "123456"

    @pytest.mark.asyncio
    async def test_pipeline_filter(self):
        """Pipeline should handle filtered frames (None return)."""
        pipeline = Pipeline([
            MockFilterProcessor(),
        ])

        # Frame without audio should be filtered
        input_frame = AudioInputFrame(media_url="http://test.com")
        result = await pipeline.execute(input_frame)

        assert result.success is True
        assert len(result.output_frames) == 0

    @pytest.mark.asyncio
    async def test_pipeline_fanout(self):
        """Pipeline should handle multiple output frames."""
        pipeline = Pipeline([
            MockFanOutProcessor(),
        ])

        input_frame = AudioInputFrame()
        result = await pipeline.execute(input_frame)

        assert result.success is True
        assert len(result.output_frames) == 2

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Pipeline should convert exceptions to ErrorFrames."""
        pipeline = Pipeline([
            MockErrorProcessor(),
        ])

        input_frame = AudioInputFrame()
        result = await pipeline.execute(input_frame)

        # Should have an error frame
        assert len(result.output_frames) == 1
        assert isinstance(result.output_frames[0], ErrorFrame)
        assert "Test error" in result.output_frames[0].error_message
        assert result.output_frames[0].processor_name == "mock_error"

    @pytest.mark.asyncio
    async def test_pipeline_context_passed_to_processors(self):
        """Pipeline should pass context to all processors."""
        received_contexts = []

        class ContextCapturingProcessor(Processor):
            @property
            def name(self) -> str:
                return "context_capture"

            async def process(self, frame: Frame, ctx: PipelineContext):
                received_contexts.append(ctx)
                return frame

        pipeline = Pipeline([
            ContextCapturingProcessor(),
            ContextCapturingProcessor(),
        ])

        ctx = PipelineContext(sender_phone="test")
        await pipeline.execute(AudioInputFrame(), ctx)

        assert len(received_contexts) == 2
        assert all(c.sender_phone == "test" for c in received_contexts)
        assert received_contexts[0] is received_contexts[1]  # Same context object

    @pytest.mark.asyncio
    async def test_pipeline_records_timings(self):
        """Pipeline should record processor timings."""
        pipeline = Pipeline([
            MockPassthroughProcessor(),
        ])

        ctx = PipelineContext()
        await pipeline.execute(AudioInputFrame(), ctx)

        assert "mock_passthrough" in ctx.processor_timings
        assert ctx.processor_timings["mock_passthrough"] >= 0

    @pytest.mark.asyncio
    async def test_pipeline_creates_default_context(self):
        """Pipeline should create context if not provided."""
        pipeline = Pipeline([MockPassthroughProcessor()])

        result = await pipeline.execute(AudioInputFrame())

        assert result.context is not None
        assert result.execution_id is not None

    def test_pipeline_requires_processors(self):
        """Pipeline should require at least one processor."""
        with pytest.raises(ValueError):
            Pipeline([])


class TestPipelineBuilder:
    """Tests for PipelineBuilder."""

    def test_builder_add_processors(self):
        """Builder should add processors in order."""
        pipeline = (
            PipelineBuilder()
            .add(MockPassthroughProcessor())
            .add(MockTransformProcessor())
            .build()
        )

        assert len(pipeline.processors) == 2
        assert pipeline.processor_names == ["mock_passthrough", "mock_transform"]

    def test_builder_add_if_condition_true(self):
        """add_if should add processor when condition is true."""
        pipeline = (
            PipelineBuilder()
            .add(MockPassthroughProcessor())
            .add_if(True, MockTransformProcessor())
            .build()
        )

        assert len(pipeline.processors) == 2

    def test_builder_add_if_condition_false(self):
        """add_if should skip processor when condition is false."""
        pipeline = (
            PipelineBuilder()
            .add(MockPassthroughProcessor())
            .add_if(False, MockTransformProcessor())
            .build()
        )

        assert len(pipeline.processors) == 1


class TestPipelineContext:
    """Tests for PipelineContext."""

    def test_context_has_execution_id(self):
        """Context should have auto-generated execution ID."""
        ctx = PipelineContext()
        assert ctx.execution_id is not None

    def test_context_has_timestamp(self):
        """Context should have started_at timestamp."""
        ctx = PipelineContext()
        assert ctx.started_at is not None

    def test_context_elapsed_ms(self):
        """elapsed_ms should return positive value."""
        ctx = PipelineContext()
        # Small sleep to ensure time passes
        import time
        time.sleep(0.01)
        assert ctx.elapsed_ms > 0

    def test_context_record_timing(self):
        """record_timing should store processor timings."""
        ctx = PipelineContext()
        ctx.record_timing("test_processor", 123.45)

        assert ctx.processor_timings["test_processor"] == 123.45

    def test_context_record_frame(self):
        """record_frame should add to frame log."""
        ctx = PipelineContext()
        frame_dict = {"id": "test", "type": "AudioInputFrame"}
        ctx.record_frame(frame_dict, "processor_name")

        assert len(ctx.frame_log) == 1
        assert ctx.frame_log[0]["frame"] == frame_dict
        assert ctx.frame_log[0]["processor"] == "processor_name"


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_result_success_default(self):
        """Result should default to success."""
        result = PipelineResult(context=PipelineContext())
        assert result.success is True

    def test_result_get_frame_by_type(self):
        """get_frame should find frame by type."""
        ctx = PipelineContext()
        result = PipelineResult(
            context=ctx,
            output_frames=[
                AudioInputFrame(),
                TranscriptionFrame(original_text="test"),
            ],
        )

        trans = result.get_frame(TranscriptionFrame)
        assert trans is not None
        assert trans.original_text == "test"

    def test_result_get_frame_not_found(self):
        """get_frame should return None if type not found."""
        result = PipelineResult(
            context=PipelineContext(),
            output_frames=[AudioInputFrame()],
        )

        assert result.get_frame(TranscriptionFrame) is None
