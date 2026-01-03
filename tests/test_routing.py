"""
Tests for Knomly pipeline routing primitives.
"""
import asyncio
from dataclasses import dataclass

import pytest

from knomly.pipeline import (
    Conditional,
    FanOut,
    FanOutStrategy,
    Filter,
    Guard,
    PassthroughProcessor,
    Pipeline,
    PipelineContext,
    PipelineExit,
    Processor,
    RoutingDecision,
    RoutingError,
    Switch,
    TypeRouter,
)
from knomly.pipeline.frames import (
    AudioInputFrame,
    ErrorFrame,
    Frame,
    TextInputFrame,
    TranscriptionFrame,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def ctx() -> PipelineContext:
    """Create a fresh context for each test."""
    return PipelineContext()


@pytest.fixture
def audio_frame() -> AudioInputFrame:
    return AudioInputFrame(
        audio_data=b"test audio",
        sender_phone="919876543210",
    )


@pytest.fixture
def text_frame() -> TextInputFrame:
    return TextInputFrame(
        text="Hello world",
        sender_phone="919876543210",
    )


@pytest.fixture
def high_confidence_transcription() -> TranscriptionFrame:
    return TranscriptionFrame(
        original_text="Test transcription",
        english_text="Test transcription",
        confidence=0.95,
        sender_phone="919876543210",
    )


@pytest.fixture
def low_confidence_transcription() -> TranscriptionFrame:
    return TranscriptionFrame(
        original_text="Test transcription",
        english_text="Test transcription",
        confidence=0.3,
        sender_phone="919876543210",
    )


# =============================================================================
# Helper Processors for Testing
# =============================================================================


class MarkerProcessor(Processor):
    """Processor that adds a marker to frame metadata."""

    def __init__(self, marker: str):
        self._marker = marker

    @property
    def name(self) -> str:
        return f"marker_{self._marker}"

    async def process(self, frame: Frame, ctx: PipelineContext):
        return frame.with_metadata(marker=self._marker)


class TransformProcessor(Processor):
    """Processor that transforms frame type."""

    @property
    def name(self) -> str:
        return "transform"

    async def process(self, frame: Frame, ctx: PipelineContext):
        return TranscriptionFrame(
            original_text="transformed",
            english_text="transformed",
            confidence=1.0,
            sender_phone=getattr(frame, "sender_phone", ""),
        )


class SlowProcessor(Processor):
    """Processor with configurable delay."""

    def __init__(self, delay: float, marker: str):
        self._delay = delay
        self._marker = marker

    @property
    def name(self) -> str:
        return f"slow_{self._marker}"

    async def process(self, frame: Frame, ctx: PipelineContext):
        await asyncio.sleep(self._delay)
        return frame.with_metadata(marker=self._marker)


class FailingProcessor(Processor):
    """Processor that always raises an exception."""

    def __init__(self, error_message: str = "Intentional failure"):
        self._error_message = error_message

    @property
    def name(self) -> str:
        return "failing"

    async def process(self, frame: Frame, ctx: PipelineContext):
        raise RuntimeError(self._error_message)


# =============================================================================
# Conditional Router Tests
# =============================================================================


class TestConditional:
    """Tests for binary conditional routing."""

    @pytest.mark.asyncio
    async def test_routes_to_if_true_when_condition_true(
        self, high_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        router = Conditional(
            condition=lambda f, c: f.confidence > 0.5,
            if_true=MarkerProcessor("high"),
            if_false=MarkerProcessor("low"),
            condition_name="confidence_check",
        )

        result = await router.process(high_confidence_transcription, ctx)

        assert result.metadata["marker"] == "high"

    @pytest.mark.asyncio
    async def test_routes_to_if_false_when_condition_false(
        self, low_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        router = Conditional(
            condition=lambda f, c: f.confidence > 0.5,
            if_true=MarkerProcessor("high"),
            if_false=MarkerProcessor("low"),
        )

        result = await router.process(low_confidence_transcription, ctx)

        assert result.metadata["marker"] == "low"

    @pytest.mark.asyncio
    async def test_records_routing_decision(
        self, high_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        router = Conditional(
            condition=lambda f, c: True,
            if_true=MarkerProcessor("a"),
            if_false=MarkerProcessor("b"),
            condition_name="test_condition",
        )

        await router.process(high_confidence_transcription, ctx)

        assert len(ctx.routing_decisions) == 1
        decision = ctx.routing_decisions[0]
        assert decision.router_name == "Conditional(test_condition)"
        assert decision.selected_branch == "if_true"
        assert "test_condition = True" in decision.evaluated_condition

    @pytest.mark.asyncio
    async def test_supports_async_condition(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        async def async_condition(frame: Frame, ctx: PipelineContext) -> bool:
            await asyncio.sleep(0.001)  # Simulate async operation
            return True

        router = Conditional(
            condition=async_condition,
            if_true=MarkerProcessor("async_true"),
            if_false=MarkerProcessor("async_false"),
        )

        result = await router.process(audio_frame, ctx)

        assert result.metadata["marker"] == "async_true"

    @pytest.mark.asyncio
    async def test_works_with_pipeline_as_branch(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        pipeline_branch = Pipeline([
            MarkerProcessor("step1"),
            MarkerProcessor("step2"),
        ])

        router = Conditional(
            condition=lambda f, c: True,
            if_true=pipeline_branch,
            if_false=MarkerProcessor("false"),
        )

        result = await router.process(audio_frame, ctx)

        # Pipeline returns list, last processor's marker should be present
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0].metadata["marker"] == "step2"


# =============================================================================
# Switch Router Tests
# =============================================================================


class TestSwitch:
    """Tests for multi-way switch routing."""

    @pytest.mark.asyncio
    async def test_routes_to_matching_case(
        self, ctx: PipelineContext
    ):
        frame = TranscriptionFrame(
            original_text="Hola",
            english_text="Hello",
            detected_language="es",
            sender_phone="test",
        )

        router = Switch(
            key=lambda f, c: f.detected_language,
            cases={
                "en": MarkerProcessor("english"),
                "es": MarkerProcessor("spanish"),
                "hi": MarkerProcessor("hindi"),
            },
            key_name="language",
        )

        result = await router.process(frame, ctx)

        assert result.metadata["marker"] == "spanish"

    @pytest.mark.asyncio
    async def test_routes_to_default_when_no_match(
        self, ctx: PipelineContext
    ):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
            detected_language="fr",
            sender_phone="test",
        )

        router = Switch(
            key=lambda f, c: f.detected_language,
            cases={
                "en": MarkerProcessor("english"),
                "es": MarkerProcessor("spanish"),
            },
            default=MarkerProcessor("fallback"),
        )

        result = await router.process(frame, ctx)

        assert result.metadata["marker"] == "fallback"

    @pytest.mark.asyncio
    async def test_raises_when_no_match_and_no_default(
        self, ctx: PipelineContext
    ):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
            detected_language="fr",
            sender_phone="test",
        )

        router = Switch(
            key=lambda f, c: f.detected_language,
            cases={
                "en": MarkerProcessor("english"),
            },
            default=None,
        )

        with pytest.raises(RoutingError) as exc_info:
            await router.process(frame, ctx)

        assert "No case for key 'fr'" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_records_routing_decision(
        self, ctx: PipelineContext
    ):
        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
            detected_language="en",
            sender_phone="test",
        )

        router = Switch(
            key=lambda f, c: f.detected_language,
            cases={"en": MarkerProcessor("english")},
            key_name="lang",
        )

        await router.process(frame, ctx)

        assert len(ctx.routing_decisions) == 1
        decision = ctx.routing_decisions[0]
        assert decision.router_name == "Switch(lang)"
        assert decision.selected_branch == "en"


# =============================================================================
# TypeRouter Tests
# =============================================================================


class TestTypeRouter:
    """Tests for type-based routing."""

    @pytest.mark.asyncio
    async def test_routes_by_exact_type(
        self, audio_frame: AudioInputFrame, text_frame: TextInputFrame, ctx: PipelineContext
    ):
        router = TypeRouter(
            routes={
                AudioInputFrame: MarkerProcessor("audio"),
                TextInputFrame: MarkerProcessor("text"),
            },
        )

        audio_result = await router.process(audio_frame, ctx)
        text_result = await router.process(text_frame, ctx)

        assert audio_result.metadata["marker"] == "audio"
        assert text_result.metadata["marker"] == "text"

    @pytest.mark.asyncio
    async def test_routes_by_inheritance(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        # AudioInputFrame inherits from Frame
        router = TypeRouter(
            routes={
                Frame: MarkerProcessor("generic_frame"),
            },
        )

        result = await router.process(audio_frame, ctx)

        assert result.metadata["marker"] == "generic_frame"

    @pytest.mark.asyncio
    async def test_exact_match_takes_precedence(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        router = TypeRouter(
            routes={
                Frame: MarkerProcessor("generic"),
                AudioInputFrame: MarkerProcessor("specific"),
            },
        )

        result = await router.process(audio_frame, ctx)

        assert result.metadata["marker"] == "specific"

    @pytest.mark.asyncio
    async def test_routes_to_default_when_no_match(
        self, ctx: PipelineContext
    ):
        @dataclass(frozen=True, kw_only=True, slots=True)
        class CustomFrame(Frame):
            custom_field: str = "test"

        router = TypeRouter(
            routes={
                AudioInputFrame: MarkerProcessor("audio"),
            },
            default=MarkerProcessor("default"),
        )

        result = await router.process(CustomFrame(), ctx)

        assert result.metadata["marker"] == "default"

    @pytest.mark.asyncio
    async def test_raises_when_no_match_and_no_default(
        self, text_frame: TextInputFrame, ctx: PipelineContext
    ):
        router = TypeRouter(
            routes={
                AudioInputFrame: MarkerProcessor("audio"),
            },
            default=None,
        )

        with pytest.raises(RoutingError) as exc_info:
            await router.process(text_frame, ctx)

        assert "No route for type TextInputFrame" in str(exc_info.value)


# =============================================================================
# Filter Tests
# =============================================================================


class TestFilter:
    """Tests for filter/gate routing."""

    @pytest.mark.asyncio
    async def test_passes_frame_when_condition_true(
        self, high_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        filter_router = Filter(
            condition=lambda f, c: f.confidence > 0.5,
        )

        result = await filter_router.process(high_confidence_transcription, ctx)

        assert result == high_confidence_transcription

    @pytest.mark.asyncio
    async def test_returns_none_when_condition_false_and_no_reject(
        self, low_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        filter_router = Filter(
            condition=lambda f, c: f.confidence > 0.5,
            on_reject=None,
        )

        result = await filter_router.process(low_confidence_transcription, ctx)

        assert result is None

    @pytest.mark.asyncio
    async def test_returns_reject_frame_when_condition_false(
        self, low_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        reject_frame = ErrorFrame(
            error_type="low_confidence",
            error_message="Confidence too low",
        )

        filter_router = Filter(
            condition=lambda f, c: f.confidence > 0.5,
            on_reject=reject_frame,
        )

        result = await filter_router.process(low_confidence_transcription, ctx)

        assert result == reject_frame

    @pytest.mark.asyncio
    async def test_records_routing_decision(
        self, high_confidence_transcription: TranscriptionFrame, ctx: PipelineContext
    ):
        filter_router = Filter(
            condition=lambda f, c: True,
            condition_name="always_pass",
        )

        await filter_router.process(high_confidence_transcription, ctx)

        assert len(ctx.routing_decisions) == 1
        assert ctx.routing_decisions[0].selected_branch == "pass"


# =============================================================================
# Guard Tests
# =============================================================================


class TestGuard:
    """Tests for guard/early exit routing."""

    @pytest.mark.asyncio
    async def test_continues_when_condition_false(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        exit_frame = ErrorFrame(error_type="blocked")

        guard = Guard(
            condition=lambda f, c: False,
            exit_frame=exit_frame,
        )

        result = await guard.process(audio_frame, ctx)

        assert result == audio_frame

    @pytest.mark.asyncio
    async def test_raises_pipeline_exit_when_condition_true(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        exit_frame = ErrorFrame(
            error_type="spam",
            error_message="Spam detected",
        )

        guard = Guard(
            condition=lambda f, c: True,
            exit_frame=exit_frame,
        )

        with pytest.raises(PipelineExit) as exc_info:
            await guard.process(audio_frame, ctx)

        assert exc_info.value.exit_frame == exit_frame

    @pytest.mark.asyncio
    async def test_pipeline_handles_guard_exit(
        self, audio_frame: AudioInputFrame
    ):
        exit_frame = ErrorFrame(
            error_type="blocked",
            error_message="Access denied",
        )

        pipeline = Pipeline([
            Guard(
                condition=lambda f, c: True,
                exit_frame=exit_frame,
            ),
            MarkerProcessor("should_not_reach"),
        ])

        result = await pipeline.execute(audio_frame)

        assert result.success is True
        assert len(result.output_frames) == 1
        assert result.output_frames[0] == exit_frame


# =============================================================================
# FanOut Tests
# =============================================================================


class TestFanOut:
    """Tests for parallel fan-out routing."""

    @pytest.mark.asyncio
    async def test_executes_all_branches_in_parallel(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(
            branches=[
                MarkerProcessor("branch1"),
                MarkerProcessor("branch2"),
                MarkerProcessor("branch3"),
            ],
            strategy=FanOutStrategy.ALL,
        )

        results = await fanout.process(audio_frame, ctx)

        assert len(results) == 3
        markers = {r.metadata["marker"] for r in results}
        assert markers == {"branch1", "branch2", "branch3"}

    @pytest.mark.asyncio
    async def test_all_settled_collects_successes_and_failures(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(
            branches=[
                MarkerProcessor("success1"),
                FailingProcessor(),
                MarkerProcessor("success2"),
            ],
            strategy=FanOutStrategy.ALL_SETTLED,
        )

        results = await fanout.process(audio_frame, ctx)

        # Should have 2 successful results (failures logged but not raised)
        assert len(results) == 2
        markers = {r.metadata["marker"] for r in results}
        assert markers == {"success1", "success2"}

    @pytest.mark.asyncio
    async def test_first_success_returns_first_completed(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(
            branches=[
                SlowProcessor(0.1, "slow"),
                MarkerProcessor("fast"),
            ],
            strategy=FanOutStrategy.FIRST_SUCCESS,
        )

        result = await fanout.process(audio_frame, ctx)

        # Fast processor should complete first
        assert result.metadata["marker"] == "fast"

    @pytest.mark.asyncio
    async def test_race_returns_first_completed(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(
            branches=[
                SlowProcessor(0.1, "slow"),
                MarkerProcessor("fast"),
            ],
            strategy=FanOutStrategy.RACE,
        )

        result = await fanout.process(audio_frame, ctx)

        assert result.metadata["marker"] == "fast"

    @pytest.mark.asyncio
    async def test_empty_branches_passthrough(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(branches=[], strategy=FanOutStrategy.ALL)

        result = await fanout.process(audio_frame, ctx)

        assert result == audio_frame

    @pytest.mark.asyncio
    async def test_records_routing_decision(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        fanout = FanOut(
            branches=[MarkerProcessor("a"), MarkerProcessor("b")],
            strategy=FanOutStrategy.ALL_SETTLED,
        )

        await fanout.process(audio_frame, ctx)

        assert len(ctx.routing_decisions) == 1
        decision = ctx.routing_decisions[0]
        assert "parallel(2)" in decision.selected_branch


# =============================================================================
# Integration Tests
# =============================================================================


class TestRoutingIntegration:
    """Integration tests for routing with pipelines."""

    @pytest.mark.asyncio
    async def test_nested_conditionals(
        self, ctx: PipelineContext
    ):
        """Test conditionals inside conditionals."""
        inner_router = Conditional(
            condition=lambda f, c: f.confidence > 0.9,
            if_true=MarkerProcessor("excellent"),
            if_false=MarkerProcessor("good"),
        )

        outer_router = Conditional(
            condition=lambda f, c: f.confidence > 0.5,
            if_true=inner_router,
            if_false=MarkerProcessor("low"),
        )

        # Test excellent (> 0.9)
        excellent_frame = TranscriptionFrame(confidence=0.95, sender_phone="test")
        result = await outer_router.process(excellent_frame, ctx)
        assert result.metadata["marker"] == "excellent"

        # Test good (0.5 < x <= 0.9)
        good_frame = TranscriptionFrame(confidence=0.7, sender_phone="test")
        result = await outer_router.process(good_frame, ctx)
        assert result.metadata["marker"] == "good"

        # Test low (<= 0.5)
        low_frame = TranscriptionFrame(confidence=0.3, sender_phone="test")
        result = await outer_router.process(low_frame, ctx)
        assert result.metadata["marker"] == "low"

    @pytest.mark.asyncio
    async def test_switch_with_pipeline_branches(
        self, ctx: PipelineContext
    ):
        """Test switch routing to different pipelines."""
        router = Switch(
            key=lambda f, c: f.detected_language,
            cases={
                "en": Pipeline([
                    MarkerProcessor("en_step1"),
                    MarkerProcessor("en_step2"),
                ]),
                "es": Pipeline([
                    MarkerProcessor("es_step1"),
                    MarkerProcessor("es_step2"),
                ]),
            },
            default=MarkerProcessor("default"),
        )

        en_frame = TranscriptionFrame(detected_language="en", sender_phone="test")
        result = await router.process(en_frame, ctx)

        assert isinstance(result, list)
        assert result[0].metadata["marker"] == "en_step2"

    @pytest.mark.asyncio
    async def test_routing_in_pipeline(
        self, audio_frame: AudioInputFrame
    ):
        """Test routers as processors in a pipeline."""
        pipeline = Pipeline([
            TransformProcessor(),  # AudioInput -> Transcription
            Filter(
                condition=lambda f, c: f.confidence > 0.5,
                on_reject=ErrorFrame(error_type="low_confidence"),
            ),
            Conditional(
                condition=lambda f, c: f.is_translated,
                if_true=MarkerProcessor("translated"),
                if_false=MarkerProcessor("original"),
            ),
        ])

        result = await pipeline.execute(audio_frame)

        assert result.success
        assert len(result.output_frames) == 1
        # TransformProcessor creates English transcription (not translated)
        assert result.output_frames[0].metadata["marker"] == "original"

    @pytest.mark.asyncio
    async def test_complex_routing_workflow(self):
        """Test complex workflow with multiple routing types."""
        @dataclass(frozen=True, kw_only=True, slots=True)
        class WorkflowFrame(Frame):
            stage: str = "start"
            score: float = 0.0

        # Build workflow with type routing, guards, and conditionals
        workflow = Pipeline([
            # First, check authorization
            Guard(
                condition=lambda f, c: f.score < 0,
                exit_frame=ErrorFrame(error_type="unauthorized"),
                condition_name="auth_check",
            ),
            # Then route based on score
            Conditional(
                condition=lambda f, c: f.score > 0.8,
                if_true=MarkerProcessor("premium"),
                if_false=MarkerProcessor("standard"),
                condition_name="tier_check",
            ),
        ])

        # Test authorized premium
        premium_frame = WorkflowFrame(score=0.9)
        result = await workflow.execute(premium_frame)
        assert result.success
        assert result.output_frames[0].metadata["marker"] == "premium"

        # Test authorized standard
        standard_frame = WorkflowFrame(score=0.5)
        result = await workflow.execute(standard_frame)
        assert result.success
        assert result.output_frames[0].metadata["marker"] == "standard"

        # Test unauthorized
        unauthorized_frame = WorkflowFrame(score=-1)
        result = await workflow.execute(unauthorized_frame)
        assert result.success  # Guard returns cleanly
        assert isinstance(result.output_frames[0], ErrorFrame)
        assert result.output_frames[0].error_type == "unauthorized"

    @pytest.mark.asyncio
    async def test_context_copy_isolates_routing_decisions(self):
        """Test that context copy isolates routing decisions."""
        ctx = PipelineContext()

        # Record some decisions
        ctx.routing_decisions.append(
            RoutingDecision(
                timestamp=ctx.started_at,
                router_name="test",
                frame_id=ctx.execution_id,
                frame_type="Frame",
                selected_branch="test",
            )
        )

        # Copy
        ctx_copy = ctx.copy()

        # Modify original
        ctx.routing_decisions.append(
            RoutingDecision(
                timestamp=ctx.started_at,
                router_name="test2",
                frame_id=ctx.execution_id,
                frame_type="Frame",
                selected_branch="test2",
            )
        )

        # Copy should not be affected
        assert len(ctx.routing_decisions) == 2
        assert len(ctx_copy.routing_decisions) == 1
