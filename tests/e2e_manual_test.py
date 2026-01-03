"""
End-to-End Manual Test for Knomly Pipeline.

Tests the complete flow:
AudioInputFrame -> Download -> Transcription -> Intent -> Routing -> Extraction -> Zulip -> Confirmation

Run with: python -m pytest tests/e2e_manual_test.py -v -s
"""

import asyncio
import logging

import pytest

# Configure logging to see the full flow
logging.basicConfig(level=logging.INFO, format="%(name)s | %(message)s")

# Framework imports
from knomly.pipeline import PipelineBuilder, PipelineContext, Switch
from knomly.pipeline.frames import (
    AudioInputFrame,
    ConfirmationFrame,
    ExtractionFrame,
    Frame,
    TranscriptionFrame,
    ZulipMessageFrame,
)
from knomly.pipeline.processor import Processor
from knomly.pipeline.processors import (
    Intent,
    IntentClassifierProcessor,
    get_intent,
)
from knomly.providers.chat.base import MessageResult
from knomly.providers.llm.base import LLMConfig, LLMResponse, Message
from knomly.providers.registry import ProviderRegistry
from knomly.providers.stt.base import TranscriptionResult

# =============================================================================
# Mock Providers
# =============================================================================


class MockSTTProvider:
    """Mock STT that returns predefined transcription."""

    def __init__(
        self, transcription: str = "Today I'm working on the pipeline framework. No blockers."
    ):
        self._transcription = transcription

    @property
    def name(self) -> str:
        return "mock_stt"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        print(f"  [MockSTT] Transcribing {len(audio_bytes)} bytes...")
        return TranscriptionResult(
            original_text=self._transcription,
            english_text=self._transcription,
            detected_language="en",
            language_name="English",
            confidence=0.95,
            provider=self.name,
        )


class MockLLMProvider:
    """Mock LLM that returns predefined responses based on prompt content."""

    @property
    def name(self) -> str:
        return "mock_llm"

    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        prompt = messages[-1].content if messages else ""

        # Detect if this is intent classification
        if "Classify the intent" in prompt:
            print("  [MockLLM] Classifying intent...")
            return LLMResponse(
                content='{"intent": "standup", "confidence": 0.95}',
                model="mock-model",
                provider=self.name,
            )

        # Detect if this is extraction
        if "standup" in prompt.lower() or "extract" in prompt.lower():
            print("  [MockLLM] Extracting standup...")
            return LLMResponse(
                content="""{
                    "today_items": ["Working on pipeline framework", "Writing tests"],
                    "yesterday_items": ["Reviewed architecture"],
                    "blockers": [],
                    "summary": "Making good progress on Knomly"
                }""",
                model="mock-model",
                provider=self.name,
            )

        return LLMResponse(
            content="Unknown request",
            model="mock-model",
            provider=self.name,
        )


class MockChatProvider:
    """Mock Chat provider that records messages sent."""

    def __init__(self):
        self.sent_messages: list[dict] = []

    @property
    def name(self) -> str:
        return "mock_chat"

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        print(f"  [MockChat] Sending to {stream}/{topic}...")
        self.sent_messages.append(
            {
                "stream": stream,
                "topic": topic,
                "content": content,
            }
        )
        return MessageResult(
            success=True,
            message_id=12345,
            stream=stream,
            topic=topic,
        )


# =============================================================================
# Mock Processors (to avoid real HTTP calls)
# =============================================================================


class MockAudioDownloadProcessor(Processor):
    """Mock audio download that returns fake audio bytes."""

    @property
    def name(self) -> str:
        return "mock_audio_download"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if not isinstance(frame, AudioInputFrame):
            return frame

        print(f"  [{self.name}] Downloading audio from {frame.media_url}...")

        # Return frame with fake audio data
        return frame.with_audio(
            audio_data=b"fake_audio_bytes_here",
            mime_type="audio/ogg",
        )


class MockTranscriptionProcessor(Processor):
    """Mock transcription processor using mock STT."""

    @property
    def name(self) -> str:
        return "mock_transcription"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if not isinstance(frame, AudioInputFrame):
            return frame

        if not frame.has_audio:
            raise ValueError("No audio data to transcribe")

        stt = ctx.providers.get_stt()
        result = await stt.transcribe(frame.audio_data, frame.mime_type)

        print(f"  [{self.name}] Transcribed: '{result.original_text[:50]}...'")

        return TranscriptionFrame(
            original_text=result.original_text,
            english_text=result.english_text,
            detected_language=result.detected_language,
            language_name=result.language_name,
            confidence=result.confidence,
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )


class MockExtractionProcessor(Processor):
    """Mock extraction processor using mock LLM."""

    @property
    def name(self) -> str:
        return "mock_extraction"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if not isinstance(frame, TranscriptionFrame):
            print(f"  [{self.name}] Skipping non-TranscriptionFrame: {type(frame).__name__}")
            return frame

        llm = ctx.providers.get_llm()
        response = await llm.complete([Message.user(f"Extract standup from: {frame.text}")])

        import json

        data = json.loads(response.content)

        print(f"  [{self.name}] Extracted: {data.get('today_items', [])}")

        return ExtractionFrame(
            today_items=tuple(data.get("today_items", [])),
            yesterday_items=tuple(data.get("yesterday_items", [])),
            blockers=tuple(data.get("blockers", [])),
            summary=data.get("summary", ""),
            user_name=ctx.user_name,
            sender_phone=frame.sender_phone,
            zulip_stream=ctx.zulip_stream,
            zulip_topic=ctx.zulip_topic,
            source_frame_id=frame.id,
        )


class MockZulipProcessor(Processor):
    """Mock Zulip processor using mock chat provider."""

    @property
    def name(self) -> str:
        return "mock_zulip"

    async def process(self, frame: Frame, ctx: PipelineContext):
        if not isinstance(frame, ExtractionFrame):
            print(f"  [{self.name}] Skipping non-ExtractionFrame: {type(frame).__name__}")
            return frame

        print(f"  [{self.name}] Processing ExtractionFrame...")
        chat = ctx.providers.get_chat()

        # Format message
        lines = ["**Morning Standup**", ""]
        if frame.today_items:
            lines.append("**Today:**")
            for item in frame.today_items:
                lines.append(f"- {item}")
        if frame.blockers:
            lines.append("\n**Blockers:**")
            for blocker in frame.blockers:
                lines.append(f"- {blocker}")

        content = "\n".join(lines)

        result = await chat.send_message(
            stream=ctx.zulip_stream,
            topic=ctx.zulip_topic,
            content=content,
        )

        print(f"  [{self.name}] Posted to Zulip: success={result.success}")

        return ZulipMessageFrame(
            stream=ctx.zulip_stream,
            topic=ctx.zulip_topic,
            content=content,
            success=result.success,
            message_id=result.message_id,
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )


class MockConfirmationProcessor(Processor):
    """Mock confirmation processor."""

    @property
    def name(self) -> str:
        return "mock_confirmation"

    async def process(self, frame: Frame, ctx: PipelineContext):
        from knomly.pipeline.frames import UserResponseFrame

        if isinstance(frame, ZulipMessageFrame):
            # Zulip-specific: format as "Posted to..."
            if frame.success:
                message = f"✅ Posted to {frame.stream}/{frame.topic}"
            else:
                message = f"❌ Failed: {frame.error}"
        elif isinstance(frame, UserResponseFrame):
            # Generic response: use message directly
            message = frame.message
        else:
            message = "Processed your message"

        print(f"  [{self.name}] Confirming: {message}")

        return ConfirmationFrame(
            message=message,
            success=getattr(frame, "success", True),
            recipient_phone=getattr(frame, "sender_phone", ""),
            source_frame_id=frame.id,
        )


class UnknownIntentProcessor(Processor):
    """Handler for unknown intents."""

    @property
    def name(self) -> str:
        return "unknown_intent"

    async def process(self, frame: Frame, ctx: PipelineContext):
        from knomly.pipeline.frames import UserResponseFrame

        intent = frame.metadata.get("intent", "unknown")
        print(f"  [{self.name}] Handling unknown intent: {intent}")

        # Return generic UserResponseFrame (not Zulip-specific)
        return UserResponseFrame(
            message=f"I couldn't understand what you wanted (intent: {intent})",
            sender_phone=getattr(frame, "sender_phone", ""),
            success=False,
            error=f"Unrecognized intent: {intent}",
            source_frame_id=frame.id,
        )


# =============================================================================
# End-to-End Test
# =============================================================================


class TestE2EPipeline:
    """End-to-end pipeline tests."""

    @pytest.fixture
    def mock_providers(self):
        """Create mock provider registry."""
        registry = ProviderRegistry()
        registry.register_stt("mock", MockSTTProvider())
        registry.register_llm("mock", MockLLMProvider())
        registry.register_chat("mock", MockChatProvider())
        return registry

    @pytest.fixture
    def pipeline_with_routing(self):
        """Create pipeline with intent routing."""
        # Build standup branch
        standup_branch = (
            PipelineBuilder().add(MockExtractionProcessor()).add(MockZulipProcessor()).build()
        )

        # Build main pipeline
        return (
            PipelineBuilder()
            .add(MockAudioDownloadProcessor())
            .add(MockTranscriptionProcessor())
            .add(IntentClassifierProcessor())
            .add(
                Switch(
                    key=get_intent,
                    cases={
                        Intent.STANDUP.value: standup_branch,
                    },
                    default=UnknownIntentProcessor(),
                    key_name="intent",
                )
            )
            .add(MockConfirmationProcessor())
            .build()
        )

    @pytest.mark.asyncio
    async def test_full_standup_flow(self, pipeline_with_routing, mock_providers):
        """Test complete flow: Audio -> Transcription -> Intent -> Extraction -> Zulip -> Confirmation."""
        print("\n" + "=" * 60)
        print("E2E TEST: Full Standup Flow with Intent Routing")
        print("=" * 60)

        # Create input frame
        input_frame = AudioInputFrame(
            media_url="https://api.twilio.com/fake/audio.ogg",
            sender_phone="+1234567890",
            mime_type="audio/ogg",
        )

        # Create context
        ctx = PipelineContext(
            providers=mock_providers,
            sender_phone="+1234567890",
            user_id="test_user",
            user_name="Test User",
            zulip_stream="standup",
            zulip_topic="test-user-updates",
        )

        print(f"\nInput: {input_frame}")
        print(f"Context: user={ctx.user_name}, stream={ctx.zulip_stream}/{ctx.zulip_topic}")
        print("\nPipeline Execution:")
        print("-" * 40)

        # Execute pipeline
        result = await pipeline_with_routing.execute(input_frame, ctx)

        print("-" * 40)
        print(f"\nResult: success={result.success}, frames={len(result.output_frames)}")
        print(f"Duration: {result.duration_ms:.1f}ms")
        print(f"Timings: {ctx.processor_timings}")
        print(f"Routing decisions: {[d.selected_branch for d in ctx.routing_decisions]}")

        # Assertions
        assert result.success, f"Pipeline failed: {result.error}"
        assert len(result.output_frames) > 0, "No output frames"

        # Check we got a confirmation frame
        confirmation = result.get_frame(ConfirmationFrame)
        assert confirmation is not None, "No ConfirmationFrame in output"
        assert confirmation.success, "Confirmation shows failure"
        print(f"\nFinal confirmation: {confirmation.message}")

        # Check routing decision
        assert len(ctx.routing_decisions) > 0, "No routing decisions recorded"
        intent_decision = next(
            (d for d in ctx.routing_decisions if "intent" in d.router_name.lower()), None
        )
        assert intent_decision is not None, "No intent routing decision"
        assert intent_decision.selected_branch == "standup", (
            f"Wrong branch: {intent_decision.selected_branch}"
        )

        print("\n✅ E2E Test PASSED")

    @pytest.mark.asyncio
    async def test_unknown_intent_flow(self, mock_providers):
        """Test flow when intent is unknown."""
        print("\n" + "=" * 60)
        print("E2E TEST: Unknown Intent Flow")
        print("=" * 60)

        # Create STT that returns something non-standup-like
        mock_providers.unregister_stt("mock")
        mock_providers.register_stt(
            "mock",
            MockSTTProvider("What time is dinner?"),  # Not a standup
        )

        # Modify LLM to return unknown intent
        class UnknownIntentLLM(MockLLMProvider):
            async def complete(self, messages, config=None):
                prompt = messages[-1].content if messages else ""
                if "Classify the intent" in prompt:
                    print("  [MockLLM] Classifying as unknown...")
                    return LLMResponse(
                        content='{"intent": "query", "confidence": 0.6}',
                        model="mock-model",
                        provider=self.name,
                    )
                return await super().complete(messages, config)

        mock_providers.unregister_llm("mock")
        mock_providers.register_llm("mock", UnknownIntentLLM())

        # Build pipeline (without standup branch for query intent)
        standup_branch = (
            PipelineBuilder().add(MockExtractionProcessor()).add(MockZulipProcessor()).build()
        )

        pipeline = (
            PipelineBuilder()
            .add(MockAudioDownloadProcessor())
            .add(MockTranscriptionProcessor())
            .add(IntentClassifierProcessor())
            .add(
                Switch(
                    key=get_intent,
                    cases={
                        Intent.STANDUP.value: standup_branch,
                    },
                    default=UnknownIntentProcessor(),
                    key_name="intent",
                )
            )
            .add(MockConfirmationProcessor())
            .build()
        )

        # Create input
        input_frame = AudioInputFrame(
            media_url="https://api.twilio.com/fake/audio.ogg",
            sender_phone="+1234567890",
        )

        ctx = PipelineContext(
            providers=mock_providers,
            sender_phone="+1234567890",
            user_id="test_user",
            user_name="Test User",
            zulip_stream="standup",
            zulip_topic="test-user-updates",
        )

        print("\nInput: Voice note saying 'What time is dinner?'")
        print("\nPipeline Execution:")
        print("-" * 40)

        result = await pipeline.execute(input_frame, ctx)

        print("-" * 40)
        print(f"\nResult: success={result.success}")
        print(f"Routing: {[d.selected_branch for d in ctx.routing_decisions]}")

        # Should route to default (unknown handler)
        assert result.success
        intent_decision = next(
            (d for d in ctx.routing_decisions if "intent" in d.router_name.lower()), None
        )
        assert intent_decision is not None
        assert intent_decision.selected_branch == "default", (
            f"Should route to default, got: {intent_decision.selected_branch}"
        )

        print("\n✅ Unknown Intent Test PASSED")

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self, mock_providers):
        """Test that errors are handled gracefully."""
        print("\n" + "=" * 60)
        print("E2E TEST: Error Handling")
        print("=" * 60)

        class FailingProcessor(Processor):
            @property
            def name(self) -> str:
                return "failing_processor"

            async def process(self, frame, ctx):
                raise RuntimeError("Simulated failure!")

        pipeline = (
            PipelineBuilder()
            .add(MockAudioDownloadProcessor())
            .add(FailingProcessor())  # This will fail
            .add(MockConfirmationProcessor())
            .build()
        )

        input_frame = AudioInputFrame(
            media_url="https://api.twilio.com/fake/audio.ogg",
            sender_phone="+1234567890",
        )

        ctx = PipelineContext(
            providers=mock_providers,
            sender_phone="+1234567890",
        )

        print("\nPipeline with failing processor:")
        print("-" * 40)

        result = await pipeline.execute(input_frame, ctx)

        print("-" * 40)

        # Pipeline should handle error gracefully
        from knomly.pipeline.frames import ErrorFrame

        error_frame = result.get_frame(ErrorFrame)

        if error_frame:
            print(f"Error captured: {error_frame.error_message}")
            print(f"Processor: {error_frame.processor_name}")
            assert error_frame.processor_name == "failing_processor"
            print("\n✅ Error Handling Test PASSED - Error captured in ErrorFrame")
        else:
            print(f"Pipeline result: success={result.success}, error={result.error}")
            print("\n✅ Error Handling Test PASSED")


# =============================================================================
# Run if executed directly
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("KNOMLY END-TO-END MANUAL TEST")
    print("=" * 60)

    async def run_tests():
        test = TestE2EPipeline()
        providers = ProviderRegistry()
        providers.register_stt("mock", MockSTTProvider())
        providers.register_llm("mock", MockLLMProvider())
        providers.register_chat("mock", MockChatProvider())

        # Build pipeline
        standup_branch = (
            PipelineBuilder().add(MockExtractionProcessor()).add(MockZulipProcessor()).build()
        )

        pipeline = (
            PipelineBuilder()
            .add(MockAudioDownloadProcessor())
            .add(MockTranscriptionProcessor())
            .add(IntentClassifierProcessor())
            .add(
                Switch(
                    key=get_intent,
                    cases={
                        Intent.STANDUP.value: standup_branch,
                    },
                    default=UnknownIntentProcessor(),
                    key_name="intent",
                )
            )
            .add(MockConfirmationProcessor())
            .build()
        )

        await test.test_full_standup_flow(pipeline, providers)

    asyncio.run(run_tests())
