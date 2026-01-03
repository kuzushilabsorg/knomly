"""
Voice Standup Example

This example demonstrates a complete voice-to-chat pipeline:
1. Receive audio from messaging platform
2. Transcribe using STT provider
3. Extract standup items using LLM
4. Post to chat platform
5. Send confirmation

This is a simplified version of the full Knomly standup application.

Requirements:
    pip install knomly[full]

Environment variables:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN
    TWILIO_WHATSAPP_NUMBER
    OPENAI_API_KEY
    ZULIP_API_KEY
    ZULIP_EMAIL
    ZULIP_SITE
"""

import asyncio
import os
from dataclasses import dataclass

from knomly import (
    Pipeline,
    PipelineBuilder,
    PipelineContext,
    Processor,
    TwilioTransport,
    register_transport,
)
from knomly.pipeline.frames import Frame

# =============================================================================
# Configuration
# =============================================================================


@dataclass
class Config:
    """Application configuration."""

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_from_number: str = ""
    openai_api_key: str = ""
    zulip_api_key: str = ""
    zulip_email: str = ""
    zulip_site: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        """Load configuration from environment variables."""
        return cls(
            twilio_account_sid=os.getenv("TWILIO_ACCOUNT_SID", ""),
            twilio_auth_token=os.getenv("TWILIO_AUTH_TOKEN", ""),
            twilio_from_number=os.getenv("TWILIO_WHATSAPP_NUMBER", ""),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            zulip_api_key=os.getenv("ZULIP_API_KEY", ""),
            zulip_email=os.getenv("ZULIP_EMAIL", ""),
            zulip_site=os.getenv("ZULIP_SITE", ""),
        )


# =============================================================================
# Mock Processors (replace with real implementations)
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class AudioFrame(Frame):
    """Audio input frame."""

    audio_url: str = ""
    audio_bytes: bytes | None = None
    sender_phone: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class TranscriptionFrame(Frame):
    """Transcribed text frame."""

    text: str = ""
    language: str = "en"
    sender_phone: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class StandupFrame(Frame):
    """Extracted standup items."""

    today_items: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    summary: str = ""
    sender_phone: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class PostedFrame(Frame):
    """Posted to chat."""

    message_id: int = 0
    channel: str = ""
    success: bool = True
    sender_phone: str = ""


class MockTranscriptionProcessor(Processor):
    """Mock transcription - replace with real STT provider."""

    @property
    def name(self) -> str:
        return "transcription"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, AudioFrame):
            return frame

        # In production, this would call an STT API
        print(f"  [Transcribing audio from {frame.audio_url}]")

        return TranscriptionFrame(
            text="Today I'm working on the new feature. No blockers.",
            language="en",
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )


class MockExtractionProcessor(Processor):
    """Mock extraction - replace with real LLM provider."""

    @property
    def name(self) -> str:
        return "extraction"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, TranscriptionFrame):
            return frame

        # In production, this would call an LLM API
        print(f"  [Extracting standup from: '{frame.text[:50]}...']")

        return StandupFrame(
            today_items=("Working on new feature",),
            blockers=(),
            summary="Feature development in progress",
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )


class MockZulipProcessor(Processor):
    """Mock Zulip posting - replace with real chat provider."""

    @property
    def name(self) -> str:
        return "zulip"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, StandupFrame):
            return frame

        # In production, this would call Zulip API
        print(f"  [Posting to Zulip: {len(frame.today_items)} items]")

        return PostedFrame(
            message_id=12345,
            channel="standup",
            success=True,
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )


class MockConfirmationProcessor(Processor):
    """Mock confirmation - replace with real transport."""

    @property
    def name(self) -> str:
        return "confirmation"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, PostedFrame):
            return frame

        # In production, this would use the transport registry
        print(f"  [Sending confirmation to {frame.sender_phone}]")

        return frame


# =============================================================================
# Pipeline Factory
# =============================================================================


def create_standup_pipeline(config: Config) -> Pipeline:
    """
    Create the standup processing pipeline.

    Flow:
        AudioFrame
            -> TranscriptionProcessor (STT)
            -> ExtractionProcessor (LLM)
            -> ZulipProcessor (Chat)
            -> ConfirmationProcessor (Transport)
    """
    return (
        PipelineBuilder()
        .add(MockTranscriptionProcessor())
        .add(MockExtractionProcessor())
        .add(MockZulipProcessor())
        .add(MockConfirmationProcessor())
        .build()
    )


# =============================================================================
# Main
# =============================================================================


async def main():
    print("Voice Standup Pipeline Example")
    print("=" * 50)
    print()
    print("This example uses mock processors.")
    print("In production, replace with real provider implementations.")
    print()

    # Load configuration
    config = Config.from_env()

    # Register transport (would be used by real ConfirmationProcessor)
    if config.twilio_account_sid:
        transport = TwilioTransport(
            account_sid=config.twilio_account_sid,
            auth_token=config.twilio_auth_token,
            from_number=config.twilio_from_number,
        )
        register_transport(transport)
        print("Registered Twilio transport")
    else:
        print("Twilio credentials not configured (using mocks)")

    print()

    # Create pipeline
    pipeline = create_standup_pipeline(config)
    print(f"Pipeline: {pipeline.processor_names}")
    print()

    # Simulate processing a voice note
    print("Processing simulated voice note...")
    print("-" * 40)

    initial_frame = AudioFrame(
        audio_url="https://api.twilio.com/media/audio123.ogg",
        sender_phone="+1234567890",
    )

    ctx = PipelineContext(
        sender_phone=initial_frame.sender_phone,
        message_type="audio",
        channel_id="twilio",
    )

    result = await pipeline.execute(initial_frame, ctx)

    print("-" * 40)
    print()
    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print(f"Processors: {list(ctx.processor_timings.keys())}")

    # Get final output
    posted = result.get_frame(PostedFrame)
    if posted:
        print()
        print("Result:")
        print(f"  - Posted to: {posted.channel}")
        print(f"  - Message ID: {posted.message_id}")
        print(f"  - Confirmation sent to: {posted.sender_phone}")


if __name__ == "__main__":
    asyncio.run(main())
