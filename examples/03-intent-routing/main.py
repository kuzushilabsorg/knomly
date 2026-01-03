"""
Intent Routing Example

This example demonstrates intent-based routing:
1. Classify user intent
2. Route to appropriate handler
3. Use Switch for branching

Run: python -m examples.03-intent-routing.main
"""
import asyncio
from dataclasses import dataclass
from enum import Enum

from knomly import (
    Pipeline,
    PipelineBuilder,
    PipelineContext,
    Processor,
    Switch,
)
from knomly.pipeline.frames import Frame


# =============================================================================
# Intent Enum
# =============================================================================


class Intent(Enum):
    GREETING = "greeting"
    QUESTION = "question"
    COMMAND = "command"
    UNKNOWN = "unknown"


# =============================================================================
# Custom Frames
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class UserMessageFrame(Frame):
    """User message with intent."""
    text: str = ""
    intent: str = "unknown"
    confidence: float = 0.0


@dataclass(frozen=True, kw_only=True, slots=True)
class ResponseFrame(Frame):
    """Bot response."""
    response: str = ""
    intent_handled: str = ""


# =============================================================================
# Intent Classifier
# =============================================================================


class IntentClassifier(Processor):
    """Simple keyword-based intent classifier."""

    GREETING_KEYWORDS = {"hello", "hi", "hey", "greetings", "good morning"}
    QUESTION_KEYWORDS = {"what", "why", "how", "when", "where", "who", "?"}
    COMMAND_KEYWORDS = {"do", "run", "execute", "start", "stop", "create"}

    @property
    def name(self) -> str:
        return "intent_classifier"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UserMessageFrame):
            return frame

        text_lower = frame.text.lower()
        words = set(text_lower.split())

        # Simple keyword matching
        if words & self.GREETING_KEYWORDS:
            intent = Intent.GREETING.value
            confidence = 0.9
        elif words & self.QUESTION_KEYWORDS or "?" in frame.text:
            intent = Intent.QUESTION.value
            confidence = 0.85
        elif words & self.COMMAND_KEYWORDS:
            intent = Intent.COMMAND.value
            confidence = 0.8
        else:
            intent = Intent.UNKNOWN.value
            confidence = 0.5

        return frame.derive(
            intent=intent,
            confidence=confidence,
            metadata={**frame.metadata, "intent": intent, "confidence": confidence},
        )


# =============================================================================
# Intent Handlers
# =============================================================================


class GreetingHandler(Processor):
    """Handles greeting intents."""

    @property
    def name(self) -> str:
        return "greeting_handler"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UserMessageFrame):
            return frame

        return ResponseFrame(
            response=f"Hello! Nice to meet you. You said: '{frame.text}'",
            intent_handled="greeting",
            source_frame_id=frame.id,
        )


class QuestionHandler(Processor):
    """Handles question intents."""

    @property
    def name(self) -> str:
        return "question_handler"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UserMessageFrame):
            return frame

        return ResponseFrame(
            response=f"That's a great question! Let me think about: '{frame.text}'",
            intent_handled="question",
            source_frame_id=frame.id,
        )


class CommandHandler(Processor):
    """Handles command intents."""

    @property
    def name(self) -> str:
        return "command_handler"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UserMessageFrame):
            return frame

        return ResponseFrame(
            response=f"Executing command: '{frame.text}'",
            intent_handled="command",
            source_frame_id=frame.id,
        )


class UnknownHandler(Processor):
    """Handles unknown intents."""

    @property
    def name(self) -> str:
        return "unknown_handler"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UserMessageFrame):
            return frame

        return ResponseFrame(
            response="I'm not sure what you mean. Can you rephrase?",
            intent_handled="unknown",
            source_frame_id=frame.id,
        )


# =============================================================================
# Helper Functions
# =============================================================================


def get_intent(frame: Frame, ctx: PipelineContext) -> str:
    """Extract intent from frame metadata."""
    return frame.metadata.get("intent", "unknown")


# =============================================================================
# Main
# =============================================================================


async def main():
    # Build pipeline with intent routing
    pipeline = (
        PipelineBuilder()
        .add(IntentClassifier())
        .add(Switch(
            key=get_intent,
            cases={
                Intent.GREETING.value: GreetingHandler(),
                Intent.QUESTION.value: QuestionHandler(),
                Intent.COMMAND.value: CommandHandler(),
            },
            default=UnknownHandler(),
            key_name="intent",
        ))
        .build()
    )

    print("Intent Routing Pipeline")
    print("=" * 50)
    print()

    # Test messages
    test_messages = [
        "Hello, how are you?",
        "What is the weather like today?",
        "Run the daily report",
        "asdfghjkl random text",
    ]

    for message in test_messages:
        frame = UserMessageFrame(text=message)
        result = await pipeline.execute(frame)

        response = result.get_frame(ResponseFrame)
        if response:
            print(f"User: {message}")
            print(f"Intent: {response.intent_handled}")
            print(f"Bot: {response.response}")
            print()


if __name__ == "__main__":
    asyncio.run(main())
