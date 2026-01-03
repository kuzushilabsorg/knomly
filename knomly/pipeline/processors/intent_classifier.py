"""
Intent Classifier Processor for Knomly.

Classifies the intent of transcribed voice messages to route
to appropriate downstream processors.
"""
from __future__ import annotations

import json
import logging
from enum import Enum
from typing import TYPE_CHECKING

from ..processor import Processor

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import Frame

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """Recognized intents for voice messages."""

    STANDUP = "standup"
    TASK = "task"
    REMINDER = "reminder"
    QUERY = "query"
    UNKNOWN = "unknown"


# Classification prompt - kept minimal for fast inference
# Note: Double braces {{ }} are escaped for Python .format()
CLASSIFICATION_PROMPT = """Classify the intent of this voice message transcript into exactly one category.

Categories:
- standup: Daily standup update with work items, blockers, or progress
- task: Request to create/manage a task or todo item
- reminder: Request to set a reminder or alarm
- query: Question or information request
- unknown: Cannot determine intent

Return JSON only: {{"intent": "<category>", "confidence": 0.0-1.0}}

Transcript: {text}"""


class IntentClassifierProcessor(Processor):
    """
    Classifies voice message intent using LLM.

    Input: TranscriptionFrame
    Output: TranscriptionFrame with intent in metadata

    The classified intent is stored in frame.metadata["intent"]
    and can be used by downstream Switch routers.

    Example:
        # In pipeline builder
        builder.add(IntentClassifierProcessor())
        builder.add(
            Switch(
                key=lambda f, ctx: f.metadata.get("intent", "unknown"),
                cases={
                    "standup": ExtractionProcessor(),
                    "task": TaskProcessor(),
                },
                default=UnknownIntentHandler(),
            )
        )
    """

    def __init__(
        self,
        provider_name: str | None = None,
        confidence_threshold: float = 0.6,
    ):
        """
        Args:
            provider_name: Specific LLM provider to use (or default)
            confidence_threshold: Minimum confidence to accept classification
        """
        self._provider_name = provider_name
        self._confidence_threshold = confidence_threshold

    @property
    def name(self) -> str:
        return "intent_classifier"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | None":
        from ..frames import TranscriptionFrame

        if not isinstance(frame, TranscriptionFrame):
            return frame

        if not frame.text.strip():
            logger.warning("Empty transcription, defaulting to unknown intent")
            return frame.with_metadata(intent=Intent.UNKNOWN.value, intent_confidence=0.0)

        if ctx.providers is None:
            raise RuntimeError("No providers configured in context")

        llm = ctx.providers.get_llm(self._provider_name)

        logger.info(f"Classifying intent for {len(frame.text)} chars with {llm.name}")

        try:
            # Use fast LLM call for classification
            from knomly.providers.llm import LLMConfig, Message

            prompt = CLASSIFICATION_PROMPT.format(text=frame.text[:500])  # Truncate for speed

            response = await llm.complete(
                messages=[Message.user(prompt)],
                config=LLMConfig(
                    temperature=0.1,  # Low temperature for classification
                    max_tokens=50,  # Short response expected
                ),
            )

            # Parse classification result
            intent, confidence = self._parse_classification(response.content)

            # Apply confidence threshold
            if confidence < self._confidence_threshold:
                logger.info(
                    f"Intent '{intent}' below confidence threshold "
                    f"({confidence:.2f} < {self._confidence_threshold}), using unknown"
                )
                intent = Intent.UNKNOWN.value

            logger.info(f"Classified intent: {intent} (confidence={confidence:.2f})")

            return frame.with_metadata(
                intent=intent,
                intent_confidence=confidence,
            )

        except Exception as e:
            logger.error(f"Intent classification failed: {e}", exc_info=True)
            # On failure, default to standup (backwards compatibility for Phase 1)
            return frame.with_metadata(
                intent=Intent.STANDUP.value,
                intent_confidence=0.0,
                intent_error=str(e),
            )

    def _parse_classification(self, content: str) -> tuple[str, float]:
        """Parse LLM response to extract intent and confidence."""
        # Clean markdown
        text = content.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        try:
            result = json.loads(text.strip())
            intent = result.get("intent", "unknown").lower()
            confidence = float(result.get("confidence", 0.5))

            # Validate intent is known
            valid_intents = {e.value for e in Intent}
            if intent not in valid_intents:
                logger.warning(f"Unknown intent '{intent}', defaulting to unknown")
                intent = Intent.UNKNOWN.value

            return intent, min(max(confidence, 0.0), 1.0)  # Clamp to [0, 1]

        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Failed to parse classification: {e}, content={content[:100]}")
            return Intent.UNKNOWN.value, 0.0


# Convenience function for key extraction in Switch router
def get_intent(frame: "Frame", ctx: "PipelineContext") -> str:
    """
    Extract intent from frame metadata for use with Switch router.

    Example:
        Switch(
            key=get_intent,
            cases={"standup": ..., "task": ...},
            default=...,
        )
    """
    return frame.metadata.get("intent", Intent.UNKNOWN.value)


__all__ = [
    "Intent",
    "IntentClassifierProcessor",
    "get_intent",
]
