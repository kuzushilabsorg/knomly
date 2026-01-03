"""
Pipeline Builder Factory for Knomly.

Creates pre-configured pipelines for different use cases.
Uses routing primitives for intent-based processing.

Architecture (v1.5 + v2 Agent Layer):
    Audio → Transcription → Intent Classification → Context Enrichment → Router
                                                                            |
                                          +--------------------------------+--------------------------------+
                                          |                                |                                |
                                      [standup]                         [task]                         [unknown]
                                          |                                |                                |
                                    Extraction                     AgentBridgeProcessor               Helpful Response
                                          |                     (v2 Agent with Tools)                       |
                                       Zulip                              |                                 |
                                          |                               |                                 |
                                          +---------------+---------------+---------------------------------+
                                                          |
                                                    Confirmation
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .executor import Pipeline, PipelineBuilder
from .processor import Processor
from .routing import Switch
from .processors import (
    AgentBridgeProcessor,
    AudioDownloadProcessor,
    ConfirmationProcessor,
    ContextEnrichmentProcessor,
    ExtractionProcessor,
    Intent,
    IntentClassifierProcessor,
    TranscriptionProcessor,
    ZulipProcessor,
    create_task_agent_bridge,
    get_intent,
)

if TYPE_CHECKING:
    from knomly.config.schemas import AppSettings
    from knomly.integrations.plane import PlaneClient
    from knomly.integrations.plane.cache import PlaneEntityCache
    from knomly.providers.llm import LLMProvider
    from .context import PipelineContext
    from .frames import Frame

logger = logging.getLogger(__name__)


# =============================================================================
# Unknown Intent Handler
# =============================================================================


class UnknownIntentProcessor(Processor):
    """
    Handles messages with unknown or unrecognized intent.

    Instead of forcing into standup extraction, sends a helpful
    response back to the user via a generic UserResponseFrame.

    This is transport-agnostic - it doesn't assume Zulip, Slack, or
    any specific integration. The downstream ConfirmationProcessor
    handles delivery via the appropriate channel.
    """

    @property
    def name(self) -> str:
        return "unknown_intent"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | None":
        from .frames import UserResponseFrame

        # Extract intent info from metadata
        intent = frame.metadata.get("intent", "unknown")
        confidence = frame.metadata.get("intent_confidence", 0.0)

        logger.info(
            f"Unknown intent handler: intent={intent}, confidence={confidence:.2f}"
        )

        # Return a generic UserResponseFrame (not Zulip-specific)
        # ConfirmationProcessor will send this back via the appropriate channel
        return UserResponseFrame(
            message=(
                "I couldn't understand what you wanted to do. "
                "Try sending a standup update like:\n"
                '"Today I\'m working on X. Blocker: Y."'
            ),
            sender_phone=getattr(frame, "sender_phone", "") or "",
            success=False,
            error=f"Unrecognized intent: {intent}",
            source_frame_id=frame.id,
        )


# =============================================================================
# Pipeline Factory
# =============================================================================


class PipelineFactory:
    """
    Factory for creating pre-configured pipelines.

    Creates pipelines for:
    - Voice message processing with intent-based routing
    - Standup: Voice note -> Zulip post
    - Task: Voice note -> Agent -> Plane task creation (v2)
    - More intent handlers can be added as the system evolves

    Architecture (v1.5 + v2 Agent):
        Audio -> Transcription -> Intent -> Context Enrichment -> Router
                                                                    |
                                      +-----------------------------+-----------------------------+
                                      |                             |                             |
                                  [standup]                      [task]                      [unknown]
                                      |                             |                             |
                               Extraction              AgentBridgeProcessor              Helpful Response
                                      |               (Tools + LLM reasoning)                    |
                                   Zulip                           |                             |
                                      |                            |                             |
                                      +----------------------------+-----------------------------+
                                                                   |
                                                             Confirmation
    """

    def __init__(
        self,
        settings: "AppSettings",
        *,
        plane_client: "PlaneClient | None" = None,
        plane_cache: "PlaneEntityCache | None" = None,
        llm_provider: "LLMProvider | None" = None,
    ):
        """
        Initialize pipeline factory with settings.

        Args:
            settings: Application settings with API keys and config
            plane_client: Plane API client for task management (optional)
            plane_cache: Plane entity cache for name resolution (optional)
            llm_provider: LLM provider for agent decisions (optional)

        Note:
            If plane_client, plane_cache, and llm_provider are not provided,
            TASK intent will fall through to UnknownIntentProcessor.
        """
        self._settings = settings
        self._plane_client = plane_client
        self._plane_cache = plane_cache
        self._llm_provider = llm_provider

    def create_voice_pipeline(self) -> Pipeline:
        """
        Create the main voice message pipeline with intent routing.

        This is the recommended pipeline for production use.
        Routes messages based on detected intent.

        Flow:
            1. Audio Download (Twilio)
            2. Transcription (Deepgram/Gemini)
            3. Intent Classification
            4. Context Enrichment (Plane projects, users)
            5. Intent Router:
               - STANDUP → Extraction → Zulip
               - TASK → AgentBridgeProcessor (v2 agent with tools)
               - UNKNOWN → Helpful response
            6. Confirmation (Twilio WhatsApp)

        Returns:
            Configured Pipeline instance
        """
        builder = PipelineBuilder()

        # 1. Download audio from Twilio URL
        builder.add(
            AudioDownloadProcessor(
                twilio_account_sid=self._settings.twilio_account_sid,
                twilio_auth_token=self._settings.twilio_auth_token,
            )
        )

        # 2. Transcribe audio to text
        builder.add(TranscriptionProcessor())

        # 3. Classify intent
        builder.add(IntentClassifierProcessor())

        # 4. Enrich with entity context (Plane projects, users)
        # This enables context-aware extraction and agent decisions
        if self._plane_cache:
            builder.add(ContextEnrichmentProcessor(plane_cache=self._plane_cache))

        # 5. Route based on intent
        builder.add(
            Switch(
                key=get_intent,
                cases=self._build_intent_cases(),
                default=UnknownIntentProcessor(),
                key_name="intent",
            )
        )

        # 6. Send confirmation back to user
        builder.add(
            ConfirmationProcessor(
                twilio_account_sid=self._settings.twilio_account_sid,
                twilio_auth_token=self._settings.twilio_auth_token,
                twilio_from_number=self._settings.twilio_whatsapp_number,
            )
        )

        return builder.build()

    def _build_intent_cases(self) -> dict[str, Pipeline | Processor]:
        """
        Build the intent routing cases.

        Returns:
            Dict mapping intent values to processors/pipelines
        """
        cases: dict[str, Pipeline | Processor] = {
            Intent.STANDUP.value: self._create_standup_branch(),
        }

        # Add TASK intent if agent dependencies are available
        task_processor = self._create_task_processor()
        if task_processor:
            cases[Intent.TASK.value] = task_processor
            logger.info("[pipeline_factory] TASK intent routing enabled (v2 agent)")
        else:
            logger.warning(
                "[pipeline_factory] TASK intent routing disabled - "
                "missing plane_client, plane_cache, or llm_provider"
            )

        return cases

    def _create_task_processor(self) -> AgentBridgeProcessor | None:
        """
        Create the AgentBridgeProcessor for TASK intent.

        Returns:
            AgentBridgeProcessor if all dependencies available, else None
        """
        if not (self._plane_client and self._plane_cache and self._llm_provider):
            return None

        return create_task_agent_bridge(
            plane_client=self._plane_client,
            plane_cache=self._plane_cache,
            llm_provider=self._llm_provider,
            max_iterations=5,
        )

    def _create_standup_branch(self) -> Pipeline:
        """Create the standup processing branch."""
        return (
            PipelineBuilder()
            .add(ExtractionProcessor())
            .add(ZulipProcessor())
            .build()
        )

    def create_standup_pipeline(self) -> Pipeline:
        """
        Create a simple linear standup pipeline (legacy).

        For backward compatibility. New code should use create_voice_pipeline().

        Flow:
        AudioInputFrame
            -> AudioDownloadProcessor
            -> TranscriptionProcessor
            -> ExtractionProcessor (assumes standup)
            -> ZulipProcessor
            -> ConfirmationProcessor
            -> ConfirmationFrame

        Returns:
            Configured Pipeline instance
        """
        builder = PipelineBuilder()

        builder.add(
            AudioDownloadProcessor(
                twilio_account_sid=self._settings.twilio_account_sid,
                twilio_auth_token=self._settings.twilio_auth_token,
            )
        )
        builder.add(TranscriptionProcessor())
        builder.add(ExtractionProcessor())
        builder.add(ZulipProcessor())
        builder.add(
            ConfirmationProcessor(
                twilio_account_sid=self._settings.twilio_account_sid,
                twilio_auth_token=self._settings.twilio_auth_token,
                twilio_from_number=self._settings.twilio_whatsapp_number,
            )
        )

        return builder.build()

    def create_transcription_only_pipeline(self) -> Pipeline:
        """
        Create a transcription-only pipeline (for testing).

        Flow:
        AudioInputFrame
            -> AudioDownloadProcessor
            -> TranscriptionProcessor
            -> TranscriptionFrame

        Returns:
            Configured Pipeline instance
        """
        return (
            PipelineBuilder()
            .add(
                AudioDownloadProcessor(
                    twilio_account_sid=self._settings.twilio_account_sid,
                    twilio_auth_token=self._settings.twilio_auth_token,
                )
            )
            .add(TranscriptionProcessor())
            .build()
        )


# =============================================================================
# Convenience Functions
# =============================================================================


def create_voice_pipeline(
    settings: "AppSettings",
    *,
    plane_client: "PlaneClient | None" = None,
    plane_cache: "PlaneEntityCache | None" = None,
    llm_provider: "LLMProvider | None" = None,
) -> Pipeline:
    """
    Create the main voice message pipeline with intent routing.

    This is the recommended function for production use.

    Args:
        settings: Application settings
        plane_client: Plane API client for task management (optional)
        plane_cache: Plane entity cache for name resolution (optional)
        llm_provider: LLM provider for agent decisions (optional)

    Returns:
        Configured voice Pipeline with intent routing

    Note:
        If Plane dependencies are provided, TASK intent will be routed
        to the v2 agent layer for intelligent task management.
    """
    factory = PipelineFactory(
        settings,
        plane_client=plane_client,
        plane_cache=plane_cache,
        llm_provider=llm_provider,
    )
    return factory.create_voice_pipeline()


def create_standup_pipeline(settings: "AppSettings") -> Pipeline:
    """
    Create a simple linear standup pipeline (legacy).

    For backward compatibility. New code should use create_voice_pipeline().

    Args:
        settings: Application settings

    Returns:
        Configured standup Pipeline
    """
    factory = PipelineFactory(settings)
    return factory.create_standup_pipeline()


def create_default_pipeline(
    settings: "AppSettings",
    *,
    plane_client: "PlaneClient | None" = None,
    plane_cache: "PlaneEntityCache | None" = None,
    llm_provider: "LLMProvider | None" = None,
) -> Pipeline:
    """
    Create the default pipeline for voice message processing.

    Now uses intent-based routing for better handling of different
    message types.

    Args:
        settings: Application settings
        plane_client: Plane API client for task management (optional)
        plane_cache: Plane entity cache for name resolution (optional)
        llm_provider: LLM provider for agent decisions (optional)

    Returns:
        Configured Pipeline with intent routing
    """
    return create_voice_pipeline(
        settings,
        plane_client=plane_client,
        plane_cache=plane_cache,
        llm_provider=llm_provider,
    )
