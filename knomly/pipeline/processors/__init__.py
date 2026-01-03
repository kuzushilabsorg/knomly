"""
Knomly Pipeline Processors

Processors are single-responsibility frame transformers.
See ADR-001 for design decisions.
"""

from .agent_bridge import AgentBridgeProcessor, create_task_agent_bridge
from .audio_download import AudioDownloadProcessor
from .confirmation import ConfirmationProcessor
from .context_enrichment import (
    ContextEnrichmentProcessor,
    extract_plane_context,
    get_project_prompt_section,
    get_user_prompt_section,
)
from .extraction import ExtractionProcessor
from .integrations import PlaneProcessor, TaskCreatorProcessor
from .intent_classifier import Intent, IntentClassifierProcessor, get_intent
from .transcription import TranscriptionProcessor
from .zulip import ZulipProcessor

__all__ = [
    # Agent bridge (v2)
    "AgentBridgeProcessor",
    # Core processors
    "AudioDownloadProcessor",
    "ConfirmationProcessor",
    # Context enrichment (v1.5)
    "ContextEnrichmentProcessor",
    "ExtractionProcessor",
    # Intent utilities
    "Intent",
    "IntentClassifierProcessor",
    # Integration processors (accept generic frames)
    "PlaneProcessor",
    "TaskCreatorProcessor",
    "TranscriptionProcessor",
    "ZulipProcessor",
    "create_task_agent_bridge",
    "extract_plane_context",
    "get_intent",
    "get_project_prompt_section",
    "get_user_prompt_section",
]
