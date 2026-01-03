"""
Knomly Pipeline Processors

Processors are single-responsibility frame transformers.
See ADR-001 for design decisions.
"""

from .audio_download import AudioDownloadProcessor
from .confirmation import ConfirmationProcessor
from .context_enrichment import (
    ContextEnrichmentProcessor,
    extract_plane_context,
    get_project_prompt_section,
    get_user_prompt_section,
)
from .extraction import ExtractionProcessor
from .intent_classifier import Intent, IntentClassifierProcessor, get_intent
from .transcription import TranscriptionProcessor
from .zulip import ZulipProcessor
from .integrations import PlaneProcessor, TaskCreatorProcessor
from .agent_bridge import AgentBridgeProcessor, create_task_agent_bridge

__all__ = [
    # Core processors
    "AudioDownloadProcessor",
    "TranscriptionProcessor",
    "IntentClassifierProcessor",
    "ExtractionProcessor",
    "ZulipProcessor",
    "ConfirmationProcessor",
    # Context enrichment (v1.5)
    "ContextEnrichmentProcessor",
    "extract_plane_context",
    "get_project_prompt_section",
    "get_user_prompt_section",
    # Integration processors (accept generic frames)
    "PlaneProcessor",
    "TaskCreatorProcessor",
    # Agent bridge (v2)
    "AgentBridgeProcessor",
    "create_task_agent_bridge",
    # Intent utilities
    "Intent",
    "get_intent",
]
