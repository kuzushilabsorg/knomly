"""
Context Enrichment Processor.

This processor enriches frames with entity context (projects, users) to enable
context-aware LLM extraction without requiring full agent loops.

Design Principle (ADR-004 compliant):
    - Context data flows through Frame METADATA, not PipelineContext
    - Cache is a SERVICE, accessed via provider registry
    - Output frames contain all necessary context for downstream processors

This is the v1.5 solution to the "Blind Macro" problem:
    - User says: "Create a task for the Mobile App"
    - Without context: LLM hallucinates UUID
    - With context: LLM sees valid projects in metadata, outputs correct UUID

Usage:
    pipeline = (
        PipelineBuilder()
        .add(TranscriptionProcessor())
        .add(ContextEnrichmentProcessor(cache=plane_cache))  # Enriches metadata
        .add(ExtractionProcessor())  # Reads metadata for prompt
        .add(PlaneProcessor(client))
        .build()
    )

Flow:
    TranscriptionFrame
        ↓
    ContextEnrichmentProcessor (adds metadata.plane_context)
        ↓
    TranscriptionFrame (enriched)
        ↓
    ExtractionProcessor (reads metadata, injects into prompt)
        ↓
    TaskFrame
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Sequence

from knomly.pipeline.processor import Processor
from knomly.pipeline.frames.base import Frame

if TYPE_CHECKING:
    from knomly.pipeline.context import PipelineContext
    from knomly.integrations.plane.cache import PlaneEntityCache

logger = logging.getLogger(__name__)


class ContextEnrichmentProcessor(Processor):
    """
    Enriches frames with entity context from integration caches.

    This processor adds context about valid entities (projects, users, etc.)
    to frame metadata, enabling downstream processors to make informed decisions.

    The enrichment follows ADR-004:
        - Context data flows through Frame metadata
        - Not through PipelineContext
        - Visible in Frame stream for debugging/replay
        - Context snapshot is logged for observability

    Failure Mode (Graceful Degradation):
        - If cache refresh fails, processor continues with empty/stale context
        - Pipeline never crashes due to context unavailability
        - Error status is included in Frame metadata for debugging

    Supported caches:
        - PlaneEntityCache: Adds plane_context with projects, users

    Future: Can be extended to support multiple caches (Linear, Twenty, etc.)
    """

    def __init__(
        self,
        *,
        plane_cache: "PlaneEntityCache | None" = None,
        auto_refresh: bool = True,
    ):
        """
        Initialize context enrichment processor.

        Args:
            plane_cache: Plane entity cache for project/user context
            auto_refresh: Whether to auto-refresh stale caches
        """
        self._plane_cache = plane_cache
        self._auto_refresh = auto_refresh

    @property
    def name(self) -> str:
        return "context_enrichment"

    async def process(
        self,
        frame: Frame,
        ctx: "PipelineContext",
    ) -> Frame | Sequence[Frame] | None:
        """
        Enrich frame with entity context.

        The enrichment is added to frame.metadata, which is then available
        to downstream processors (e.g., ExtractionProcessor for prompt injection).

        IMPORTANT: This processor NEVER throws. On any error, it logs and
        returns the original frame unchanged.

        Args:
            frame: Any frame to enrich
            ctx: Pipeline context (used for logging only)

        Returns:
            Same frame with enriched metadata (or original on error)
        """
        try:
            enrichments: dict = {}

            # Enrich with Plane context (uses safe_get_context, never throws)
            if self._plane_cache:
                enrichments.update(await self._enrich_plane_context())

            if not enrichments:
                # No caches configured, pass through
                return frame

            # Merge enrichments into frame metadata
            # Note: Frame is frozen, so we return a new frame with updated metadata
            new_metadata = {**frame.metadata, **enrichments}

            # Create new frame with enriched metadata
            enriched_frame = self._with_metadata(frame, new_metadata)

            # Log context snapshot for observability (ADR-004 compliance)
            self._log_context_snapshot(enrichments, frame)

            return enriched_frame

        except Exception as e:
            # This should never happen, but if it does, continue without context
            logger.error(
                f"[context_enrichment] Unexpected error enriching {frame.frame_type}: {e}. "
                f"Continuing without context."
            )
            return frame

    async def _enrich_plane_context(self) -> dict:
        """
        Get Plane context for prompt injection.

        Uses safe_get_context() which NEVER throws and handles all
        failures gracefully (returns empty context on error).

        Returns:
            Dict with plane_context key containing projects, users
        """
        if not self._plane_cache:
            return {}

        # Use safe_get_context which handles all errors gracefully
        return await self._plane_cache.safe_get_context(
            auto_refresh=self._auto_refresh
        )

    def _log_context_snapshot(self, enrichments: dict, frame: Frame) -> None:
        """
        Log the context snapshot for observability.

        This ensures the Frame stream is self-describing:
        "If I can't explain an execution by looking only at the Frame stream,
         the design is broken." - ADR-004

        The log shows exactly what context was available at this point in time.
        """
        plane_context = enrichments.get("plane_context", {})
        projects = plane_context.get("projects", [])
        users = plane_context.get("users", [])
        cache_healthy = plane_context.get("cache_healthy", False)
        cache_error = plane_context.get("cache_error")

        # Build observability log
        project_summary = [p.get("name", "?") for p in projects[:5]]
        if len(projects) > 5:
            project_summary.append(f"...+{len(projects) - 5} more")

        if cache_healthy:
            logger.info(
                f"[context_enrichment] Enriched {frame.frame_type} | "
                f"projects={project_summary} | users={len(users)} | "
                f"frame_id={frame.id}"
            )
        else:
            logger.warning(
                f"[context_enrichment] Enriched {frame.frame_type} with DEGRADED context | "
                f"cache_error={cache_error} | "
                f"frame_id={frame.id}"
            )

    def _with_metadata(self, frame: Frame, metadata: dict) -> Frame:
        """
        Create a copy of frame with updated metadata.

        Since frames are frozen, we need to create a new instance.
        This preserves the frame type and all other fields.
        """
        from dataclasses import fields, replace

        try:
            # Use dataclass replace for frozen dataclasses
            return replace(frame, metadata=metadata)
        except Exception:
            # Fallback: manually copy and update
            # This handles cases where replace doesn't work
            frame_dict = {f.name: getattr(frame, f.name) for f in fields(frame)}
            frame_dict["metadata"] = metadata
            return type(frame)(**frame_dict)


# =============================================================================
# Utility Functions
# =============================================================================


def extract_plane_context(frame: Frame) -> dict | None:
    """
    Extract Plane context from enriched frame metadata.

    Utility for downstream processors to read the enriched context.

    Args:
        frame: Frame that was enriched by ContextEnrichmentProcessor

    Returns:
        Plane context dict or None if not present
    """
    return frame.metadata.get("plane_context")


def get_project_prompt_section(frame: Frame) -> str:
    """
    Generate prompt section for project context.

    Used by ExtractionProcessor to inject project list into LLM prompt.

    Args:
        frame: Enriched frame with plane_context

    Returns:
        Formatted string for prompt injection
    """
    context = extract_plane_context(frame)
    if not context:
        return ""

    projects = context.get("projects", [])
    if not projects:
        return ""

    lines = ["**Valid Projects (use exact ID):**"]
    for p in projects:
        lines.append(f"  - {p['name']} ({p['identifier']}): {p['id']}")

    return "\n".join(lines)


def get_user_prompt_section(frame: Frame) -> str:
    """
    Generate prompt section for user context.

    Used by ExtractionProcessor to inject user list into LLM prompt.

    Args:
        frame: Enriched frame with plane_context

    Returns:
        Formatted string for prompt injection
    """
    context = extract_plane_context(frame)
    if not context:
        return ""

    users = context.get("users", [])
    if not users:
        return ""

    lines = ["**Valid Users (use exact ID for assignees):**"]
    for u in users:
        lines.append(f"  - {u['name']} ({u.get('email', '')}): {u['id']}")

    return "\n".join(lines)
