"""
Extraction Processor for Knomly.

Extracts structured standup information from transcriptions using LLM.

v1.5 Enhancement: Context-Aware Extraction
    If frame.metadata contains 'plane_context', the processor injects
    valid project/user lists into the LLM prompt. This enables the LLM
    to output correct entity IDs without requiring agent loops.

    Flow:
        TranscriptionFrame (with metadata.plane_context)
            ↓
        ExtractionProcessor (injects context into prompt)
            ↓
        ExtractionFrame (with resolved project/user names)

Design Principle (ADR-004):
    Context data flows through Frame metadata, not PipelineContext.
    The enriched metadata is visible in the Frame stream for debugging.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..processor import Processor
from .context_enrichment import extract_plane_context

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import Frame

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = """You are a standup extraction assistant. Extract structured information from the voice message transcription.

Extract:
1. today_items: Tasks planned for today
2. yesterday_items: Tasks completed yesterday (if mentioned)
3. blockers: Blockers or issues (if any)
4. summary: Brief 1-sentence summary

Return JSON only, no markdown:
{"today_items": [], "yesterday_items": [], "blockers": [], "summary": ""}

If a category is not mentioned, use empty list. Always return valid JSON."""

# Context-aware prompt template
CONTEXT_AWARE_PROMPT = """You are a standup extraction assistant. Extract structured information from the voice message transcription.

{context_section}

Extract:
1. today_items: Tasks planned for today (include project name if mentioned)
2. yesterday_items: Tasks completed yesterday (if mentioned)
3. blockers: Blockers or issues (if any)
4. summary: Brief 1-sentence summary
5. project: Project name if mentioned (match against valid projects above)

Return JSON only, no markdown:
{{"today_items": [], "yesterday_items": [], "blockers": [], "summary": "", "project": null}}

If a category is not mentioned, use empty list or null. Always return valid JSON.
When the user mentions a project, match it to the closest valid project name from the list above."""


class ExtractionProcessor(Processor):
    """
    Extracts standup items from transcription using LLM.

    Input: TranscriptionFrame
    Output: ExtractionFrame with structured standup data

    Uses LLM provider from ctx.providers.
    User config (zulip_stream, zulip_topic) from ctx.

    v1.5 Context-Aware Mode:
        If frame.metadata contains 'plane_context' (added by
        ContextEnrichmentProcessor), the LLM prompt is enhanced
        with valid project/user lists for better extraction.
    """

    def __init__(
        self,
        provider_name: str | None = None,
        *,
        use_context: bool = True,
    ):
        """
        Args:
            provider_name: Specific LLM provider to use (or default)
            use_context: Whether to use enriched context from metadata
        """
        self._provider_name = provider_name
        self._use_context = use_context

    @property
    def name(self) -> str:
        return "extraction"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | None:
        from ..frames import ExtractionFrame, TranscriptionFrame

        if not isinstance(frame, TranscriptionFrame):
            return frame

        if not frame.text.strip():
            raise ValueError("TranscriptionFrame has empty text")

        if ctx.providers is None:
            raise RuntimeError("No providers configured in context")

        llm = ctx.providers.get_llm(self._provider_name)

        logger.info(f"Extracting standup from {len(frame.text)} chars with {llm.name}")

        # Build system prompt with optional context injection
        system_prompt = self._build_prompt(frame, ctx)

        # Make LLM call with JSON mode if supported
        from knomly.providers.llm import LLMConfig, Message

        messages = [
            Message.system(system_prompt),
            Message.user(f"Transcription:\n{frame.text}"),
        ]

        response = await llm.complete(
            messages=messages,
            config=LLMConfig(
                temperature=0.3,
                max_tokens=512,
                response_format="json",  # Request JSON mode if provider supports it
            ),
        )

        # Parse extraction using robust JSON parser
        from knomly.utils.json_parser import parse_standup_json

        extraction = parse_standup_json(response.content)

        logger.info(
            f"Extracted: {len(extraction.get('today_items', []))} today, "
            f"{len(extraction.get('blockers', []))} blockers"
        )

        return ExtractionFrame(
            today_items=tuple(extraction.get("today_items", [])),
            yesterday_items=tuple(extraction.get("yesterday_items", [])),
            blockers=tuple(extraction.get("blockers", [])),
            summary=extraction.get("summary", ""),
            sender_phone=frame.sender_phone,
            user_name=ctx.user_name,
            zulip_stream=ctx.zulip_stream,
            zulip_topic=ctx.zulip_topic,
            source_frame_id=frame.id,
            # Forward context metadata for downstream processors
            metadata={
                **frame.metadata,
                "extracted_project": extraction.get("project"),
            },
        )

    def _build_prompt(self, frame: Frame, ctx: PipelineContext) -> str:
        """
        Build system prompt, optionally with context injection.

        If frame has plane_context metadata and use_context is True,
        injects valid project/user lists into the prompt.
        """
        # Check for context from ContextEnrichmentProcessor
        plane_context = extract_plane_context(frame) if self._use_context else None

        if plane_context:
            # Build context section for prompt
            context_section = self._build_context_section(plane_context)
            logger.info(
                f"[extraction] Using context-aware prompt with "
                f"{len(plane_context.get('projects', []))} projects"
            )
            return CONTEXT_AWARE_PROMPT.format(context_section=context_section)

        # Use default prompt or config-based prompt
        if ctx.config:
            # Note: This is sync context, so we can't await here
            # In practice, the prompt would be pre-fetched
            pass

        return DEFAULT_SYSTEM_PROMPT

    def _build_context_section(self, plane_context: dict) -> str:
        """
        Build the context section for prompt injection.

        Args:
            plane_context: Dict from ContextEnrichmentProcessor

        Returns:
            Formatted context section for prompt
        """
        lines = []

        # Add projects
        projects = plane_context.get("projects", [])
        if projects:
            lines.append("**Valid Projects:**")
            for p in projects:
                identifier = p.get("identifier", "")
                name = p.get("name", "")
                if identifier:
                    lines.append(f"  - {name} ({identifier})")
                else:
                    lines.append(f"  - {name}")
            lines.append("")

        # Add users
        users = plane_context.get("users", [])
        if users:
            lines.append("**Valid Team Members:**")
            for u in users:
                name = u.get("name", "")
                email = u.get("email", "")
                if email:
                    lines.append(f"  - {name} ({email})")
                else:
                    lines.append(f"  - {name}")
            lines.append("")

        return "\n".join(lines) if lines else ""
