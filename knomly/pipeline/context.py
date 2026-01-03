"""
Pipeline Context for Knomly.

The context provides request-scoped state and access to providers
for all processors in the pipeline.

See ADR-001 for design decisions.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

if TYPE_CHECKING:
    from knomly.providers.registry import ProviderRegistry
    from knomly.config.service import ConfigurationService
    from .routing import RoutingDecision


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class PipelineContext:
    """
    Request-scoped context passed through the pipeline.

    Provides:
    - Unique execution ID for tracing
    - Provider access (STT, LLM, Chat)
    - Configuration service access
    - User context from config lookup
    - Audit trail of frame processing

    The context is created at pipeline start and passed to
    every processor's process() method.
    """

    # Execution identification
    execution_id: UUID = field(default_factory=uuid4)
    started_at: datetime = field(default_factory=_utc_now)

    # Provider and config access (injected by pipeline runner)
    providers: ProviderRegistry | None = None
    config: ConfigurationService | None = None

    # Request context (from webhook)
    sender_phone: str = ""
    message_type: str = "audio"
    channel_id: str = ""  # Transport channel (e.g., "twilio", "telegram")

    # User context (populated after config lookup)
    user_id: str = ""
    user_name: str = ""
    zulip_stream: str = "standup"
    zulip_topic: str = ""

    # Audit trail
    processor_timings: dict[str, float] = field(default_factory=dict)
    frame_log: list[dict[str, Any]] = field(default_factory=list)
    routing_decisions: list["RoutingDecision"] = field(default_factory=list)

    @property
    def elapsed_ms(self) -> float:
        """Milliseconds since pipeline started."""
        delta = datetime.now(timezone.utc) - self.started_at
        return delta.total_seconds() * 1000

    def record_frame(self, frame_dict: dict[str, Any], processor_name: str) -> None:
        """Record a frame in the audit trail."""
        self.frame_log.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "processor": processor_name,
            "frame": frame_dict,
            "elapsed_ms": self.elapsed_ms,
        })

    def record_timing(self, processor_name: str, duration_ms: float) -> None:
        """Record processor execution timing."""
        self.processor_timings[processor_name] = duration_ms

    def to_audit_dict(self) -> dict[str, Any]:
        """Generate audit record for storage."""
        return {
            "execution_id": str(self.execution_id),
            "started_at": self.started_at.isoformat(),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "duration_ms": self.elapsed_ms,
            "sender_phone": self.sender_phone,
            "message_type": self.message_type,
            "channel_id": self.channel_id,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "processor_timings": self.processor_timings,
            "frame_count": len(self.frame_log),
            "routing_decisions": [d.to_dict() for d in self.routing_decisions],
        }

    def copy(self) -> "PipelineContext":
        """
        Create an isolated copy for background execution.

        Shares read-only references (providers, config) but copies
        mutable state (frame_log, timings, routing_decisions).
        """
        return PipelineContext(
            execution_id=self.execution_id,
            started_at=self.started_at,
            providers=self.providers,  # Shared, stateless
            config=self.config,  # Shared, stateless
            sender_phone=self.sender_phone,
            message_type=self.message_type,
            channel_id=self.channel_id,
            user_id=self.user_id,
            user_name=self.user_name,
            zulip_stream=self.zulip_stream,
            zulip_topic=self.zulip_topic,
            processor_timings=copy.copy(self.processor_timings),
            frame_log=copy.copy(self.frame_log),
            routing_decisions=copy.copy(self.routing_decisions),
        )


@dataclass
class PipelineResult:
    """
    Result of pipeline execution.

    Contains all frames produced by the pipeline,
    timing information, and any errors encountered.
    """

    context: PipelineContext
    output_frames: list[Any] = field(default_factory=list)
    success: bool = True
    error: str | None = None

    @property
    def execution_id(self) -> UUID:
        return self.context.execution_id

    @property
    def duration_ms(self) -> float:
        return self.context.elapsed_ms

    def get_frame(self, frame_type: type) -> Any | None:
        """Get the first frame of a specific type."""
        for frame in self.output_frames:
            if isinstance(frame, frame_type):
                return frame
        return None

    def get_frames(self, frame_type: type) -> list[Any]:
        """Get all frames of a specific type."""
        return [f for f in self.output_frames if isinstance(f, frame_type)]

    def to_dict(self) -> dict[str, Any]:
        """Serialize result for logging/API response."""
        return {
            "execution_id": str(self.execution_id),
            "success": self.success,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "output_frame_count": len(self.output_frames),
            "output_frame_types": [f.frame_type for f in self.output_frames],
        }
