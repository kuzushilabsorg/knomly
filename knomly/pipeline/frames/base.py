"""
Base Frame abstraction for Knomly pipeline.

Frames are immutable data containers that flow through the pipeline.
Adapted from Pipecat patterns for HTTP request/response context.

See ADR-001 for design decisions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, TypeVar
from uuid import UUID, uuid4

# Type variable for generic derive() method
F = TypeVar("F", bound="Frame")


def _utc_now() -> datetime:
    """Get current UTC time."""
    return datetime.now(UTC)


@dataclass(frozen=True, kw_only=True, slots=True)
class Frame:
    """
    Base class for all frames in the Knomly pipeline.

    Frames are immutable data containers with:
    - Unique ID for tracking
    - Creation timestamp
    - Source frame ID for lineage tracking
    - Metadata for extensibility

    Design Principles (from ADR-001):
    - Truly immutable (frozen dataclass, no __setattr__ hacks)
    - Create new frames via derive() instead of mutation
    - Lineage tracked via source_frame_id chain
    """

    id: UUID = field(default_factory=uuid4)
    created_at: datetime = field(default_factory=_utc_now)
    source_frame_id: UUID | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_type(self) -> str:
        """Frame type name for logging and debugging."""
        return self.__class__.__name__

    def derive(self: F, **changes: Any) -> F:
        """
        Create a new frame derived from this one.

        The new frame will have:
        - New unique ID
        - New timestamp
        - source_frame_id pointing to this frame
        - Any field overrides from **changes

        Example:
            new_frame = old_frame.derive(some_field="new_value")
        """
        return replace(
            self,
            id=uuid4(),
            created_at=_utc_now(),
            source_frame_id=self.id,
            **changes,
        )

    def with_metadata(self: F, **extra: Any) -> F:
        """Create new frame with additional metadata."""
        return self.derive(metadata={**self.metadata, **extra})

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame to dictionary for logging/storage."""
        return {
            "id": str(self.id),
            "frame_type": self.frame_type,
            "created_at": self.created_at.isoformat(),
            "source_frame_id": str(self.source_frame_id) if self.source_frame_id else None,
            "metadata": self.metadata,
        }

    def __repr__(self) -> str:
        return f"{self.frame_type}(id={str(self.id)[:8]}...)"


@dataclass(frozen=True, kw_only=True, slots=True)
class ErrorFrame(Frame):
    """
    Frame representing an error that occurred during processing.

    ErrorFrames flow through the pipeline like regular frames,
    allowing downstream processors to handle or report them.
    """

    error_type: str = "unknown"
    error_message: str = "An error occurred"
    error_code: str | None = None
    processor_name: str = ""
    original_frame_type: str = ""
    exception_class: str | None = None
    is_fatal: bool = False
    sender_phone: str | None = None

    @classmethod
    def from_exception(
        cls,
        exc: Exception,
        processor_name: str,
        source_frame: Frame | None = None,
        is_fatal: bool = False,
    ) -> ErrorFrame:
        """Create ErrorFrame from an exception."""
        return cls(
            error_type=_classify_error(exc),
            error_message=str(exc),
            processor_name=processor_name,
            original_frame_type=source_frame.frame_type if source_frame else "",
            exception_class=type(exc).__name__,
            is_fatal=is_fatal,
            source_frame_id=source_frame.id if source_frame else None,
            sender_phone=getattr(source_frame, "sender_phone", None),
        )

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update(
            {
                "error_type": self.error_type,
                "error_message": self.error_message,
                "error_code": self.error_code,
                "processor_name": self.processor_name,
                "original_frame_type": self.original_frame_type,
                "is_fatal": self.is_fatal,
            }
        )
        return base

    def format_user_message(self) -> str:
        """Format a user-friendly error message."""
        messages = {
            "network": "Network issue. Please try again.",
            "timeout": "Request timed out. Please try again.",
            "rate_limit": "Too many requests. Please wait a moment.",
            "validation": f"Invalid input: {self.error_message}",
            "auth": "Authentication issue. We're looking into it.",
            "config": "Configuration issue. Please contact support.",
        }
        return messages.get(self.error_type, "Something went wrong. Please try again.")


def _classify_error(exc: Exception) -> str:
    """Classify exception into error type."""
    name = type(exc).__name__.lower()
    msg = str(exc).lower()

    if "timeout" in name or "timeout" in msg:
        return "timeout"
    if "connection" in name or "network" in name:
        return "network"
    if "ratelimit" in name or "429" in msg:
        return "rate_limit"
    if "auth" in name or "401" in msg or "403" in msg:
        return "auth"
    if "validation" in name or "invalid" in name:
        return "validation"
    return "internal"


# =============================================================================
# Control Frames
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class StartFrame(Frame):
    """
    Control frame signaling the start of pipeline execution.

    Emitted by pipeline at the beginning of execution.
    Processors can use this for initialization that depends on
    being inside a pipeline run.
    """

    pipeline_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base["pipeline_id"] = self.pipeline_id
        return base


@dataclass(frozen=True, kw_only=True, slots=True)
class EndFrame(Frame):
    """
    Control frame signaling the end of pipeline execution.

    Emitted by pipeline at the end of execution.
    Processors can use this for cleanup or finalization that depends on
    all frames having been processed.
    """

    pipeline_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base["pipeline_id"] = self.pipeline_id
        return base


# Backwards compatibility for ErrorType enum usage
class ErrorType:
    """Error type constants for ErrorFrame classification."""

    NETWORK = "network"
    TIMEOUT = "timeout"
    RATE_LIMIT = "rate_limit"
    AUTH = "auth"
    VALIDATION = "validation"
    CONFIG = "config"
    INTERNAL = "internal"
