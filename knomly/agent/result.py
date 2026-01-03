"""
Agent Execution Result.

This module defines the result structure returned by AgentExecutor.
It contains all frames emitted during execution for full observability.

Design Principle (ADR-004):
    The result contains the complete frame stream.
    Execution can be reconstructed from frames alone.

Usage:
    result = await executor.run(goal="Create a task")

    if result.success:
        print(result.response.response_text)
    else:
        print(f"Failed: {result.error_message}")

    # Audit: inspect all frames
    for frame in result.frames:
        print(f"{frame.frame_type}: {frame.id}")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knomly.pipeline.frames.base import Frame
    from .frames import AgentResponseFrame


@dataclass(frozen=True, slots=True)
class AgentResult:
    """
    Result from agent execution.

    Contains:
    - success: Whether the goal was achieved
    - response: Final AgentResponseFrame (if successful)
    - frames: Complete frame stream for audit/replay
    - iterations: How many loop iterations were used
    - tools_called: Which tools were invoked

    The frames tuple contains every frame emitted during execution,
    in order. This enables complete reconstruction of the execution.

    Example:
        result = await executor.run(goal="Create a task")

        # Check success
        if result.success:
            print(result.response.response_text)

        # Audit all frames
        for frame in result.frames:
            if frame.frame_type == "tool_call":
                print(f"Called: {frame.tool_name}")
            elif frame.frame_type == "tool_result":
                print(f"Result: {frame.result_text}")
    """

    # Whether the goal was achieved
    success: bool

    # Final response frame (present if success=True)
    response: "AgentResponseFrame | None" = None

    # Complete frame stream (for audit/replay)
    frames: tuple["Frame", ...] = ()

    # How many iterations were used
    iterations: int = 0

    # Which tools were called (in order)
    tools_called: tuple[str, ...] = ()

    # Execution timing
    started_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime = field(default_factory=datetime.utcnow)

    # Error information (if success=False)
    error_message: str = ""
    error_type: str = ""  # "timeout", "max_iterations", "tool_error", "unexpected"

    @property
    def duration_ms(self) -> float:
        """Total execution time in milliseconds."""
        return (self.completed_at - self.started_at).total_seconds() * 1000

    @property
    def response_text(self) -> str:
        """Convenience accessor for response text."""
        if self.response:
            return self.response.response_text
        return self.error_message or "No response"

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Useful for storing in audit logs or sending to monitoring.
        """
        return {
            "success": self.success,
            "response_text": self.response_text,
            "iterations": self.iterations,
            "tools_called": list(self.tools_called),
            "duration_ms": self.duration_ms,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat(),
            "error_message": self.error_message,
            "error_type": self.error_type,
            "frame_count": len(self.frames),
            "frame_types": [f.frame_type for f in self.frames],
        }


# =============================================================================
# Factory Functions
# =============================================================================


def success_result(
    response: "AgentResponseFrame",
    frames: tuple["Frame", ...],
    *,
    iterations: int = 0,
    tools_called: tuple[str, ...] = (),
    started_at: datetime | None = None,
) -> AgentResult:
    """
    Create a successful result.

    Args:
        response: Final AgentResponseFrame
        frames: Complete frame stream
        iterations: Number of iterations used
        tools_called: Tools that were invoked
        started_at: When execution started

    Returns:
        AgentResult with success=True
    """
    return AgentResult(
        success=True,
        response=response,
        frames=frames,
        iterations=iterations,
        tools_called=tools_called,
        started_at=started_at or datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
    )


def timeout_result(
    frames: tuple["Frame", ...],
    *,
    iterations: int = 0,
    tools_called: tuple[str, ...] = (),
    timeout_seconds: float = 0.0,
    started_at: datetime | None = None,
) -> AgentResult:
    """
    Create a timeout result.

    Args:
        frames: Frames emitted before timeout
        iterations: Iterations completed before timeout
        tools_called: Tools called before timeout
        timeout_seconds: Configured timeout
        started_at: When execution started

    Returns:
        AgentResult with success=False, error_type="timeout"
    """
    return AgentResult(
        success=False,
        frames=frames,
        iterations=iterations,
        tools_called=tools_called,
        started_at=started_at or datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_message=f"Agent execution timed out after {timeout_seconds}s",
        error_type="timeout",
    )


def max_iterations_result(
    frames: tuple["Frame", ...],
    *,
    max_iterations: int = 5,
    tools_called: tuple[str, ...] = (),
    started_at: datetime | None = None,
) -> AgentResult:
    """
    Create a max iterations result.

    Args:
        frames: Frames emitted during execution
        max_iterations: Maximum iterations that was reached
        tools_called: Tools called during execution
        started_at: When execution started

    Returns:
        AgentResult with success=False, error_type="max_iterations"
    """
    return AgentResult(
        success=False,
        frames=frames,
        iterations=max_iterations,
        tools_called=tools_called,
        started_at=started_at or datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_message=f"Agent reached maximum iterations ({max_iterations})",
        error_type="max_iterations",
    )


def error_result(
    error: Exception,
    frames: tuple["Frame", ...],
    *,
    iterations: int = 0,
    tools_called: tuple[str, ...] = (),
    started_at: datetime | None = None,
) -> AgentResult:
    """
    Create an error result.

    Args:
        error: The exception that occurred
        frames: Frames emitted before error
        iterations: Iterations completed before error
        tools_called: Tools called before error
        started_at: When execution started

    Returns:
        AgentResult with success=False, error_type="unexpected"
    """
    return AgentResult(
        success=False,
        frames=frames,
        iterations=iterations,
        tools_called=tools_called,
        started_at=started_at or datetime.now(timezone.utc),
        completed_at=datetime.now(timezone.utc),
        error_message=str(error),
        error_type="unexpected",
    )
