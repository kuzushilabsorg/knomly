"""
Observability for Knomly Pipeline.

Provides structured logging, metrics, and tracing support for
pipeline execution monitoring and debugging.

Design Philosophy:
- Structured logging by default (JSON-formatted)
- Optional OpenTelemetry integration
- Minimal overhead when not enabled
- Audit trail for compliance and debugging

See ADR-001 for design decisions.
"""
from __future__ import annotations

import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID

if TYPE_CHECKING:
    from .context import PipelineContext
    from .frames import Frame

logger = logging.getLogger(__name__)


# =============================================================================
# Log Levels
# =============================================================================


class LogLevel(Enum):
    """Standard log levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


# =============================================================================
# Structured Logger Protocol
# =============================================================================


class StructuredLogger(Protocol):
    """
    Protocol for structured logging implementations.

    Structured loggers emit logs as key-value pairs rather than
    plain strings, enabling better searchability and analysis.
    """

    def debug(self, message: str, **context: Any) -> None:
        """Log debug message."""
        ...

    def info(self, message: str, **context: Any) -> None:
        """Log info message."""
        ...

    def warning(self, message: str, **context: Any) -> None:
        """Log warning message."""
        ...

    def error(self, message: str, **context: Any) -> None:
        """Log error message."""
        ...


# =============================================================================
# JSON Logger Implementation
# =============================================================================


@dataclass
class JSONLogger:
    """
    Structured logger that outputs JSON-formatted logs.

    Each log entry includes:
    - timestamp (ISO 8601)
    - level
    - message
    - context fields
    - optional request_id for correlation

    Example output:
        {"timestamp": "2026-01-02T10:30:00Z", "level": "info",
         "message": "Pipeline started", "request_id": "abc-123",
         "processors": ["a", "b"]}
    """

    name: str = "knomly"
    request_id: str | None = None
    extra_context: dict[str, Any] = field(default_factory=dict)
    _python_logger: logging.Logger | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self._python_logger = logging.getLogger(self.name)

    def _log(self, level: LogLevel, message: str, context: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": level.value,
            "message": message,
            **self.extra_context,
            **context,
        }

        if self.request_id:
            record["request_id"] = self.request_id

        json_str = json.dumps(record, default=str)

        if self._python_logger:
            log_method = getattr(self._python_logger, level.value)
            log_method(json_str)
        else:
            print(json_str)

    def debug(self, message: str, **context: Any) -> None:
        self._log(LogLevel.DEBUG, message, context)

    def info(self, message: str, **context: Any) -> None:
        self._log(LogLevel.INFO, message, context)

    def warning(self, message: str, **context: Any) -> None:
        self._log(LogLevel.WARNING, message, context)

    def error(self, message: str, **context: Any) -> None:
        self._log(LogLevel.ERROR, message, context)

    def with_context(self, **extra: Any) -> "JSONLogger":
        """Create a new logger with additional context."""
        return JSONLogger(
            name=self.name,
            request_id=self.request_id,
            extra_context={**self.extra_context, **extra},
        )


# =============================================================================
# Pipeline Logger
# =============================================================================


@dataclass
class PipelineLogger:
    """
    Specialized logger for pipeline execution.

    Provides convenience methods for common pipeline events:
    - Pipeline start/end
    - Processor start/end
    - Frame processing
    - Errors and warnings

    Example:
        logger = PipelineLogger(request_id="abc-123")
        logger.pipeline_started(pipeline_name="standup", processors=["a", "b"])
        logger.processor_started(processor_name="transcription", frame_type="AudioInput")
        logger.processor_completed(processor_name="transcription", duration_ms=1234.5)
        logger.pipeline_completed(success=True, duration_ms=5000.0)
    """

    request_id: str
    pipeline_name: str = ""
    inner: StructuredLogger = field(default_factory=JSONLogger)

    def __post_init__(self) -> None:
        if isinstance(self.inner, JSONLogger):
            self.inner = JSONLogger(
                name="knomly.pipeline",
                request_id=self.request_id,
                extra_context={"pipeline": self.pipeline_name} if self.pipeline_name else {},
            )

    # Pipeline lifecycle
    def pipeline_started(
        self,
        pipeline_name: str,
        processors: list[str],
        frame_type: str,
        frame_id: str,
    ) -> None:
        self.inner.info(
            "Pipeline started",
            pipeline_name=pipeline_name,
            processors=processors,
            processor_count=len(processors),
            initial_frame_type=frame_type,
            initial_frame_id=frame_id,
        )

    def pipeline_completed(
        self,
        success: bool,
        duration_ms: float,
        output_count: int,
        error: str | None = None,
    ) -> None:
        if success:
            self.inner.info(
                "Pipeline completed",
                success=True,
                duration_ms=round(duration_ms, 2),
                output_count=output_count,
            )
        else:
            self.inner.error(
                "Pipeline failed",
                success=False,
                duration_ms=round(duration_ms, 2),
                output_count=output_count,
                error=error,
            )

    # Processor lifecycle
    def processor_started(
        self,
        processor_name: str,
        frame_type: str,
        frame_id: str,
    ) -> None:
        self.inner.debug(
            "Processor started",
            processor=processor_name,
            frame_type=frame_type,
            frame_id=frame_id,
        )

    def processor_completed(
        self,
        processor_name: str,
        duration_ms: float,
        output_type: str | None,
        output_count: int,
    ) -> None:
        self.inner.debug(
            "Processor completed",
            processor=processor_name,
            duration_ms=round(duration_ms, 2),
            output_type=output_type,
            output_count=output_count,
        )

    def processor_error(
        self,
        processor_name: str,
        error: str,
        error_type: str,
        frame_id: str,
    ) -> None:
        self.inner.error(
            "Processor error",
            processor=processor_name,
            error=error,
            error_type=error_type,
            frame_id=frame_id,
        )

    # Routing
    def routing_decision(
        self,
        router_name: str,
        selected_branch: str,
        frame_type: str,
        condition: str | None = None,
    ) -> None:
        self.inner.debug(
            "Routing decision",
            router=router_name,
            selected_branch=selected_branch,
            frame_type=frame_type,
            condition=condition,
        )

    # Async handoff
    def async_handoff(
        self,
        continuation_id: str,
        processors_remaining: int,
        frame_type: str,
    ) -> None:
        self.inner.info(
            "Async handoff",
            continuation_id=continuation_id,
            processors_remaining=processors_remaining,
            frame_type=frame_type,
        )

    # Retry
    def retry_attempt(
        self,
        operation: str,
        attempt: int,
        max_attempts: int,
        error: str,
        delay_ms: float,
    ) -> None:
        self.inner.warning(
            "Retry attempt",
            operation=operation,
            attempt=attempt,
            max_attempts=max_attempts,
            error=error,
            delay_ms=round(delay_ms, 2),
        )

    # Circuit breaker
    def circuit_state_change(
        self,
        circuit_name: str,
        from_state: str,
        to_state: str,
        failure_count: int,
    ) -> None:
        level = LogLevel.WARNING if to_state == "open" else LogLevel.INFO
        self.inner._log(
            level,
            "Circuit breaker state change",
            {
                "circuit": circuit_name,
                "from_state": from_state,
                "to_state": to_state,
                "failure_count": failure_count,
            },
        )


# =============================================================================
# Audit Entry
# =============================================================================


@dataclass
class AuditEntry:
    """
    Audit log entry for pipeline execution.

    Captures complete execution details for compliance,
    debugging, and analysis.
    """

    # Identification
    execution_id: str
    request_id: str
    timestamp: datetime

    # Pipeline info
    pipeline_name: str
    processors: list[str]

    # Execution details
    status: str  # "started", "completed", "failed"
    duration_ms: float | None = None
    success: bool | None = None
    error: str | None = None

    # Frame info
    input_frame_type: str = ""
    input_frame_id: str = ""
    output_frame_types: list[str] = field(default_factory=list)
    output_count: int = 0

    # Timing breakdown
    processor_timings: dict[str, float] = field(default_factory=dict)

    # Routing decisions
    routing_decisions: list[dict[str, Any]] = field(default_factory=list)

    # Context
    user_id: str = ""
    sender_phone: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "execution_id": self.execution_id,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "pipeline_name": self.pipeline_name,
            "processors": self.processors,
            "status": self.status,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
            "input_frame_type": self.input_frame_type,
            "input_frame_id": self.input_frame_id,
            "output_frame_types": self.output_frame_types,
            "output_count": self.output_count,
            "processor_timings": self.processor_timings,
            "routing_decisions": self.routing_decisions,
            "user_id": self.user_id,
            "sender_phone": self.sender_phone,
            "metadata": self.metadata,
        }


# =============================================================================
# Audit Repository Protocol
# =============================================================================


class AuditRepository(Protocol):
    """
    Protocol for persisting audit entries.

    Implementations can store to:
    - MongoDB (production)
    - In-memory (testing)
    - File (local development)
    """

    async def save(self, entry: AuditEntry) -> None:
        """Save an audit entry."""
        ...

    async def find_by_execution_id(self, execution_id: str) -> AuditEntry | None:
        """Find audit entry by execution ID."""
        ...

    async def find_by_request_id(self, request_id: str) -> list[AuditEntry]:
        """Find all audit entries for a request."""
        ...


# =============================================================================
# In-Memory Audit Repository (for testing)
# =============================================================================


@dataclass
class InMemoryAuditRepository:
    """
    In-memory audit repository for testing.

    Not suitable for production use.
    """

    entries: list[AuditEntry] = field(default_factory=list)
    max_entries: int = 1000

    async def save(self, entry: AuditEntry) -> None:
        self.entries.append(entry)

        # Prevent unbounded growth
        if len(self.entries) > self.max_entries:
            self.entries = self.entries[-self.max_entries :]

    async def find_by_execution_id(self, execution_id: str) -> AuditEntry | None:
        for entry in reversed(self.entries):
            if entry.execution_id == execution_id:
                return entry
        return None

    async def find_by_request_id(self, request_id: str) -> list[AuditEntry]:
        return [e for e in self.entries if e.request_id == request_id]

    def clear(self) -> None:
        self.entries.clear()


# =============================================================================
# Metrics
# =============================================================================


@dataclass
class PipelineMetrics:
    """
    Pipeline execution metrics.

    Tracks:
    - Execution counts
    - Duration histograms
    - Error rates
    - Processor performance

    Can be exported to Prometheus, StatsD, or other systems.
    """

    # Counters
    executions_total: int = 0
    executions_success: int = 0
    executions_failed: int = 0
    frames_processed: int = 0
    retries_total: int = 0
    circuit_opens: int = 0

    # Histograms (simplified as lists for now)
    execution_durations_ms: list[float] = field(default_factory=list)
    processor_durations_ms: dict[str, list[float]] = field(default_factory=dict)

    # Max entries for histograms
    max_histogram_entries: int = 1000

    def record_execution(self, success: bool, duration_ms: float) -> None:
        """Record a pipeline execution."""
        self.executions_total += 1
        if success:
            self.executions_success += 1
        else:
            self.executions_failed += 1

        self.execution_durations_ms.append(duration_ms)
        self._trim_histogram(self.execution_durations_ms)

    def record_processor(self, name: str, duration_ms: float) -> None:
        """Record a processor execution."""
        self.frames_processed += 1

        if name not in self.processor_durations_ms:
            self.processor_durations_ms[name] = []

        self.processor_durations_ms[name].append(duration_ms)
        self._trim_histogram(self.processor_durations_ms[name])

    def record_retry(self) -> None:
        """Record a retry attempt."""
        self.retries_total += 1

    def record_circuit_open(self) -> None:
        """Record a circuit breaker opening."""
        self.circuit_opens += 1

    def _trim_histogram(self, histogram: list[float]) -> None:
        """Trim histogram to max entries."""
        if len(histogram) > self.max_histogram_entries:
            del histogram[: len(histogram) - self.max_histogram_entries]

    def get_stats(self) -> dict[str, Any]:
        """Get summary statistics."""
        def percentile(data: list[float], p: float) -> float | None:
            if not data:
                return None
            sorted_data = sorted(data)
            k = (len(sorted_data) - 1) * p
            f = int(k)
            c = f + 1 if f + 1 < len(sorted_data) else f
            return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])

        return {
            "executions": {
                "total": self.executions_total,
                "success": self.executions_success,
                "failed": self.executions_failed,
                "success_rate": (
                    self.executions_success / self.executions_total
                    if self.executions_total > 0
                    else None
                ),
            },
            "duration_ms": {
                "p50": percentile(self.execution_durations_ms, 0.5),
                "p95": percentile(self.execution_durations_ms, 0.95),
                "p99": percentile(self.execution_durations_ms, 0.99),
            },
            "frames_processed": self.frames_processed,
            "retries_total": self.retries_total,
            "circuit_opens": self.circuit_opens,
        }

    def reset(self) -> None:
        """Reset all metrics."""
        self.executions_total = 0
        self.executions_success = 0
        self.executions_failed = 0
        self.frames_processed = 0
        self.retries_total = 0
        self.circuit_opens = 0
        self.execution_durations_ms.clear()
        self.processor_durations_ms.clear()


# Global metrics instance (can be replaced with actual metrics backend)
_global_metrics = PipelineMetrics()


def get_metrics() -> PipelineMetrics:
    """Get the global metrics instance."""
    return _global_metrics


def reset_metrics() -> None:
    """Reset global metrics (useful for testing)."""
    _global_metrics.reset()


# =============================================================================
# Span/Trace Support (OpenTelemetry-compatible interface)
# =============================================================================


class Span(Protocol):
    """Protocol for trace spans (OpenTelemetry-compatible)."""

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        ...

    def set_status(self, status: str, description: str | None = None) -> None:
        """Set span status (ok, error)."""
        ...

    def record_exception(self, exception: Exception) -> None:
        """Record exception on span."""
        ...

    def end(self) -> None:
        """End the span."""
        ...


@dataclass
class NoOpSpan:
    """No-op span implementation when tracing is disabled."""

    def set_attribute(self, key: str, value: Any) -> None:
        pass

    def set_status(self, status: str, description: str | None = None) -> None:
        pass

    def record_exception(self, exception: Exception) -> None:
        pass

    def end(self) -> None:
        pass


class Tracer(Protocol):
    """Protocol for trace creation (OpenTelemetry-compatible)."""

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Start a new span."""
        ...


@dataclass
class NoOpTracer:
    """No-op tracer implementation when tracing is disabled."""

    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        return NoOpSpan()


# Global tracer (can be replaced with OpenTelemetry tracer)
_global_tracer: Tracer = NoOpTracer()


def get_tracer() -> Tracer:
    """Get the global tracer."""
    return _global_tracer


def set_tracer(tracer: Tracer) -> None:
    """Set the global tracer (for OpenTelemetry integration)."""
    global _global_tracer
    _global_tracer = tracer


# =============================================================================
# Convenience: Create audit entry from context
# =============================================================================


def create_audit_entry(
    ctx: "PipelineContext",
    pipeline_name: str,
    processors: list[str],
    input_frame: "Frame",
    status: str,
    success: bool | None = None,
    error: str | None = None,
    output_frames: list["Frame"] | None = None,
) -> AuditEntry:
    """Create an audit entry from pipeline context."""
    return AuditEntry(
        execution_id=str(ctx.execution_id),
        request_id=str(ctx.execution_id),  # Could be separate
        timestamp=ctx.started_at,
        pipeline_name=pipeline_name,
        processors=processors,
        status=status,
        duration_ms=ctx.elapsed_ms,
        success=success,
        error=error,
        input_frame_type=input_frame.frame_type,
        input_frame_id=str(input_frame.id),
        output_frame_types=[f.frame_type for f in (output_frames or [])],
        output_count=len(output_frames or []),
        processor_timings=ctx.processor_timings,
        routing_decisions=[d.to_dict() for d in ctx.routing_decisions],
        user_id=ctx.user_id,
        sender_phone=ctx.sender_phone,
        metadata={},
    )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Logging
    "LogLevel",
    "StructuredLogger",
    "JSONLogger",
    "PipelineLogger",
    # Audit
    "AuditEntry",
    "AuditRepository",
    "InMemoryAuditRepository",
    "create_audit_entry",
    # Metrics
    "PipelineMetrics",
    "get_metrics",
    "reset_metrics",
    # Tracing
    "Span",
    "Tracer",
    "NoOpSpan",
    "NoOpTracer",
    "get_tracer",
    "set_tracer",
]
