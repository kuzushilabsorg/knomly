"""
Knomly Pipeline Framework

A Pipecat-inspired pipeline architecture adapted for HTTP/FastAPI.
See ADR-001 for design decisions.

Core Components:
- Frame: Immutable data containers that flow through the pipeline
- Processor: Single-responsibility transformers
- Pipeline: Sequential executor
- Context: Request-scoped state and provider access
- Routing: Control flow primitives (Conditional, Switch, Filter, etc.)
- Transport: Bidirectional message channel adapters

For async execution in HTTP contexts (e.g., Twilio webhooks), use
FastAPI's native BackgroundTasks. See app/api/webhooks/twilio.py
for the pattern.

Transport Pattern:
    # At app startup
    from knomly.pipeline.transports import TwilioTransport, register_transport
    register_transport(TwilioTransport(...))

    # In webhook handler
    transport = get_transport("twilio")
    frame = await transport.normalize_request(request)

    # ConfirmationProcessor uses ctx.channel_id automatically
"""

from .context import PipelineContext, PipelineResult
from .executor import Pipeline, PipelineBuilder
from .processor import PassthroughProcessor, Processor
from .transports import (
    SendResult,
    TransportAdapter,
    TransportNotFoundError,
    TransportRegistry,
    TwilioTransport,
    create_twilio_transport,
    get_transport,
    get_transport_registry,
    register_transport,
    reset_transport_registry,
)
from .observability import (
    AuditEntry,
    AuditRepository,
    InMemoryAuditRepository,
    JSONLogger,
    LogLevel,
    NoOpSpan,
    NoOpTracer,
    PipelineLogger,
    PipelineMetrics,
    Span,
    StructuredLogger,
    Tracer,
    create_audit_entry,
    get_metrics,
    get_tracer,
    reset_metrics,
    set_tracer,
)
from .ratelimit import (
    CompositeRateLimiter,
    InMemoryStorage,
    RateLimitExceeded,
    RateLimitStorage,
    RateLimiter,
    SlidingWindowLimiter,
    TokenBucket,
    rate_limited,
)
from .retry import (
    AGGRESSIVE_RETRY,
    NO_RETRY,
    RETRY_ONCE,
    RETRY_WITH_BACKOFF,
    BackoffStrategy,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ConstantBackoff,
    DecorrelatedJitter,
    ExponentialBackoff,
    LinearBackoff,
    NoBackoff,
    ResilientProcessor,
    RetryPolicy,
    RetryResult,
    with_retry,
)
from .routing import (
    Conditional,
    Executable,
    FanOut,
    FanOutStrategy,
    Filter,
    Guard,
    PipelineExit,
    Router,
    RoutingDecision,
    RoutingError,
    Switch,
    TypeRouter,
)

__all__ = [
    # Core
    "Pipeline",
    "PipelineBuilder",
    "PipelineContext",
    "PipelineResult",
    "Processor",
    "PassthroughProcessor",
    # Routing
    "Conditional",
    "Switch",
    "TypeRouter",
    "Filter",
    "Guard",
    "FanOut",
    "FanOutStrategy",
    "Router",
    "RoutingDecision",
    "RoutingError",
    "PipelineExit",
    "Executable",
    # Transport
    "TransportAdapter",
    "SendResult",
    "TransportRegistry",
    "TransportNotFoundError",
    "TwilioTransport",
    "create_twilio_transport",
    "get_transport_registry",
    "get_transport",
    "register_transport",
    "reset_transport_registry",
    # Retry & Resilience
    "BackoffStrategy",
    "NoBackoff",
    "ConstantBackoff",
    "LinearBackoff",
    "ExponentialBackoff",
    "DecorrelatedJitter",
    "RetryPolicy",
    "RetryResult",
    "with_retry",
    "NO_RETRY",
    "RETRY_ONCE",
    "RETRY_WITH_BACKOFF",
    "AGGRESSIVE_RETRY",
    "CircuitState",
    "CircuitBreaker",
    "CircuitOpenError",
    "ResilientProcessor",
    # Observability
    "LogLevel",
    "StructuredLogger",
    "JSONLogger",
    "PipelineLogger",
    "AuditEntry",
    "AuditRepository",
    "InMemoryAuditRepository",
    "create_audit_entry",
    "PipelineMetrics",
    "get_metrics",
    "reset_metrics",
    "Span",
    "Tracer",
    "NoOpSpan",
    "NoOpTracer",
    "get_tracer",
    "set_tracer",
    # Rate Limiting
    "RateLimitExceeded",
    "TokenBucket",
    "RateLimitStorage",
    "InMemoryStorage",
    "RateLimiter",
    "SlidingWindowLimiter",
    "CompositeRateLimiter",
    "rate_limited",
]
