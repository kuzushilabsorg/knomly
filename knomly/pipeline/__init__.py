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
from .processor import PassthroughProcessor, Processor
from .ratelimit import (
    CompositeRateLimiter,
    InMemoryStorage,
    RateLimiter,
    RateLimitExceeded,
    RateLimitStorage,
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

__all__ = [
    "AGGRESSIVE_RETRY",
    "NO_RETRY",
    "RETRY_ONCE",
    "RETRY_WITH_BACKOFF",
    "AuditEntry",
    "AuditRepository",
    # Retry & Resilience
    "BackoffStrategy",
    "CircuitBreaker",
    "CircuitOpenError",
    "CircuitState",
    "CompositeRateLimiter",
    # Routing
    "Conditional",
    "ConstantBackoff",
    "DecorrelatedJitter",
    "Executable",
    "ExponentialBackoff",
    "FanOut",
    "FanOutStrategy",
    "Filter",
    "Guard",
    "InMemoryAuditRepository",
    "InMemoryStorage",
    "JSONLogger",
    "LinearBackoff",
    # Observability
    "LogLevel",
    "NoBackoff",
    "NoOpSpan",
    "NoOpTracer",
    "PassthroughProcessor",
    # Core
    "Pipeline",
    "PipelineBuilder",
    "PipelineContext",
    "PipelineExit",
    "PipelineLogger",
    "PipelineMetrics",
    "PipelineResult",
    "Processor",
    # Rate Limiting
    "RateLimitExceeded",
    "RateLimitStorage",
    "RateLimiter",
    "ResilientProcessor",
    "RetryPolicy",
    "RetryResult",
    "Router",
    "RoutingDecision",
    "RoutingError",
    "SendResult",
    "SlidingWindowLimiter",
    "Span",
    "StructuredLogger",
    "Switch",
    "TokenBucket",
    "Tracer",
    # Transport
    "TransportAdapter",
    "TransportNotFoundError",
    "TransportRegistry",
    "TwilioTransport",
    "TypeRouter",
    "create_audit_entry",
    "create_twilio_transport",
    "get_metrics",
    "get_tracer",
    "get_transport",
    "get_transport_registry",
    "rate_limited",
    "register_transport",
    "reset_metrics",
    "reset_transport_registry",
    "set_tracer",
    "with_retry",
]
