"""
Retry and Resilience Patterns for Knomly Pipeline.

Provides mechanisms for handling transient failures:
- RetryPolicy: Configurable retry behavior for processors
- BackoffStrategy: Delay calculation between retries
- CircuitBreaker: Prevent cascading failures

Design Philosophy:
- Configurable per-processor retry policies
- Composable backoff strategies
- Circuit breaker for external service protection
- Graceful degradation for non-critical operations

See ADR-001 for design decisions.
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from .context import PipelineContext
    from .frames import Frame

logger = logging.getLogger(__name__)

T = TypeVar("T")


# =============================================================================
# Backoff Strategies
# =============================================================================


class BackoffStrategy(ABC):
    """
    Abstract base for backoff delay calculation.

    Backoff strategies determine how long to wait between retry attempts.
    Different strategies suit different failure modes.
    """

    @abstractmethod
    def get_delay(self, attempt: int) -> float:
        """
        Calculate delay before next retry attempt.

        Args:
            attempt: Current attempt number (1-indexed, first retry is attempt 1)

        Returns:
            Delay in seconds before next attempt
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Reset any internal state (for reuse)."""
        ...


@dataclass
class NoBackoff(BackoffStrategy):
    """
    No delay between retries.

    Use for:
    - Quick local retries
    - Testing
    - Operations that should fail fast
    """

    def get_delay(self, attempt: int) -> float:
        return 0.0

    def reset(self) -> None:
        pass


@dataclass
class ConstantBackoff(BackoffStrategy):
    """
    Fixed delay between retries.

    Use for:
    - Simple retry scenarios
    - When consistent timing is needed

    Example:
        backoff = ConstantBackoff(delay=1.0)
        # Always waits 1 second between retries
    """

    delay: float = 1.0

    def get_delay(self, attempt: int) -> float:
        return self.delay

    def reset(self) -> None:
        pass


@dataclass
class LinearBackoff(BackoffStrategy):
    """
    Linearly increasing delay between retries.

    delay = initial + (attempt - 1) * increment

    Use for:
    - Gradual backpressure
    - Resource contention scenarios

    Example:
        backoff = LinearBackoff(initial=1.0, increment=0.5, max_delay=10.0)
        # Attempt 1: 1.0s, Attempt 2: 1.5s, Attempt 3: 2.0s, ...
    """

    initial: float = 1.0
    increment: float = 0.5
    max_delay: float = 60.0

    def get_delay(self, attempt: int) -> float:
        delay = self.initial + (attempt - 1) * self.increment
        return min(delay, self.max_delay)

    def reset(self) -> None:
        pass


@dataclass
class ExponentialBackoff(BackoffStrategy):
    """
    Exponentially increasing delay between retries.

    delay = base * (multiplier ^ (attempt - 1))

    With optional jitter to prevent thundering herd.

    Use for:
    - External API calls
    - Network operations
    - Database connections

    Example:
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=30.0)
        # Attempt 1: 1s, Attempt 2: 2s, Attempt 3: 4s, Attempt 4: 8s, ...
    """

    base: float = 1.0
    multiplier: float = 2.0
    max_delay: float = 60.0
    jitter: bool = True
    jitter_factor: float = 0.25  # +/- 25%

    def get_delay(self, attempt: int) -> float:
        delay = self.base * (self.multiplier ** (attempt - 1))
        delay = min(delay, self.max_delay)

        if self.jitter:
            jitter_range = delay * self.jitter_factor
            delay += random.uniform(-jitter_range, jitter_range)
            delay = max(0, delay)  # Ensure non-negative

        return delay

    def reset(self) -> None:
        pass


@dataclass
class DecorrelatedJitter(BackoffStrategy):
    """
    AWS-style decorrelated jitter backoff.

    delay = random(base, previous_delay * 3)

    This provides better spread than simple exponential with jitter.

    Use for:
    - High-concurrency scenarios
    - Preventing correlated retries

    Reference: https://aws.amazon.com/blogs/architecture/exponential-backoff-and-jitter/
    """

    base: float = 1.0
    max_delay: float = 60.0
    _previous_delay: float = field(default=0.0, init=False)

    def get_delay(self, attempt: int) -> float:
        if attempt == 1:
            self._previous_delay = self.base
        else:
            upper = min(self._previous_delay * 3, self.max_delay)
            self._previous_delay = random.uniform(self.base, upper)

        return self._previous_delay

    def reset(self) -> None:
        self._previous_delay = 0.0


# =============================================================================
# Retry Policy
# =============================================================================


@dataclass
class RetryPolicy:
    """
    Configures retry behavior for a processor or operation.

    Determines:
    - How many times to retry
    - Which exceptions trigger retry
    - How long to wait between retries

    Example:
        policy = RetryPolicy(
            max_attempts=3,
            backoff=ExponentialBackoff(base=1.0),
            retry_on=(ConnectionError, TimeoutError),
        )
    """

    max_attempts: int = 1  # 1 = no retry (single attempt)
    backoff: BackoffStrategy = field(default_factory=NoBackoff)
    retry_on: tuple[type[Exception], ...] = (Exception,)
    retry_on_result: Callable[[Any], bool] | None = None  # Retry if returns True

    def should_retry(self, attempt: int, error: Exception | None = None) -> bool:
        """
        Determine if retry should be attempted.

        Args:
            attempt: Current attempt number (1-indexed)
            error: Exception that caused failure (if any)

        Returns:
            True if should retry, False otherwise
        """
        if attempt >= self.max_attempts:
            return False

        if error is not None:
            return isinstance(error, self.retry_on)

        return False

    def get_delay(self, attempt: int) -> float:
        """Get delay before next retry attempt."""
        return self.backoff.get_delay(attempt)

    def reset(self) -> None:
        """Reset backoff state for reuse."""
        self.backoff.reset()


# Default policies for common scenarios
NO_RETRY = RetryPolicy(max_attempts=1)

RETRY_ONCE = RetryPolicy(
    max_attempts=2,
    backoff=ConstantBackoff(delay=0.5),
)

RETRY_WITH_BACKOFF = RetryPolicy(
    max_attempts=3,
    backoff=ExponentialBackoff(base=1.0, multiplier=2.0, max_delay=10.0),
)

AGGRESSIVE_RETRY = RetryPolicy(
    max_attempts=5,
    backoff=ExponentialBackoff(base=0.5, multiplier=2.0, max_delay=30.0, jitter=True),
)


# =============================================================================
# Retry Executor
# =============================================================================


@dataclass
class RetryResult:
    """Result of a retry-wrapped operation."""

    success: bool
    result: Any = None
    attempts: int = 0
    total_delay: float = 0.0
    errors: list[Exception] = field(default_factory=list)

    @property
    def final_error(self) -> Exception | None:
        """Get the last error encountered."""
        return self.errors[-1] if self.errors else None


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    operation_name: str = "operation",
) -> RetryResult:
    """
    Execute an async operation with retry logic.

    Args:
        operation: Async callable to execute
        policy: Retry policy to apply
        operation_name: Name for logging

    Returns:
        RetryResult with success status and result/errors

    Example:
        async def fetch_data():
            return await api.get("/data")

        result = await with_retry(
            fetch_data,
            policy=RETRY_WITH_BACKOFF,
            operation_name="fetch_data",
        )

        if result.success:
            data = result.result
        else:
            logger.error(f"Failed after {result.attempts} attempts")
    """
    policy.reset()
    errors: list[Exception] = []
    total_delay = 0.0
    attempt = 0

    while True:
        attempt += 1

        try:
            result = await operation()

            # Check if result should trigger retry
            if (
                policy.retry_on_result
                and policy.retry_on_result(result)
                and attempt < policy.max_attempts
            ):
                delay = policy.get_delay(attempt)
                total_delay += delay
                logger.warning(
                    f"{operation_name}: Result triggered retry "
                    f"(attempt {attempt}/{policy.max_attempts}), "
                    f"waiting {delay:.2f}s"
                )
                await asyncio.sleep(delay)
                continue

            return RetryResult(
                success=True,
                result=result,
                attempts=attempt,
                total_delay=total_delay,
                errors=errors,
            )

        except Exception as e:
            errors.append(e)

            if policy.should_retry(attempt, e):
                delay = policy.get_delay(attempt)
                total_delay += delay
                logger.warning(
                    f"{operation_name}: Attempt {attempt}/{policy.max_attempts} "
                    f"failed with {type(e).__name__}: {e}, "
                    f"retrying in {delay:.2f}s"
                )
                await asyncio.sleep(delay)
            else:
                logger.error(
                    f"{operation_name}: Failed after {attempt} attempts, " f"last error: {e}"
                )
                return RetryResult(
                    success=False,
                    result=None,
                    attempts=attempt,
                    total_delay=total_delay,
                    errors=errors,
                )


# =============================================================================
# Circuit Breaker
# =============================================================================


class CircuitState(Enum):
    """Circuit breaker states."""

    CLOSED = "closed"  # Normal operation
    OPEN = "open"  # Failing, reject requests
    HALF_OPEN = "half_open"  # Testing recovery


class CircuitOpenError(Exception):
    """Raised when circuit breaker is open."""

    def __init__(self, circuit_name: str, reset_after: float):
        self.circuit_name = circuit_name
        self.reset_after = reset_after
        super().__init__(
            f"Circuit '{circuit_name}' is open, " f"will attempt reset in {reset_after:.1f}s"
        )


@dataclass
class CircuitBreaker:
    """
    Circuit breaker pattern for protecting external services.

    States:
    - CLOSED: Normal operation, requests pass through
    - OPEN: Service is failing, requests rejected immediately
    - HALF_OPEN: Testing if service recovered

    Example:
        breaker = CircuitBreaker(
            name="external_api",
            failure_threshold=5,
            recovery_timeout=30.0,
        )

        async with breaker:
            await external_api.call()

    Or with explicit call:
        result = await breaker.call(external_api.call)
    """

    name: str = "circuit"
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: float = 30.0  # Seconds before half-open
    half_open_max_calls: int = 1  # Test calls in half-open

    # Internal state (mutable)
    _state: CircuitState = field(default=CircuitState.CLOSED, init=False)
    _failure_count: int = field(default=0, init=False)
    _success_count: int = field(default=0, init=False)
    _last_failure_time: float | None = field(default=None, init=False)
    _half_open_calls: int = field(default=0, init=False)

    @property
    def state(self) -> CircuitState:
        """Get current circuit state (may transition to half-open)."""
        if self._state == CircuitState.OPEN and self._should_attempt_reset():
            self._transition_to_half_open()
        return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt recovery."""
        if self._last_failure_time is None:
            return False
        elapsed = time.monotonic() - self._last_failure_time
        return elapsed >= self.recovery_timeout

    def _transition_to_half_open(self) -> None:
        """Transition to half-open state."""
        logger.info(f"Circuit '{self.name}': OPEN -> HALF_OPEN")
        self._state = CircuitState.HALF_OPEN
        self._half_open_calls = 0

    def _transition_to_open(self) -> None:
        """Transition to open state."""
        logger.warning(
            f"Circuit '{self.name}': {self._state.value} -> OPEN "
            f"(failures={self._failure_count})"
        )
        self._state = CircuitState.OPEN
        self._last_failure_time = time.monotonic()

    def _transition_to_closed(self) -> None:
        """Transition to closed state."""
        logger.info(f"Circuit '{self.name}': HALF_OPEN -> CLOSED")
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_calls = 0

    def record_success(self) -> None:
        """Record a successful call."""
        if self._state == CircuitState.HALF_OPEN:
            self._success_count += 1
            if self._success_count >= self.half_open_max_calls:
                self._transition_to_closed()
        elif self._state == CircuitState.CLOSED:
            # Reset failure count on success
            self._failure_count = 0

    def record_failure(self) -> None:
        """Record a failed call."""
        self._failure_count += 1
        self._last_failure_time = time.monotonic()

        if self._state == CircuitState.HALF_OPEN:
            # Any failure in half-open reopens circuit
            self._transition_to_open()
        elif self._state == CircuitState.CLOSED and self._failure_count >= self.failure_threshold:
            self._transition_to_open()

    async def call(
        self,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        """
        Execute operation through circuit breaker.

        Raises:
            CircuitOpenError: If circuit is open
            Exception: Any exception from the operation

        Example:
            result = await breaker.call(api.fetch_data)
        """
        state = self.state  # May transition to half-open

        if state == CircuitState.OPEN:
            time_since_failure = time.monotonic() - (self._last_failure_time or 0)
            reset_after = max(0, self.recovery_timeout - time_since_failure)
            raise CircuitOpenError(self.name, reset_after)

        if state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls > self.half_open_max_calls:
                raise CircuitOpenError(self.name, self.recovery_timeout)

        try:
            result = await operation()
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise

    async def __aenter__(self) -> CircuitBreaker:
        """Async context manager entry."""
        state = self.state

        if state == CircuitState.OPEN:
            time_since_failure = time.monotonic() - (self._last_failure_time or 0)
            reset_after = max(0, self.recovery_timeout - time_since_failure)
            raise CircuitOpenError(self.name, reset_after)

        if state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            if self._half_open_calls > self.half_open_max_calls:
                raise CircuitOpenError(self.name, self.recovery_timeout)

        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        if exc_type is None:
            self.record_success()
        else:
            self.record_failure()

    def reset(self) -> None:
        """Reset circuit breaker to initial state."""
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time = None
        self._half_open_calls = 0

    def get_stats(self) -> dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            "name": self.name,
            "state": self._state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout": self.recovery_timeout,
            "time_since_last_failure": (
                time.monotonic() - self._last_failure_time if self._last_failure_time else None
            ),
        }


# =============================================================================
# Resilient Processor Wrapper
# =============================================================================


@dataclass
class ResilientProcessor:
    """
    Wraps a processor with retry and circuit breaker support.

    Example:
        resilient = ResilientProcessor(
            processor=ExternalAPIProcessor(),
            retry_policy=RETRY_WITH_BACKOFF,
            circuit_breaker=CircuitBreaker(name="api", failure_threshold=3),
        )

        pipeline = Pipeline([
            QuickProcessor(),
            resilient,
            NextProcessor(),
        ])
    """

    processor: Any  # Processor instance
    retry_policy: RetryPolicy = field(default_factory=lambda: NO_RETRY)
    circuit_breaker: CircuitBreaker | None = None
    critical: bool = True  # If False, errors don't stop pipeline
    fallback: Callable[[Frame, Exception], Frame] | None = None

    @property
    def name(self) -> str:
        return f"resilient_{self.processor.name}"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | None:
        """Process with retry and circuit breaker."""

        async def do_process() -> Frame | None:
            if self.circuit_breaker:
                return await self.circuit_breaker.call(lambda: self.processor.process(frame, ctx))
            return await self.processor.process(frame, ctx)

        try:
            result = await with_retry(
                do_process,
                self.retry_policy,
                operation_name=self.processor.name,
            )

            if result.success:
                return result.result
            else:
                # All retries failed
                error = result.final_error
                if error:
                    raise error
                raise RuntimeError("Retry failed without error")

        except CircuitOpenError as e:
            logger.warning(f"Circuit breaker open for {self.processor.name}: {e}")
            if self.fallback:
                return self.fallback(frame, e)
            if not self.critical:
                return frame  # Pass through on non-critical
            raise

        except Exception as e:
            if self.fallback:
                return self.fallback(frame, e)
            if not self.critical:
                logger.warning(
                    f"Non-critical processor {self.processor.name} failed, " f"continuing: {e}"
                )
                return frame
            raise


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "AGGRESSIVE_RETRY",
    # Default policies
    "NO_RETRY",
    "RETRY_ONCE",
    "RETRY_WITH_BACKOFF",
    # Backoff strategies
    "BackoffStrategy",
    "CircuitBreaker",
    "CircuitOpenError",
    # Circuit breaker
    "CircuitState",
    "ConstantBackoff",
    "DecorrelatedJitter",
    "ExponentialBackoff",
    "LinearBackoff",
    "NoBackoff",
    # Wrapper
    "ResilientProcessor",
    # Retry
    "RetryPolicy",
    "RetryResult",
    "with_retry",
]
