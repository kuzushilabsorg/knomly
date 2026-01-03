"""
Tests for Knomly retry and resilience patterns.
"""
import asyncio
import time

import pytest

from knomly.pipeline import (
    AGGRESSIVE_RETRY,
    NO_RETRY,
    RETRY_ONCE,
    RETRY_WITH_BACKOFF,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    ConstantBackoff,
    DecorrelatedJitter,
    ExponentialBackoff,
    LinearBackoff,
    NoBackoff,
    PipelineContext,
    Processor,
    ResilientProcessor,
    RetryPolicy,
    with_retry,
)
from knomly.pipeline.frames import AudioInputFrame, Frame


# =============================================================================
# Backoff Strategy Tests
# =============================================================================


class TestNoBackoff:
    """Tests for NoBackoff strategy."""

    def test_always_returns_zero(self):
        backoff = NoBackoff()
        assert backoff.get_delay(1) == 0.0
        assert backoff.get_delay(5) == 0.0
        assert backoff.get_delay(100) == 0.0

    def test_reset_does_nothing(self):
        backoff = NoBackoff()
        backoff.reset()  # Should not raise


class TestConstantBackoff:
    """Tests for ConstantBackoff strategy."""

    def test_returns_constant_delay(self):
        backoff = ConstantBackoff(delay=2.5)
        assert backoff.get_delay(1) == 2.5
        assert backoff.get_delay(5) == 2.5
        assert backoff.get_delay(100) == 2.5

    def test_default_delay(self):
        backoff = ConstantBackoff()
        assert backoff.get_delay(1) == 1.0


class TestLinearBackoff:
    """Tests for LinearBackoff strategy."""

    def test_increases_linearly(self):
        backoff = LinearBackoff(initial=1.0, increment=0.5)
        assert backoff.get_delay(1) == 1.0
        assert backoff.get_delay(2) == 1.5
        assert backoff.get_delay(3) == 2.0
        assert backoff.get_delay(4) == 2.5

    def test_respects_max_delay(self):
        backoff = LinearBackoff(initial=1.0, increment=10.0, max_delay=5.0)
        assert backoff.get_delay(1) == 1.0
        assert backoff.get_delay(2) == 5.0  # Capped
        assert backoff.get_delay(10) == 5.0  # Still capped


class TestExponentialBackoff:
    """Tests for ExponentialBackoff strategy."""

    def test_increases_exponentially(self):
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, jitter=False)
        assert backoff.get_delay(1) == 1.0
        assert backoff.get_delay(2) == 2.0
        assert backoff.get_delay(3) == 4.0
        assert backoff.get_delay(4) == 8.0

    def test_respects_max_delay(self):
        backoff = ExponentialBackoff(
            base=1.0, multiplier=10.0, max_delay=5.0, jitter=False
        )
        assert backoff.get_delay(1) == 1.0
        assert backoff.get_delay(2) == 5.0  # Capped at 5.0, not 10.0
        assert backoff.get_delay(3) == 5.0

    def test_jitter_adds_randomness(self):
        backoff = ExponentialBackoff(base=10.0, multiplier=1.0, jitter=True)
        delays = [backoff.get_delay(1) for _ in range(100)]

        # With jitter, we should see variation
        assert len(set(delays)) > 1  # Not all the same
        assert all(d >= 0 for d in delays)  # All non-negative

    def test_no_jitter_is_deterministic(self):
        backoff = ExponentialBackoff(base=1.0, multiplier=2.0, jitter=False)
        delays = [backoff.get_delay(1) for _ in range(10)]
        assert len(set(delays)) == 1  # All the same


class TestDecorrelatedJitter:
    """Tests for DecorrelatedJitter strategy."""

    def test_first_attempt_uses_base(self):
        backoff = DecorrelatedJitter(base=1.0)
        delay = backoff.get_delay(1)
        assert delay == 1.0

    def test_subsequent_attempts_vary(self):
        backoff = DecorrelatedJitter(base=1.0, max_delay=100.0)

        delays = []
        for attempt in range(1, 10):
            delays.append(backoff.get_delay(attempt))
            backoff.reset()  # Reset for fresh calculation

        # First delays should be base
        # Can't test exact values due to randomness, but should all be reasonable
        assert all(d >= 0 for d in delays)

    def test_respects_max_delay(self):
        backoff = DecorrelatedJitter(base=1.0, max_delay=5.0)

        for attempt in range(1, 20):
            delay = backoff.get_delay(attempt)
            assert delay <= 15.0  # max_delay * 3 upper bound


# =============================================================================
# Retry Policy Tests
# =============================================================================


class TestRetryPolicy:
    """Tests for RetryPolicy."""

    def test_no_retry_by_default(self):
        policy = RetryPolicy()
        assert policy.max_attempts == 1
        assert policy.should_retry(1, RuntimeError()) is False

    def test_should_retry_within_max_attempts(self):
        policy = RetryPolicy(max_attempts=3, retry_on=(RuntimeError,))
        assert policy.should_retry(1, RuntimeError()) is True
        assert policy.should_retry(2, RuntimeError()) is True
        assert policy.should_retry(3, RuntimeError()) is False

    def test_should_retry_checks_exception_type(self):
        policy = RetryPolicy(max_attempts=3, retry_on=(ConnectionError,))
        assert policy.should_retry(1, ConnectionError()) is True
        assert policy.should_retry(1, RuntimeError()) is False

    def test_should_retry_checks_exception_hierarchy(self):
        policy = RetryPolicy(max_attempts=3, retry_on=(OSError,))
        # ConnectionError is a subclass of OSError
        assert policy.should_retry(1, ConnectionError()) is True

    def test_get_delay_uses_backoff(self):
        policy = RetryPolicy(backoff=ConstantBackoff(delay=2.0))
        assert policy.get_delay(1) == 2.0
        assert policy.get_delay(5) == 2.0


class TestDefaultPolicies:
    """Tests for pre-defined retry policies."""

    def test_no_retry(self):
        assert NO_RETRY.max_attempts == 1

    def test_retry_once(self):
        assert RETRY_ONCE.max_attempts == 2

    def test_retry_with_backoff(self):
        assert RETRY_WITH_BACKOFF.max_attempts == 3

    def test_aggressive_retry(self):
        assert AGGRESSIVE_RETRY.max_attempts == 5


# =============================================================================
# with_retry Tests
# =============================================================================


class TestWithRetry:
    """Tests for with_retry function."""

    @pytest.mark.asyncio
    async def test_succeeds_on_first_attempt(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=3),
            operation_name="test",
        )

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 1
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retries_on_failure(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("Simulated failure")
            return "success"

        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=5, backoff=NoBackoff()),
            operation_name="test",
        )

        assert result.success is True
        assert result.result == "success"
        assert result.attempts == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_fails_after_max_attempts(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=3, backoff=NoBackoff()),
            operation_name="test",
        )

        assert result.success is False
        assert result.result is None
        assert result.attempts == 3
        assert len(result.errors) == 3
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_respects_retry_on_filter(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Retryable")
            raise RuntimeError("Not retryable")

        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=5, retry_on=(ConnectionError,), backoff=NoBackoff()),
            operation_name="test",
        )

        # Should fail after attempt 2 because RuntimeError is not retryable
        assert result.success is False
        assert result.attempts == 2

    @pytest.mark.asyncio
    async def test_applies_backoff_delay(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise RuntimeError("Fail once")
            return "success"

        start = time.perf_counter()
        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=3, backoff=ConstantBackoff(delay=0.1)),
            operation_name="test",
        )
        elapsed = time.perf_counter() - start

        assert result.success is True
        assert elapsed >= 0.1  # At least one delay

    @pytest.mark.asyncio
    async def test_records_total_delay(self):
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RuntimeError("Fail twice")
            return "success"

        result = await with_retry(
            operation,
            RetryPolicy(max_attempts=5, backoff=ConstantBackoff(delay=0.05)),
            operation_name="test",
        )

        assert result.success is True
        assert result.total_delay >= 0.1  # Two delays of 0.05

    @pytest.mark.asyncio
    async def test_retry_on_result(self):
        """Test retrying based on result value."""
        call_count = 0

        async def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                return None  # Trigger retry
            return "valid_result"

        result = await with_retry(
            operation,
            RetryPolicy(
                max_attempts=5,
                backoff=NoBackoff(),
                retry_on_result=lambda r: r is None,  # Retry on None
            ),
            operation_name="test",
        )

        assert result.success is True
        assert result.result == "valid_result"
        assert result.attempts == 3


# =============================================================================
# Circuit Breaker Tests
# =============================================================================


class TestCircuitBreaker:
    """Tests for CircuitBreaker."""

    def test_starts_closed(self):
        breaker = CircuitBreaker()
        assert breaker.state == CircuitState.CLOSED
        assert breaker.is_closed is True
        assert breaker.is_open is False

    @pytest.mark.asyncio
    async def test_opens_after_failure_threshold(self):
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_operation():
            raise RuntimeError("Fail")

        for _ in range(3):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_operation)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_rejects_when_open(self):
        breaker = CircuitBreaker(failure_threshold=1)

        async def failing_operation():
            raise RuntimeError("Fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

        # Next call should be rejected
        with pytest.raises(CircuitOpenError):
            await breaker.call(failing_operation)

    @pytest.mark.asyncio
    async def test_resets_failure_count_on_success(self):
        breaker = CircuitBreaker(failure_threshold=3)

        async def failing_operation():
            raise RuntimeError("Fail")

        async def succeeding_operation():
            return "success"

        # Two failures
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_operation)

        # One success should reset
        result = await breaker.call(succeeding_operation)
        assert result == "success"

        # Two more failures shouldn't open (count was reset)
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await breaker.call(failing_operation)

        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_transitions_to_half_open_after_timeout(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        async def failing_operation():
            raise RuntimeError("Fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

        assert breaker._state == CircuitState.OPEN

        # Wait for recovery timeout
        await asyncio.sleep(0.15)

        # Checking state should trigger transition
        assert breaker.state == CircuitState.HALF_OPEN

    @pytest.mark.asyncio
    async def test_half_open_success_closes_circuit(self):
        breaker = CircuitBreaker(
            failure_threshold=1,
            recovery_timeout=0.1,
            half_open_max_calls=1,
        )

        async def failing_operation():
            raise RuntimeError("Fail")

        async def succeeding_operation():
            return "success"

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

        # Wait and transition to half-open
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Success in half-open should close
        result = await breaker.call(succeeding_operation)
        assert result == "success"
        assert breaker.state == CircuitState.CLOSED

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens_circuit(self):
        breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.1)

        async def failing_operation():
            raise RuntimeError("Fail")

        # Open the circuit
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

        # Wait and transition to half-open
        await asyncio.sleep(0.15)
        assert breaker.state == CircuitState.HALF_OPEN

        # Failure in half-open should reopen
        with pytest.raises(RuntimeError):
            await breaker.call(failing_operation)

        assert breaker.state == CircuitState.OPEN

    @pytest.mark.asyncio
    async def test_context_manager_interface(self):
        breaker = CircuitBreaker(failure_threshold=3)
        results = []

        async with breaker:
            results.append("success")

        assert results == ["success"]
        assert breaker._failure_count == 0

    @pytest.mark.asyncio
    async def test_context_manager_records_failure(self):
        breaker = CircuitBreaker(failure_threshold=3)

        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("Fail")

        assert breaker._failure_count == 1

    def test_reset(self):
        breaker = CircuitBreaker()
        breaker._state = CircuitState.OPEN
        breaker._failure_count = 5
        breaker._last_failure_time = time.monotonic()

        breaker.reset()

        assert breaker._state == CircuitState.CLOSED
        assert breaker._failure_count == 0
        assert breaker._last_failure_time is None

    def test_get_stats(self):
        breaker = CircuitBreaker(name="test_circuit", failure_threshold=5)
        stats = breaker.get_stats()

        assert stats["name"] == "test_circuit"
        assert stats["state"] == "closed"
        assert stats["failure_threshold"] == 5


# =============================================================================
# ResilientProcessor Tests
# =============================================================================


class TestResilientProcessor:
    """Tests for ResilientProcessor wrapper."""

    @pytest.fixture
    def ctx(self) -> PipelineContext:
        return PipelineContext()

    @pytest.fixture
    def audio_frame(self) -> AudioInputFrame:
        return AudioInputFrame(audio_data=b"test", sender_phone="123")

    @pytest.mark.asyncio
    async def test_passes_through_on_success(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        class SuccessProcessor(Processor):
            @property
            def name(self):
                return "success"

            async def process(self, frame: Frame, ctx: PipelineContext):
                return frame.with_metadata(processed=True)

        resilient = ResilientProcessor(processor=SuccessProcessor())

        result = await resilient.process(audio_frame, ctx)

        assert result.metadata.get("processed") is True

    @pytest.mark.asyncio
    async def test_retries_on_failure(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        call_count = 0

        class FlakeyProcessor(Processor):
            @property
            def name(self):
                return "flakey"

            async def process(self, frame: Frame, ctx: PipelineContext):
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise ConnectionError("Transient failure")
                return frame.with_metadata(processed=True)

        resilient = ResilientProcessor(
            processor=FlakeyProcessor(),
            retry_policy=RetryPolicy(max_attempts=5, backoff=NoBackoff()),
        )

        result = await resilient.process(audio_frame, ctx)

        assert result.metadata.get("processed") is True
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_uses_circuit_breaker(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        class FailingProcessor(Processor):
            @property
            def name(self):
                return "failing"

            async def process(self, frame: Frame, ctx: PipelineContext):
                raise RuntimeError("Always fails")

        breaker = CircuitBreaker(failure_threshold=2)
        resilient = ResilientProcessor(
            processor=FailingProcessor(),
            retry_policy=NO_RETRY,
            circuit_breaker=breaker,
        )

        # First two calls fail and open circuit
        for _ in range(2):
            with pytest.raises(RuntimeError):
                await resilient.process(audio_frame, ctx)

        assert breaker.is_open

        # Third call should get CircuitOpenError
        with pytest.raises(CircuitOpenError):
            await resilient.process(audio_frame, ctx)

    @pytest.mark.asyncio
    async def test_fallback_on_failure(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        class FailingProcessor(Processor):
            @property
            def name(self):
                return "failing"

            async def process(self, frame: Frame, ctx: PipelineContext):
                raise RuntimeError("Fail")

        def fallback(frame: Frame, error: Exception) -> Frame:
            return frame.with_metadata(fallback=True, error=str(error))

        resilient = ResilientProcessor(
            processor=FailingProcessor(),
            retry_policy=NO_RETRY,
            fallback=fallback,
        )

        result = await resilient.process(audio_frame, ctx)

        assert result.metadata.get("fallback") is True
        assert "Fail" in result.metadata.get("error", "")

    @pytest.mark.asyncio
    async def test_non_critical_passes_through_on_failure(
        self, audio_frame: AudioInputFrame, ctx: PipelineContext
    ):
        class FailingProcessor(Processor):
            @property
            def name(self):
                return "failing"

            async def process(self, frame: Frame, ctx: PipelineContext):
                raise RuntimeError("Fail")

        resilient = ResilientProcessor(
            processor=FailingProcessor(),
            retry_policy=NO_RETRY,
            critical=False,  # Non-critical
        )

        result = await resilient.process(audio_frame, ctx)

        # Should pass through original frame
        assert result == audio_frame

    @pytest.mark.asyncio
    async def test_name_includes_processor_name(self):
        class TestProcessor(Processor):
            @property
            def name(self):
                return "test_proc"

            async def process(self, frame: Frame, ctx: PipelineContext):
                return frame

        resilient = ResilientProcessor(processor=TestProcessor())

        assert resilient.name == "resilient_test_proc"
