"""
Tests for Knomly rate limiting module.
"""
import asyncio
import time

import pytest

from knomly.pipeline import (
    CompositeRateLimiter,
    InMemoryStorage,
    RateLimitExceeded,
    RateLimiter,
    SlidingWindowLimiter,
    TokenBucket,
    rate_limited,
)


# =============================================================================
# TokenBucket Tests
# =============================================================================


class TestTokenBucket:
    """Tests for TokenBucket."""

    def test_initial_state(self):
        bucket = TokenBucket(
            tokens=10.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=1.0,
        )

        assert bucket.tokens == 10.0
        assert bucket.capacity == 10.0

    def test_consume_success(self):
        bucket = TokenBucket(
            tokens=10.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=1.0,
        )

        assert bucket.consume(5.0) is True
        assert bucket.tokens == 5.0

    def test_consume_failure(self):
        bucket = TokenBucket(
            tokens=5.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=1.0,
        )

        assert bucket.consume(10.0) is False
        # Tokens should remain approximately unchanged (may have tiny replenish)
        assert bucket.tokens >= 5.0
        assert bucket.tokens < 6.0  # Should not have gained much

    def test_replenish(self):
        bucket = TokenBucket(
            tokens=0.0,
            last_update=time.monotonic() - 5.0,  # 5 seconds ago
            capacity=10.0,
            rate=1.0,
        )

        bucket.replenish()

        # Should have gained ~5 tokens
        assert bucket.tokens >= 4.5  # Allow for timing variance
        assert bucket.tokens <= 10.0  # Capped at capacity

    def test_replenish_caps_at_capacity(self):
        bucket = TokenBucket(
            tokens=0.0,
            last_update=time.monotonic() - 100.0,  # 100 seconds ago
            capacity=10.0,
            rate=1.0,
        )

        bucket.replenish()

        assert bucket.tokens == 10.0  # Capped

    def test_time_until_available(self):
        bucket = TokenBucket(
            tokens=0.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=2.0,  # 2 tokens per second
        )

        wait_time = bucket.time_until_available(4.0)

        assert wait_time == pytest.approx(2.0, rel=0.1)  # 4 tokens / 2 per sec

    def test_time_until_available_when_sufficient(self):
        bucket = TokenBucket(
            tokens=10.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=1.0,
        )

        wait_time = bucket.time_until_available(5.0)

        assert wait_time == 0.0


# =============================================================================
# InMemoryStorage Tests
# =============================================================================


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_get_nonexistent_returns_none(self):
        storage = InMemoryStorage()

        result = await storage.get("nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self):
        storage = InMemoryStorage()
        bucket = TokenBucket(
            tokens=10.0,
            last_update=time.monotonic(),
            capacity=10.0,
            rate=1.0,
        )

        await storage.set("key", bucket)
        result = await storage.get("key")

        assert result is bucket

    @pytest.mark.asyncio
    async def test_cleanup_on_max_keys(self):
        storage = InMemoryStorage(max_keys=5)

        # Add more than max_keys
        for i in range(10):
            bucket = TokenBucket(
                tokens=10.0,
                last_update=time.monotonic() + i,  # Different times
                capacity=10.0,
                rate=1.0,
            )
            await storage.set(f"key_{i}", bucket)

        # Should have cleaned up
        assert len(storage.buckets) <= 5


# =============================================================================
# RateLimiter Tests
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    @pytest.fixture
    def limiter(self) -> RateLimiter:
        return RateLimiter(rate=10.0, capacity=10)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter: RateLimiter):
        success = await limiter.acquire("test_key")
        assert success is True

    @pytest.mark.asyncio
    async def test_acquire_exhausts_tokens(self, limiter: RateLimiter):
        # Exhaust all tokens
        for _ in range(10):
            success = await limiter.acquire("test_key")
            assert success is True

        # Next one should fail
        success = await limiter.acquire("test_key")
        assert success is False

    @pytest.mark.asyncio
    async def test_acquire_with_cost(self, limiter: RateLimiter):
        # Use 5 tokens
        success = await limiter.acquire("test_key", cost=5.0)
        assert success is True

        # Use 5 more
        success = await limiter.acquire("test_key", cost=5.0)
        assert success is True

        # Should fail (only 10 total)
        success = await limiter.acquire("test_key", cost=1.0)
        assert success is False

    @pytest.mark.asyncio
    async def test_different_keys_independent(self, limiter: RateLimiter):
        # Exhaust key1
        for _ in range(10):
            await limiter.acquire("key1")

        # key2 should still work
        success = await limiter.acquire("key2")
        assert success is True

    @pytest.mark.asyncio
    async def test_wait_success(self):
        limiter = RateLimiter(rate=100.0, capacity=1)  # Fast rate for testing

        # Exhaust
        await limiter.acquire("test_key")

        # Wait should succeed quickly due to high rate
        start = time.perf_counter()
        success = await limiter.wait("test_key", timeout=1.0)
        elapsed = time.perf_counter() - start

        assert success is True
        assert elapsed < 0.5  # Should be very fast

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        limiter = RateLimiter(rate=0.1, capacity=1)  # Very slow rate

        # Exhaust
        await limiter.acquire("test_key")

        # Wait should timeout
        start = time.perf_counter()
        success = await limiter.wait("test_key", timeout=0.1)
        elapsed = time.perf_counter() - start

        assert success is False
        assert elapsed < 0.5  # Should respect timeout

    @pytest.mark.asyncio
    async def test_check(self, limiter: RateLimiter):
        would_succeed, wait_time = await limiter.check("test_key")

        assert would_succeed is True
        assert wait_time == 0.0

    @pytest.mark.asyncio
    async def test_check_after_exhaustion(self, limiter: RateLimiter):
        # Exhaust
        for _ in range(10):
            await limiter.acquire("test_key")

        would_succeed, wait_time = await limiter.check("test_key")

        assert would_succeed is False
        assert wait_time > 0

    @pytest.mark.asyncio
    async def test_reset(self, limiter: RateLimiter):
        # Exhaust
        for _ in range(10):
            await limiter.acquire("test_key")

        # Reset
        await limiter.reset("test_key")

        # Should succeed now
        success = await limiter.acquire("test_key")
        assert success is True

    def test_get_config(self, limiter: RateLimiter):
        config = limiter.get_config()

        assert config["rate"] == 10.0
        assert config["capacity"] == 10


# =============================================================================
# SlidingWindowLimiter Tests
# =============================================================================


class TestSlidingWindowLimiter:
    """Tests for SlidingWindowLimiter."""

    @pytest.fixture
    def limiter(self) -> SlidingWindowLimiter:
        return SlidingWindowLimiter(max_requests=5, window_seconds=1.0)

    @pytest.mark.asyncio
    async def test_acquire_success(self, limiter: SlidingWindowLimiter):
        success = await limiter.acquire("test_key")
        assert success is True

    @pytest.mark.asyncio
    async def test_acquire_exhausts_limit(self, limiter: SlidingWindowLimiter):
        # Use all requests
        for _ in range(5):
            success = await limiter.acquire("test_key")
            assert success is True

        # Next should fail
        success = await limiter.acquire("test_key")
        assert success is False

    @pytest.mark.asyncio
    async def test_window_expiry(self):
        limiter = SlidingWindowLimiter(max_requests=2, window_seconds=0.1)

        # Exhaust
        await limiter.acquire("test_key")
        await limiter.acquire("test_key")

        # Should fail
        success = await limiter.acquire("test_key")
        assert success is False

        # Wait for window to expire
        await asyncio.sleep(0.15)

        # Should succeed now
        success = await limiter.acquire("test_key")
        assert success is True

    @pytest.mark.asyncio
    async def test_check(self, limiter: SlidingWindowLimiter):
        would_succeed, remaining = await limiter.check("test_key")

        assert would_succeed is True
        assert remaining == 5

    @pytest.mark.asyncio
    async def test_check_after_some_usage(self, limiter: SlidingWindowLimiter):
        await limiter.acquire("test_key")
        await limiter.acquire("test_key")

        would_succeed, remaining = await limiter.check("test_key")

        assert would_succeed is True
        assert remaining == 3

    @pytest.mark.asyncio
    async def test_wait_success(self):
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=0.1)

        # Exhaust
        await limiter.acquire("test_key")

        # Wait should succeed after window expires
        success = await limiter.wait("test_key", timeout=0.5)
        assert success is True

    @pytest.mark.asyncio
    async def test_wait_timeout(self):
        limiter = SlidingWindowLimiter(max_requests=1, window_seconds=10.0)

        # Exhaust
        await limiter.acquire("test_key")

        # Wait should timeout
        success = await limiter.wait("test_key", timeout=0.1)
        assert success is False


# =============================================================================
# CompositeRateLimiter Tests
# =============================================================================


class TestCompositeRateLimiter:
    """Tests for CompositeRateLimiter."""

    @pytest.mark.asyncio
    async def test_all_limiters_must_allow(self):
        limiter1 = RateLimiter(rate=10.0, capacity=10)
        limiter2 = RateLimiter(rate=10.0, capacity=5)

        composite = CompositeRateLimiter([
            (limiter1, lambda ctx: f"limiter1:{ctx}"),
            (limiter2, lambda ctx: f"limiter2:{ctx}"),
        ])

        # First 5 should succeed (limited by limiter2's capacity)
        for i in range(5):
            success = await composite.acquire(f"test")
            assert success is True, f"Request {i} should succeed"

        # Next should fail (limiter2 exhausted)
        success = await composite.acquire("test")
        assert success is False

    @pytest.mark.asyncio
    async def test_different_keys_per_limiter(self):
        limiter1 = RateLimiter(rate=10.0, capacity=2)
        limiter2 = RateLimiter(rate=10.0, capacity=10)

        composite = CompositeRateLimiter([
            (limiter1, lambda ctx: "global"),  # Shared key
            (limiter2, lambda ctx: f"user:{ctx}"),  # Per-context key
        ])

        # User A uses 2 requests
        await composite.acquire("user_a")
        await composite.acquire("user_a")

        # Global limit exhausted
        success = await composite.acquire("user_b")
        assert success is False


# =============================================================================
# rate_limited Decorator Tests
# =============================================================================


class TestRateLimitedDecorator:
    """Tests for rate_limited decorator."""

    @pytest.mark.asyncio
    async def test_decorator_allows_requests(self):
        limiter = RateLimiter(rate=10.0, capacity=10)
        call_count = 0

        @rate_limited(limiter, key_func="test_key", block=True, timeout=1.0)
        async def test_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await test_func()

        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_decorator_raises_on_limit(self):
        limiter = RateLimiter(rate=0.1, capacity=1)

        @rate_limited(limiter, key_func="test_key", block=False)
        async def test_func():
            return "success"

        # First call succeeds
        await test_func()

        # Second call should raise
        with pytest.raises(RateLimitExceeded):
            await test_func()

    @pytest.mark.asyncio
    async def test_decorator_with_callable_key(self):
        limiter = RateLimiter(rate=10.0, capacity=10)
        keys_used = []

        @rate_limited(limiter, key_func=lambda user_id: f"user:{user_id}")
        async def test_func(user_id: str):
            keys_used.append(user_id)
            return f"hello {user_id}"

        await test_func("alice")
        await test_func("bob")

        assert keys_used == ["alice", "bob"]


# =============================================================================
# Integration Tests
# =============================================================================


class TestRateLimitingIntegration:
    """Integration tests for rate limiting."""

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        limiter = RateLimiter(rate=100.0, capacity=5)

        async def make_request(request_id: int):
            success = await limiter.acquire("concurrent")
            return (request_id, success)

        # Make 10 concurrent requests
        results = await asyncio.gather(
            *[make_request(i) for i in range(10)]
        )

        successes = [r for r in results if r[1]]
        failures = [r for r in results if not r[1]]

        # Exactly 5 should succeed (capacity)
        assert len(successes) == 5
        assert len(failures) == 5

    @pytest.mark.asyncio
    async def test_rate_replenishment_over_time(self):
        limiter = RateLimiter(rate=10.0, capacity=2)  # 10 tokens/sec, 2 max

        # Exhaust capacity
        await limiter.acquire("test")
        await limiter.acquire("test")

        # Should fail
        success = await limiter.acquire("test")
        assert success is False

        # Wait for replenishment (0.1 sec = 1 token at 10/sec)
        await asyncio.sleep(0.15)

        # Should succeed
        success = await limiter.acquire("test")
        assert success is True
