"""
Rate Limiting for Knomly Pipeline.

Provides mechanisms for controlling request rates to protect
external services from abuse and quota exhaustion.

Design Philosophy:
- Token bucket algorithm (industry standard)
- Per-key rate limiting (user, provider, endpoint)
- Pluggable storage backends
- Non-blocking where possible

See ADR-001 for design decisions.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, ParamSpec, Protocol, TypeVar

P = ParamSpec("P")
T = TypeVar("T")

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RateLimitExceeded(Exception):
    """Raised when rate limit is exceeded and blocking is disabled."""

    def __init__(
        self,
        key: str,
        limit: float,
        window: float,
        retry_after: float | None = None,
    ):
        self.key = key
        self.limit = limit
        self.window = window
        self.retry_after = retry_after
        super().__init__(
            f"Rate limit exceeded for key '{key}': "
            f"{limit} requests per {window}s"
            + (f", retry after {retry_after:.2f}s" if retry_after else "")
        )


# =============================================================================
# Token Bucket State
# =============================================================================


@dataclass
class TokenBucket:
    """
    Token bucket state for rate limiting.

    The bucket fills at a constant rate and has a maximum capacity.
    Each request consumes tokens from the bucket.
    """

    tokens: float
    last_update: float
    capacity: float
    rate: float  # tokens per second

    def replenish(self) -> None:
        """Replenish tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
        self.last_update = now

    def consume(self, cost: float = 1.0) -> bool:
        """
        Attempt to consume tokens.

        Returns True if tokens were consumed, False if insufficient.
        """
        self.replenish()
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

    def time_until_available(self, cost: float = 1.0) -> float:
        """Calculate time until enough tokens are available."""
        self.replenish()
        if self.tokens >= cost:
            return 0.0
        needed = cost - self.tokens
        return needed / self.rate


# =============================================================================
# Rate Limit Storage Protocol
# =============================================================================


class RateLimitStorage(Protocol):
    """
    Protocol for rate limit state storage.

    Implementations can use:
    - In-memory (single process)
    - Redis (distributed)
    - Memcached (distributed)
    """

    async def get(self, key: str) -> TokenBucket | None:
        """Get bucket state for key."""
        ...

    async def set(self, key: str, bucket: TokenBucket) -> None:
        """Save bucket state for key."""
        ...


# =============================================================================
# In-Memory Storage
# =============================================================================


@dataclass
class InMemoryStorage:
    """
    In-memory rate limit storage.

    Suitable for single-process deployments.
    Not suitable for distributed systems.
    """

    buckets: dict[str, TokenBucket] = field(default_factory=dict)
    max_keys: int = 10000

    async def get(self, key: str) -> TokenBucket | None:
        return self.buckets.get(key)

    async def set(self, key: str, bucket: TokenBucket) -> None:
        self.buckets[key] = bucket

        # Cleanup old entries if too many
        if len(self.buckets) > self.max_keys:
            self._cleanup()

    def _cleanup(self) -> None:
        """Remove oldest entries."""
        if not self.buckets:
            return

        # Sort by last_update and keep newest half
        sorted_keys = sorted(
            self.buckets.keys(),
            key=lambda k: self.buckets[k].last_update,
        )
        for key in sorted_keys[: len(sorted_keys) // 2]:
            del self.buckets[key]


# =============================================================================
# Rate Limiter
# =============================================================================


@dataclass
class RateLimiter:
    """
    Token bucket rate limiter.

    Controls request rates using the token bucket algorithm.

    Example:
        limiter = RateLimiter(
            rate=10.0,      # 10 requests per second
            capacity=20,    # Burst capacity of 20
        )

        # Non-blocking check
        if await limiter.acquire("user:123"):
            # Proceed with request
        else:
            raise RateLimitExceeded(...)

        # Blocking wait
        await limiter.wait("user:123")
        # Request is now allowed

    Args:
        rate: Tokens replenished per second
        capacity: Maximum tokens (burst capacity)
        storage: Storage backend for bucket state
    """

    rate: float  # Tokens per second
    capacity: int  # Maximum tokens (burst)
    storage: RateLimitStorage = field(default_factory=InMemoryStorage)

    async def _get_or_create_bucket(self, key: str) -> TokenBucket:
        """Get existing bucket or create new one."""
        bucket = await self.storage.get(key)
        if bucket is None:
            bucket = TokenBucket(
                tokens=float(self.capacity),
                last_update=time.monotonic(),
                capacity=float(self.capacity),
                rate=self.rate,
            )
        return bucket

    async def acquire(self, key: str, cost: float = 1.0) -> bool:
        """
        Attempt to acquire rate limit tokens.

        Non-blocking: returns immediately with success/failure.

        Args:
            key: Rate limit key (e.g., "user:123", "api:transcribe")
            cost: Number of tokens to consume

        Returns:
            True if tokens acquired, False if rate limited
        """
        bucket = await self._get_or_create_bucket(key)

        if bucket.consume(cost):
            await self.storage.set(key, bucket)
            logger.debug(
                f"Rate limit acquired: key={key}, cost={cost}, " f"remaining={bucket.tokens:.1f}"
            )
            return True
        else:
            logger.debug(
                f"Rate limit exceeded: key={key}, cost={cost}, " f"tokens={bucket.tokens:.1f}"
            )
            return False

    async def wait(self, key: str, cost: float = 1.0, timeout: float | None = None) -> bool:
        """
        Wait until rate limit allows the request.

        Blocking: waits until tokens are available.

        Args:
            key: Rate limit key
            cost: Number of tokens to consume
            timeout: Maximum time to wait (None = no limit)

        Returns:
            True if tokens acquired, False if timeout exceeded

        Raises:
            asyncio.TimeoutError: If timeout exceeded
        """
        start = time.monotonic()

        while True:
            bucket = await self._get_or_create_bucket(key)

            if bucket.consume(cost):
                await self.storage.set(key, bucket)
                return True

            # Calculate wait time
            wait_time = bucket.time_until_available(cost)

            # Check timeout
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed + wait_time > timeout:
                    return False

            logger.debug(f"Rate limit waiting: key={key}, wait={wait_time:.2f}s")
            await asyncio.sleep(min(wait_time, 1.0))  # Check every second max

    async def check(self, key: str, cost: float = 1.0) -> tuple[bool, float]:
        """
        Check if request would be allowed without consuming tokens.

        Args:
            key: Rate limit key
            cost: Number of tokens needed

        Returns:
            Tuple of (would_succeed, time_until_available)
        """
        bucket = await self._get_or_create_bucket(key)
        bucket.replenish()

        would_succeed = bucket.tokens >= cost
        wait_time = bucket.time_until_available(cost)

        return would_succeed, wait_time

    async def reset(self, key: str) -> None:
        """Reset rate limit for a key."""
        bucket = TokenBucket(
            tokens=float(self.capacity),
            last_update=time.monotonic(),
            capacity=float(self.capacity),
            rate=self.rate,
        )
        await self.storage.set(key, bucket)

    def get_config(self) -> dict[str, Any]:
        """Get rate limiter configuration."""
        return {
            "rate": self.rate,
            "capacity": self.capacity,
        }


# =============================================================================
# Sliding Window Rate Limiter
# =============================================================================


@dataclass
class SlidingWindowEntry:
    """Entry for sliding window rate limiting."""

    timestamps: list[float] = field(default_factory=list)


class SlidingWindowStorage(Protocol):
    """Storage protocol for sliding window limiter."""

    async def get(self, key: str) -> SlidingWindowEntry | None: ...

    async def set(self, key: str, entry: SlidingWindowEntry) -> None: ...


@dataclass
class InMemorySlidingWindowStorage:
    """In-memory storage for sliding window limiter."""

    entries: dict[str, SlidingWindowEntry] = field(default_factory=dict)

    async def get(self, key: str) -> SlidingWindowEntry | None:
        return self.entries.get(key)

    async def set(self, key: str, entry: SlidingWindowEntry) -> None:
        self.entries[key] = entry


@dataclass
class SlidingWindowLimiter:
    """
    Sliding window rate limiter.

    More accurate than token bucket for strict rate limits,
    but requires more memory.

    Example:
        limiter = SlidingWindowLimiter(
            max_requests=100,
            window_seconds=60.0,  # 100 requests per minute
        )
    """

    max_requests: int
    window_seconds: float
    storage: SlidingWindowStorage = field(default_factory=InMemorySlidingWindowStorage)

    async def _get_or_create_entry(self, key: str) -> SlidingWindowEntry:
        entry = await self.storage.get(key)
        if entry is None:
            entry = SlidingWindowEntry()
        return entry

    def _cleanup_old_timestamps(self, entry: SlidingWindowEntry) -> None:
        """Remove timestamps outside the window."""
        cutoff = time.monotonic() - self.window_seconds
        entry.timestamps = [t for t in entry.timestamps if t > cutoff]

    async def acquire(self, key: str) -> bool:
        """
        Attempt to acquire rate limit.

        Returns True if allowed, False if rate limited.
        """
        entry = await self._get_or_create_entry(key)
        self._cleanup_old_timestamps(entry)

        if len(entry.timestamps) < self.max_requests:
            entry.timestamps.append(time.monotonic())
            await self.storage.set(key, entry)
            return True

        return False

    async def wait(self, key: str, timeout: float | None = None) -> bool:
        """Wait until rate limit allows the request."""
        start = time.monotonic()

        while True:
            if await self.acquire(key):
                return True

            entry = await self._get_or_create_entry(key)
            self._cleanup_old_timestamps(entry)

            if not entry.timestamps:
                continue

            # Wait until oldest timestamp expires
            oldest = min(entry.timestamps)
            wait_time = (oldest + self.window_seconds) - time.monotonic()
            wait_time = max(0.01, wait_time)  # Minimum wait

            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed + wait_time > timeout:
                    return False

            await asyncio.sleep(min(wait_time, 1.0))

    async def check(self, key: str) -> tuple[bool, int]:
        """
        Check current state without consuming.

        Returns (would_succeed, remaining_requests)
        """
        entry = await self._get_or_create_entry(key)
        self._cleanup_old_timestamps(entry)

        remaining = self.max_requests - len(entry.timestamps)
        would_succeed = remaining > 0

        return would_succeed, remaining


# =============================================================================
# Rate Limited Decorator
# =============================================================================


def rate_limited(
    limiter: RateLimiter | SlidingWindowLimiter,
    key_func: Callable[..., str] | str,
    cost: float = 1.0,
    block: bool = True,
    timeout: float | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """
    Decorator for rate limiting async functions.

    Example:
        limiter = RateLimiter(rate=10, capacity=20)

        @rate_limited(limiter, key_func="api_calls", block=True)
        async def call_external_api():
            ...

        @rate_limited(limiter, key_func=lambda: f"user:{current_user.id}")
        async def user_action():
            ...
    """

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            # Resolve key
            key = key_func(*args, **kwargs) if callable(key_func) else key_func

            # Apply rate limit (handle different limiter signatures)
            if block:
                if isinstance(limiter, RateLimiter):
                    success = await limiter.wait(key, cost, timeout)
                else:
                    success = await limiter.wait(key, timeout)
                if not success:
                    raise RateLimitExceeded(
                        key=key,
                        limit=getattr(limiter, "rate", getattr(limiter, "max_requests", 0)),
                        window=getattr(limiter, "window_seconds", 1.0),
                    )
            else:
                if isinstance(limiter, RateLimiter):
                    success = await limiter.acquire(key, cost)
                else:
                    success = await limiter.acquire(key)
                if not success:
                    raise RateLimitExceeded(
                        key=key,
                        limit=getattr(limiter, "rate", getattr(limiter, "max_requests", 0)),
                        window=getattr(limiter, "window_seconds", 1.0),
                    )

            return await func(*args, **kwargs)

        return wrapper

    return decorator


# =============================================================================
# Composite Rate Limiter
# =============================================================================


@dataclass
class CompositeRateLimiter:
    """
    Combines multiple rate limiters.

    Useful for applying different limits:
    - Per-user limit
    - Per-endpoint limit
    - Global limit

    All limiters must allow for the request to proceed.

    Example:
        limiter = CompositeRateLimiter([
            (RateLimiter(rate=10, capacity=20), lambda r: f"user:{r.user_id}"),
            (RateLimiter(rate=100, capacity=200), lambda r: "global"),
        ])
    """

    limiters: list[tuple[RateLimiter | SlidingWindowLimiter, Callable[[Any], str]]]

    async def acquire(self, context: Any) -> bool:
        """
        Attempt to acquire from all limiters.

        Only succeeds if ALL limiters allow.
        """
        for limiter, key_func in self.limiters:
            key = key_func(context)
            if not await limiter.acquire(key):
                return False
        return True

    async def wait(self, context: Any, timeout: float | None = None) -> bool:
        """Wait for all limiters to allow."""
        for limiter, key_func in self.limiters:
            key = key_func(context)
            if not await limiter.wait(key, timeout=timeout):
                return False
        return True


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "CompositeRateLimiter",
    "InMemorySlidingWindowStorage",
    "InMemoryStorage",
    # Exceptions
    "RateLimitExceeded",
    "RateLimitStorage",
    # Rate limiters
    "RateLimiter",
    "SlidingWindowEntry",
    "SlidingWindowLimiter",
    "SlidingWindowStorage",
    # Core types
    "TokenBucket",
    # Decorator
    "rate_limited",
]
