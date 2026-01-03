"""
Provider Health Checks for Knomly.

Provides health verification and monitoring for providers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .chat.base import ChatProvider
    from .llm.base import LLMProvider
    from .stt.base import STTProvider

logger = logging.getLogger(__name__)


class HealthStatus(str, Enum):
    """Health status for a provider."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    provider_name: str
    provider_type: str  # stt, llm, chat
    status: HealthStatus
    latency_ms: float | None = None
    message: str | None = None
    error: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_type": self.provider_type,
            "status": self.status.value,
            "latency_ms": self.latency_ms,
            "message": self.message,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }


@dataclass
class ProviderMetrics:
    """Metrics for a provider."""

    provider_name: str
    provider_type: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency_ms: float = 0.0
    last_request_time: datetime | None = None
    last_error: str | None = None
    last_error_time: datetime | None = None

    @property
    def avg_latency_ms(self) -> float:
        """Average latency per request."""
        if self.total_requests == 0:
            return 0.0
        return self.total_latency_ms / self.total_requests

    @property
    def success_rate(self) -> float:
        """Success rate (0.0 to 1.0)."""
        if self.total_requests == 0:
            return 1.0
        return self.successful_requests / self.total_requests

    def record_success(self, latency_ms: float) -> None:
        """Record a successful request."""
        self.total_requests += 1
        self.successful_requests += 1
        self.total_latency_ms += latency_ms
        self.last_request_time = datetime.now(UTC)

    def record_failure(self, error: str, latency_ms: float = 0.0) -> None:
        """Record a failed request."""
        self.total_requests += 1
        self.failed_requests += 1
        self.total_latency_ms += latency_ms
        self.last_error = error
        self.last_error_time = datetime.now(UTC)
        self.last_request_time = datetime.now(UTC)

    def reset(self) -> None:
        """Reset metrics."""
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self.total_latency_ms = 0.0
        self.last_request_time = None
        self.last_error = None
        self.last_error_time = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "provider_type": self.provider_type,
            "total_requests": self.total_requests,
            "successful_requests": self.successful_requests,
            "failed_requests": self.failed_requests,
            "avg_latency_ms": self.avg_latency_ms,
            "success_rate": self.success_rate,
            "last_request_time": self.last_request_time.isoformat()
            if self.last_request_time
            else None,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
        }


class ProviderHealthChecker:
    """
    Health checker for providers.

    Performs health checks and tracks provider status.
    """

    # Thresholds for health status
    LATENCY_DEGRADED_MS = 5000  # 5 seconds
    LATENCY_UNHEALTHY_MS = 30000  # 30 seconds

    def __init__(self, timeout_seconds: float = 10.0):
        """
        Initialize health checker.

        Args:
            timeout_seconds: Timeout for health checks
        """
        self.timeout_seconds = timeout_seconds
        self._last_checks: dict[str, HealthCheckResult] = {}
        self._metrics: dict[str, ProviderMetrics] = {}

    async def check_stt(
        self,
        provider: STTProvider,
        test_audio: bytes | None = None,
    ) -> HealthCheckResult:
        """
        Check health of an STT provider.

        Args:
            provider: STT provider to check
            test_audio: Optional test audio bytes

        Returns:
            HealthCheckResult
        """
        provider_name = getattr(provider, "name", "unknown")
        start_time = time.perf_counter()

        try:
            # Simple validation check - verify provider has required methods
            if not hasattr(provider, "transcribe"):
                return HealthCheckResult(
                    provider_name=provider_name,
                    provider_type="stt",
                    status=HealthStatus.UNHEALTHY,
                    error="Provider missing 'transcribe' method",
                )

            # If test audio provided, do actual transcription
            if test_audio:
                try:
                    await asyncio.wait_for(
                        provider.transcribe(test_audio, "audio/wav"),
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="stt",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=f"Timeout after {self.timeout_seconds}s",
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="stt",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=str(e),
                    )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine status based on latency
            if latency_ms > self.LATENCY_UNHEALTHY_MS:
                status = HealthStatus.UNHEALTHY
            elif latency_ms > self.LATENCY_DEGRADED_MS:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="stt",
                status=status,
                latency_ms=latency_ms,
                message="Provider is operational",
            )

            self._last_checks[f"stt:{provider_name}"] = result
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="stt",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e),
            )
            self._last_checks[f"stt:{provider_name}"] = result
            return result

    async def check_llm(
        self,
        provider: LLMProvider,
        test_prompt: str | None = None,
    ) -> HealthCheckResult:
        """
        Check health of an LLM provider.

        Args:
            provider: LLM provider to check
            test_prompt: Optional test prompt

        Returns:
            HealthCheckResult
        """
        provider_name = getattr(provider, "name", "unknown")
        start_time = time.perf_counter()

        try:
            # Simple validation check
            if not hasattr(provider, "complete"):
                return HealthCheckResult(
                    provider_name=provider_name,
                    provider_type="llm",
                    status=HealthStatus.UNHEALTHY,
                    error="Provider missing 'complete' method",
                )

            # If test prompt provided, do actual completion
            if test_prompt:
                try:
                    from .llm.base import LLMConfig, Message

                    messages = [Message.user(test_prompt)]
                    config = LLMConfig(max_tokens=10)

                    await asyncio.wait_for(
                        provider.complete(messages, config),
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="llm",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=f"Timeout after {self.timeout_seconds}s",
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="llm",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=str(e),
                    )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine status based on latency
            if latency_ms > self.LATENCY_UNHEALTHY_MS:
                status = HealthStatus.UNHEALTHY
            elif latency_ms > self.LATENCY_DEGRADED_MS:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="llm",
                status=status,
                latency_ms=latency_ms,
                message="Provider is operational",
            )

            self._last_checks[f"llm:{provider_name}"] = result
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="llm",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e),
            )
            self._last_checks[f"llm:{provider_name}"] = result
            return result

    async def check_chat(
        self,
        provider: ChatProvider,
    ) -> HealthCheckResult:
        """
        Check health of a Chat provider.

        Args:
            provider: Chat provider to check

        Returns:
            HealthCheckResult
        """
        provider_name = getattr(provider, "name", "unknown")
        start_time = time.perf_counter()

        try:
            # Simple validation check
            if not hasattr(provider, "send_message"):
                return HealthCheckResult(
                    provider_name=provider_name,
                    provider_type="chat",
                    status=HealthStatus.UNHEALTHY,
                    error="Provider missing 'send_message' method",
                )

            # Try to list streams if available (non-destructive check)
            if hasattr(provider, "list_streams"):
                try:
                    await asyncio.wait_for(
                        provider.list_streams(),
                        timeout=self.timeout_seconds,
                    )
                except TimeoutError:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="chat",
                        status=HealthStatus.UNHEALTHY,
                        latency_ms=latency_ms,
                        error=f"Timeout after {self.timeout_seconds}s",
                    )
                except Exception as e:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    return HealthCheckResult(
                        provider_name=provider_name,
                        provider_type="chat",
                        status=HealthStatus.DEGRADED,
                        latency_ms=latency_ms,
                        message=f"list_streams failed: {e}",
                    )

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Determine status based on latency
            if latency_ms > self.LATENCY_UNHEALTHY_MS:
                status = HealthStatus.UNHEALTHY
            elif latency_ms > self.LATENCY_DEGRADED_MS:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="chat",
                status=status,
                latency_ms=latency_ms,
                message="Provider is operational",
            )

            self._last_checks[f"chat:{provider_name}"] = result
            return result

        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000
            result = HealthCheckResult(
                provider_name=provider_name,
                provider_type="chat",
                status=HealthStatus.UNHEALTHY,
                latency_ms=latency_ms,
                error=str(e),
            )
            self._last_checks[f"chat:{provider_name}"] = result
            return result

    def get_last_check(
        self,
        provider_type: str,
        provider_name: str,
    ) -> HealthCheckResult | None:
        """Get the last health check result for a provider."""
        return self._last_checks.get(f"{provider_type}:{provider_name}")

    def get_all_checks(self) -> list[HealthCheckResult]:
        """Get all last health check results."""
        return list(self._last_checks.values())

    def get_metrics(self, provider_type: str, provider_name: str) -> ProviderMetrics:
        """Get or create metrics for a provider."""
        key = f"{provider_type}:{provider_name}"
        if key not in self._metrics:
            self._metrics[key] = ProviderMetrics(
                provider_name=provider_name,
                provider_type=provider_type,
            )
        return self._metrics[key]

    def get_all_metrics(self) -> list[ProviderMetrics]:
        """Get all provider metrics."""
        return list(self._metrics.values())

    def reset_metrics(self, provider_type: str | None = None) -> None:
        """Reset metrics for providers."""
        if provider_type:
            for _key, metrics in self._metrics.items():
                if metrics.provider_type == provider_type:
                    metrics.reset()
        else:
            for metrics in self._metrics.values():
                metrics.reset()


# Global health checker instance
_global_health_checker: ProviderHealthChecker | None = None


def get_health_checker() -> ProviderHealthChecker:
    """Get the global health checker."""
    global _global_health_checker
    if _global_health_checker is None:
        _global_health_checker = ProviderHealthChecker()
    return _global_health_checker


def set_health_checker(checker: ProviderHealthChecker) -> None:
    """Set the global health checker (for testing)."""
    global _global_health_checker
    _global_health_checker = checker


__all__ = [
    "HealthCheckResult",
    "HealthStatus",
    "ProviderHealthChecker",
    "ProviderMetrics",
    "get_health_checker",
    "set_health_checker",
]
