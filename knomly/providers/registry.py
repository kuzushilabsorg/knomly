"""
Provider Registry for Knomly.

Centralized registry for accessing and swapping providers.
Configuration-driven provider selection with health monitoring.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from .health import (
    HealthCheckResult,
    HealthStatus,
    ProviderHealthChecker,
    ProviderMetrics,
)

if TYPE_CHECKING:
    from .chat.base import ChatProvider
    from .llm.base import LLMProvider
    from .stt.base import STTProvider

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class ProviderConfig:
    """Configuration for a provider."""

    name: str
    enabled: bool = True
    priority: int = 0  # Higher = preferred
    metadata: dict[str, Any] = field(default_factory=dict)


class ProviderRegistry:
    """
    Registry for managing and accessing providers.

    Providers are registered by name and can be swapped via configuration.
    Supports lazy initialization, health checks, and metrics tracking.

    Features:
    - Multiple provider types (STT, LLM, Chat)
    - Default provider per type
    - Health checking with automatic fallback
    - Usage metrics tracking
    - Provider validation on registration

    Usage:
        registry = ProviderRegistry()
        registry.register_stt("gemini", GeminiSTTProvider(...))
        registry.register_stt("deepgram", DeepgramSTTProvider(...))
        registry.set_default_stt("gemini")

        # In processor
        stt = registry.get_stt()  # Returns default STT provider
        stt_explicit = registry.get_stt("deepgram")  # Returns specific provider

        # Health check
        results = await registry.check_health()
    """

    def __init__(self, enable_metrics: bool = True):
        """
        Initialize provider registry.

        Args:
            enable_metrics: Enable metrics tracking
        """
        # Provider storage
        self._stt_providers: dict[str, STTProvider] = {}
        self._llm_providers: dict[str, LLMProvider] = {}
        self._chat_providers: dict[str, ChatProvider] = {}

        # Provider configs
        self._stt_configs: dict[str, ProviderConfig] = {}
        self._llm_configs: dict[str, ProviderConfig] = {}
        self._chat_configs: dict[str, ProviderConfig] = {}

        # Default provider names
        self._default_stt: str | None = None
        self._default_llm: str | None = None
        self._default_chat: str | None = None

        # Health checker
        self._health_checker = ProviderHealthChecker()
        self._enable_metrics = enable_metrics

    # ==================== Validation ====================

    def _validate_stt_provider(self, provider: STTProvider) -> None:
        """Validate STT provider has required interface."""
        if not hasattr(provider, "name"):
            raise ValueError("STT provider must have 'name' property")
        if not hasattr(provider, "transcribe") or not callable(provider.transcribe):
            raise ValueError("STT provider must have 'transcribe' method")

    def _validate_llm_provider(self, provider: LLMProvider) -> None:
        """Validate LLM provider has required interface."""
        if not hasattr(provider, "name"):
            raise ValueError("LLM provider must have 'name' property")
        if not hasattr(provider, "complete") or not callable(provider.complete):
            raise ValueError("LLM provider must have 'complete' method")

    def _validate_chat_provider(self, provider: ChatProvider) -> None:
        """Validate Chat provider has required interface."""
        if not hasattr(provider, "name"):
            raise ValueError("Chat provider must have 'name' property")
        if not hasattr(provider, "send_message") or not callable(provider.send_message):
            raise ValueError("Chat provider must have 'send_message' method")

    # ==================== STT Providers ====================

    def register_stt(
        self,
        name: str,
        provider: STTProvider,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        """
        Register an STT provider.

        Args:
            name: Unique name for the provider
            provider: STT provider instance
            enabled: Whether provider is enabled
            priority: Priority for fallback selection (higher = preferred)
        """
        self._validate_stt_provider(provider)
        self._stt_providers[name] = provider
        self._stt_configs[name] = ProviderConfig(name=name, enabled=enabled, priority=priority)
        if self._default_stt is None:
            self._default_stt = name
        logger.debug(f"Registered STT provider: {name} (priority={priority})")

    def unregister_stt(self, name: str) -> None:
        """Unregister an STT provider."""
        if name in self._stt_providers:
            del self._stt_providers[name]
            del self._stt_configs[name]
            if self._default_stt == name:
                self._default_stt = next(iter(self._stt_providers.keys()), None)
            logger.debug(f"Unregistered STT provider: {name}")

    def set_default_stt(self, name: str) -> None:
        """Set the default STT provider."""
        if name not in self._stt_providers:
            raise ValueError(f"STT provider '{name}' not registered")
        self._default_stt = name

    def get_stt(self, name: str | None = None) -> STTProvider:
        """
        Get an STT provider by name or return default.

        Args:
            name: Provider name (uses default if None)

        Returns:
            STT provider instance

        Raises:
            ValueError: If provider not found or disabled
        """
        provider_name = name or self._default_stt
        if provider_name is None:
            raise ValueError("No STT provider registered")
        if provider_name not in self._stt_providers:
            raise ValueError(f"STT provider '{provider_name}' not registered")

        config = self._stt_configs[provider_name]
        if not config.enabled:
            raise ValueError(f"STT provider '{provider_name}' is disabled")

        return self._stt_providers[provider_name]

    def get_stt_with_fallback(self, preferred: str | None = None) -> STTProvider:
        """
        Get STT provider with fallback to healthy alternatives.

        Args:
            preferred: Preferred provider name

        Returns:
            Healthy STT provider

        Raises:
            ValueError: If no healthy provider found
        """
        # Try preferred first
        if preferred and preferred in self._stt_providers:
            config = self._stt_configs[preferred]
            if config.enabled:
                last_check = self._health_checker.get_last_check("stt", preferred)
                if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                    return self._stt_providers[preferred]

        # Try default
        if self._default_stt and self._default_stt in self._stt_providers:
            config = self._stt_configs[self._default_stt]
            if config.enabled:
                last_check = self._health_checker.get_last_check("stt", self._default_stt)
                if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                    return self._stt_providers[self._default_stt]

        # Sort by priority and find healthy one
        sorted_providers = sorted(
            [
                (name, self._stt_configs[name])
                for name in self._stt_providers
                if self._stt_configs[name].enabled
            ],
            key=lambda x: x[1].priority,
            reverse=True,
        )

        for name, _config in sorted_providers:
            last_check = self._health_checker.get_last_check("stt", name)
            if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                return self._stt_providers[name]

        raise ValueError("No healthy STT provider available")

    @property
    def stt(self) -> STTProvider:
        """Shorthand for getting default STT provider."""
        return self.get_stt()

    def list_stt_providers(self) -> list[str]:
        """List registered STT provider names."""
        return list(self._stt_providers.keys())

    # ==================== LLM Providers ====================

    def register_llm(
        self,
        name: str,
        provider: LLMProvider,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        """
        Register an LLM provider.

        Args:
            name: Unique name for the provider
            provider: LLM provider instance
            enabled: Whether provider is enabled
            priority: Priority for fallback selection (higher = preferred)
        """
        self._validate_llm_provider(provider)
        self._llm_providers[name] = provider
        self._llm_configs[name] = ProviderConfig(name=name, enabled=enabled, priority=priority)
        if self._default_llm is None:
            self._default_llm = name
        logger.debug(f"Registered LLM provider: {name} (priority={priority})")

    def unregister_llm(self, name: str) -> None:
        """Unregister an LLM provider."""
        if name in self._llm_providers:
            del self._llm_providers[name]
            del self._llm_configs[name]
            if self._default_llm == name:
                self._default_llm = next(iter(self._llm_providers.keys()), None)
            logger.debug(f"Unregistered LLM provider: {name}")

    def set_default_llm(self, name: str) -> None:
        """Set the default LLM provider."""
        if name not in self._llm_providers:
            raise ValueError(f"LLM provider '{name}' not registered")
        self._default_llm = name

    def get_llm(self, name: str | None = None) -> LLMProvider:
        """
        Get an LLM provider by name or return default.

        Args:
            name: Provider name (uses default if None)

        Returns:
            LLM provider instance

        Raises:
            ValueError: If provider not found or disabled
        """
        provider_name = name or self._default_llm
        if provider_name is None:
            raise ValueError("No LLM provider registered")
        if provider_name not in self._llm_providers:
            raise ValueError(f"LLM provider '{provider_name}' not registered")

        config = self._llm_configs[provider_name]
        if not config.enabled:
            raise ValueError(f"LLM provider '{provider_name}' is disabled")

        return self._llm_providers[provider_name]

    def get_llm_with_fallback(self, preferred: str | None = None) -> LLMProvider:
        """
        Get LLM provider with fallback to healthy alternatives.

        Args:
            preferred: Preferred provider name

        Returns:
            Healthy LLM provider

        Raises:
            ValueError: If no healthy provider found
        """
        # Try preferred first
        if preferred and preferred in self._llm_providers:
            config = self._llm_configs[preferred]
            if config.enabled:
                last_check = self._health_checker.get_last_check("llm", preferred)
                if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                    return self._llm_providers[preferred]

        # Try default
        if self._default_llm and self._default_llm in self._llm_providers:
            config = self._llm_configs[self._default_llm]
            if config.enabled:
                last_check = self._health_checker.get_last_check("llm", self._default_llm)
                if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                    return self._llm_providers[self._default_llm]

        # Sort by priority and find healthy one
        sorted_providers = sorted(
            [
                (name, self._llm_configs[name])
                for name in self._llm_providers
                if self._llm_configs[name].enabled
            ],
            key=lambda x: x[1].priority,
            reverse=True,
        )

        for name, _config in sorted_providers:
            last_check = self._health_checker.get_last_check("llm", name)
            if last_check is None or last_check.status != HealthStatus.UNHEALTHY:
                return self._llm_providers[name]

        raise ValueError("No healthy LLM provider available")

    @property
    def llm(self) -> LLMProvider:
        """Shorthand for getting default LLM provider."""
        return self.get_llm()

    def list_llm_providers(self) -> list[str]:
        """List registered LLM provider names."""
        return list(self._llm_providers.keys())

    # ==================== Chat Providers ====================

    def register_chat(
        self,
        name: str,
        provider: ChatProvider,
        enabled: bool = True,
        priority: int = 0,
    ) -> None:
        """
        Register a Chat provider.

        Args:
            name: Unique name for the provider
            provider: Chat provider instance
            enabled: Whether provider is enabled
            priority: Priority for fallback selection (higher = preferred)
        """
        self._validate_chat_provider(provider)
        self._chat_providers[name] = provider
        self._chat_configs[name] = ProviderConfig(name=name, enabled=enabled, priority=priority)
        if self._default_chat is None:
            self._default_chat = name
        logger.debug(f"Registered Chat provider: {name} (priority={priority})")

    def unregister_chat(self, name: str) -> None:
        """Unregister a Chat provider."""
        if name in self._chat_providers:
            del self._chat_providers[name]
            del self._chat_configs[name]
            if self._default_chat == name:
                self._default_chat = next(iter(self._chat_providers.keys()), None)
            logger.debug(f"Unregistered Chat provider: {name}")

    def set_default_chat(self, name: str) -> None:
        """Set the default Chat provider."""
        if name not in self._chat_providers:
            raise ValueError(f"Chat provider '{name}' not registered")
        self._default_chat = name

    def get_chat(self, name: str | None = None) -> ChatProvider:
        """
        Get a Chat provider by name or return default.

        Args:
            name: Provider name (uses default if None)

        Returns:
            Chat provider instance

        Raises:
            ValueError: If provider not found or disabled
        """
        provider_name = name or self._default_chat
        if provider_name is None:
            raise ValueError("No Chat provider registered")
        if provider_name not in self._chat_providers:
            raise ValueError(f"Chat provider '{provider_name}' not registered")

        config = self._chat_configs[provider_name]
        if not config.enabled:
            raise ValueError(f"Chat provider '{provider_name}' is disabled")

        return self._chat_providers[provider_name]

    @property
    def chat(self) -> ChatProvider:
        """Shorthand for getting default Chat provider."""
        return self.get_chat()

    def list_chat_providers(self) -> list[str]:
        """List registered Chat provider names."""
        return list(self._chat_providers.keys())

    # ==================== Health & Metrics ====================

    async def check_health(
        self,
        provider_types: list[str] | None = None,
    ) -> list[HealthCheckResult]:
        """
        Check health of all registered providers.

        Args:
            provider_types: Optional list of types to check (stt, llm, chat)

        Returns:
            List of health check results
        """
        results = []
        types_to_check = provider_types or ["stt", "llm", "chat"]

        # Check STT providers
        if "stt" in types_to_check:
            for _name, provider in self._stt_providers.items():
                result = await self._health_checker.check_stt(provider)
                results.append(result)

        # Check LLM providers
        if "llm" in types_to_check:
            for _name, provider in self._llm_providers.items():
                result = await self._health_checker.check_llm(provider)
                results.append(result)

        # Check Chat providers
        if "chat" in types_to_check:
            for _name, provider in self._chat_providers.items():
                result = await self._health_checker.check_chat(provider)
                results.append(result)

        return results

    async def check_health_concurrent(
        self,
        timeout: float = 30.0,
    ) -> list[HealthCheckResult]:
        """
        Check health of all providers concurrently.

        Args:
            timeout: Maximum time for all checks

        Returns:
            List of health check results
        """
        tasks = []

        for _name, provider in self._stt_providers.items():
            tasks.append(self._health_checker.check_stt(provider))

        for _name, provider in self._llm_providers.items():
            tasks.append(self._health_checker.check_llm(provider))

        for _name, provider in self._chat_providers.items():
            tasks.append(self._health_checker.check_chat(provider))

        if not tasks:
            return []

        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=timeout)
        return list(results)

    def get_health_summary(self) -> dict[str, Any]:
        """Get summary of provider health status."""
        all_checks = self._health_checker.get_all_checks()

        healthy = sum(1 for c in all_checks if c.status == HealthStatus.HEALTHY)
        degraded = sum(1 for c in all_checks if c.status == HealthStatus.DEGRADED)
        unhealthy = sum(1 for c in all_checks if c.status == HealthStatus.UNHEALTHY)

        return {
            "total_providers": len(all_checks),
            "healthy": healthy,
            "degraded": degraded,
            "unhealthy": unhealthy,
            "overall_status": (
                HealthStatus.HEALTHY.value
                if unhealthy == 0 and degraded == 0
                else HealthStatus.DEGRADED.value
                if unhealthy == 0
                else HealthStatus.UNHEALTHY.value
            ),
            "providers": [c.to_dict() for c in all_checks],
        }

    def get_metrics(self, provider_type: str, provider_name: str) -> ProviderMetrics:
        """Get metrics for a specific provider."""
        return self._health_checker.get_metrics(provider_type, provider_name)

    def get_all_metrics(self) -> list[dict[str, Any]]:
        """Get metrics for all providers."""
        return [m.to_dict() for m in self._health_checker.get_all_metrics()]

    def record_request(
        self,
        provider_type: str,
        provider_name: str,
        success: bool,
        latency_ms: float,
        error: str | None = None,
    ) -> None:
        """Record a request to a provider (for metrics)."""
        if not self._enable_metrics:
            return

        metrics = self._health_checker.get_metrics(provider_type, provider_name)
        if success:
            metrics.record_success(latency_ms)
        else:
            metrics.record_failure(error or "Unknown error", latency_ms)

    # ==================== Utility Methods ====================

    def list_providers(self) -> dict[str, dict[str, Any]]:
        """List all registered providers and defaults."""
        return {
            "stt": {
                "registered": list(self._stt_providers.keys()),
                "default": self._default_stt,
                "configs": {
                    name: {"enabled": c.enabled, "priority": c.priority}
                    for name, c in self._stt_configs.items()
                },
            },
            "llm": {
                "registered": list(self._llm_providers.keys()),
                "default": self._default_llm,
                "configs": {
                    name: {"enabled": c.enabled, "priority": c.priority}
                    for name, c in self._llm_configs.items()
                },
            },
            "chat": {
                "registered": list(self._chat_providers.keys()),
                "default": self._default_chat,
                "configs": {
                    name: {"enabled": c.enabled, "priority": c.priority}
                    for name, c in self._chat_configs.items()
                },
            },
        }

    def enable_provider(self, provider_type: str, name: str) -> None:
        """Enable a provider."""
        configs = {
            "stt": self._stt_configs,
            "llm": self._llm_configs,
            "chat": self._chat_configs,
        }.get(provider_type)

        if configs is None:
            raise ValueError(f"Unknown provider type: {provider_type}")
        if name not in configs:
            raise ValueError(f"{provider_type} provider '{name}' not registered")

        configs[name].enabled = True
        logger.debug(f"Enabled {provider_type} provider: {name}")

    def disable_provider(self, provider_type: str, name: str) -> None:
        """Disable a provider."""
        configs = {
            "stt": self._stt_configs,
            "llm": self._llm_configs,
            "chat": self._chat_configs,
        }.get(provider_type)

        if configs is None:
            raise ValueError(f"Unknown provider type: {provider_type}")
        if name not in configs:
            raise ValueError(f"{provider_type} provider '{name}' not registered")

        configs[name].enabled = False
        logger.debug(f"Disabled {provider_type} provider: {name}")

    def clear(self) -> None:
        """Clear all registered providers."""
        self._stt_providers.clear()
        self._llm_providers.clear()
        self._chat_providers.clear()
        self._stt_configs.clear()
        self._llm_configs.clear()
        self._chat_configs.clear()
        self._default_stt = None
        self._default_llm = None
        self._default_chat = None

    def __repr__(self) -> str:
        return (
            f"ProviderRegistry("
            f"stt=[{', '.join(self._stt_providers.keys())}], "
            f"llm=[{', '.join(self._llm_providers.keys())}], "
            f"chat=[{', '.join(self._chat_providers.keys())}])"
        )


# Global registry instance (can be replaced in tests)
_global_registry: ProviderRegistry | None = None


def get_registry() -> ProviderRegistry:
    """Get the global provider registry."""
    global _global_registry
    if _global_registry is None:
        _global_registry = ProviderRegistry()
    return _global_registry


def set_registry(registry: ProviderRegistry) -> None:
    """Set the global provider registry (for testing)."""
    global _global_registry
    _global_registry = registry


def reset_registry() -> None:
    """Reset the global provider registry."""
    global _global_registry
    _global_registry = None
