"""
Tests for Provider Health Checks and Enhanced Registry.
"""

import pytest

from knomly.providers import (
    HealthCheckResult,
    HealthStatus,
    ProviderHealthChecker,
    ProviderMetrics,
    ProviderRegistry,
    TranscriptionResult,
)
from knomly.providers.llm.base import LLMConfig, LLMResponse

# =============================================================================
# Mock Providers for Testing
# =============================================================================


class MockSTTProvider:
    """Mock STT provider for testing."""

    def __init__(self, name: str = "mock_stt", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str = None,
    ) -> TranscriptionResult:
        if self._should_fail:
            raise RuntimeError("Mock STT failure")
        return TranscriptionResult(
            original_text="Hello world",
            english_text="Hello world",
            detected_language="en",
            language_name="English",
            confidence=0.95,
            provider=self.name,
        )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, name: str = "mock_llm", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail
        self.default_model = "mock-model"

    @property
    def name(self) -> str:
        return self._name

    async def complete(
        self,
        messages: list,
        config: LLMConfig = None,
    ) -> LLMResponse:
        if self._should_fail:
            raise RuntimeError("Mock LLM failure")
        return LLMResponse(
            content="Hello from mock LLM",
            model="mock-model",
            provider=self.name,
        )


class MockChatProvider:
    """Mock Chat provider for testing."""

    def __init__(self, name: str = "mock_chat", should_fail: bool = False):
        self._name = name
        self._should_fail = should_fail

    @property
    def name(self) -> str:
        return self._name

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> dict:
        if self._should_fail:
            raise RuntimeError("Mock Chat failure")
        return {"success": True, "message_id": 123}

    async def list_streams(self) -> list:
        if self._should_fail:
            raise RuntimeError("Mock list_streams failure")
        return ["general", "standup"]


# =============================================================================
# HealthStatus Tests
# =============================================================================


class TestHealthStatus:
    """Tests for HealthStatus enum."""

    def test_status_values(self):
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.UNHEALTHY.value == "unhealthy"
        assert HealthStatus.UNKNOWN.value == "unknown"


# =============================================================================
# HealthCheckResult Tests
# =============================================================================


class TestHealthCheckResult:
    """Tests for HealthCheckResult."""

    def test_create_result(self):
        result = HealthCheckResult(
            provider_name="test",
            provider_type="stt",
            status=HealthStatus.HEALTHY,
            latency_ms=50.0,
            message="All good",
        )

        assert result.provider_name == "test"
        assert result.provider_type == "stt"
        assert result.status == HealthStatus.HEALTHY
        assert result.latency_ms == 50.0
        assert result.message == "All good"
        assert result.error is None

    def test_to_dict(self):
        result = HealthCheckResult(
            provider_name="test",
            provider_type="llm",
            status=HealthStatus.DEGRADED,
            latency_ms=5500.0,
            error="Slow response",
        )

        d = result.to_dict()

        assert d["provider_name"] == "test"
        assert d["provider_type"] == "llm"
        assert d["status"] == "degraded"
        assert d["latency_ms"] == 5500.0
        assert d["error"] == "Slow response"
        assert "timestamp" in d


# =============================================================================
# ProviderMetrics Tests
# =============================================================================


class TestProviderMetrics:
    """Tests for ProviderMetrics."""

    def test_initial_state(self):
        metrics = ProviderMetrics(provider_name="test", provider_type="stt")

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.avg_latency_ms == 0.0
        assert metrics.success_rate == 1.0  # No failures yet

    def test_record_success(self):
        metrics = ProviderMetrics(provider_name="test", provider_type="stt")

        metrics.record_success(100.0)
        metrics.record_success(200.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 0
        assert metrics.avg_latency_ms == 150.0
        assert metrics.success_rate == 1.0

    def test_record_failure(self):
        metrics = ProviderMetrics(provider_name="test", provider_type="stt")

        metrics.record_success(100.0)
        metrics.record_failure("Connection timeout", 50.0)

        assert metrics.total_requests == 2
        assert metrics.successful_requests == 1
        assert metrics.failed_requests == 1
        assert metrics.success_rate == 0.5
        assert metrics.last_error == "Connection timeout"
        assert metrics.last_error_time is not None

    def test_reset(self):
        metrics = ProviderMetrics(provider_name="test", provider_type="stt")
        metrics.record_success(100.0)
        metrics.record_failure("Error", 50.0)

        metrics.reset()

        assert metrics.total_requests == 0
        assert metrics.successful_requests == 0
        assert metrics.failed_requests == 0
        assert metrics.last_error is None

    def test_to_dict(self):
        metrics = ProviderMetrics(provider_name="test", provider_type="llm")
        metrics.record_success(100.0)

        d = metrics.to_dict()

        assert d["provider_name"] == "test"
        assert d["provider_type"] == "llm"
        assert d["total_requests"] == 1
        assert d["success_rate"] == 1.0


# =============================================================================
# ProviderHealthChecker Tests
# =============================================================================


class TestProviderHealthChecker:
    """Tests for ProviderHealthChecker."""

    @pytest.fixture
    def checker(self) -> ProviderHealthChecker:
        return ProviderHealthChecker(timeout_seconds=5.0)

    @pytest.mark.asyncio
    async def test_check_stt_healthy(self, checker: ProviderHealthChecker):
        provider = MockSTTProvider()

        result = await checker.check_stt(provider)

        assert result.status == HealthStatus.HEALTHY
        assert result.provider_name == "mock_stt"
        assert result.provider_type == "stt"
        assert result.latency_ms is not None
        assert result.latency_ms >= 0

    @pytest.mark.asyncio
    async def test_check_stt_missing_method(self, checker: ProviderHealthChecker):
        class BadProvider:
            name = "bad"
            # Missing transcribe method

        result = await checker.check_stt(BadProvider())

        assert result.status == HealthStatus.UNHEALTHY
        assert "transcribe" in result.error

    @pytest.mark.asyncio
    async def test_check_llm_healthy(self, checker: ProviderHealthChecker):
        provider = MockLLMProvider()

        result = await checker.check_llm(provider)

        assert result.status == HealthStatus.HEALTHY
        assert result.provider_name == "mock_llm"
        assert result.provider_type == "llm"

    @pytest.mark.asyncio
    async def test_check_llm_missing_method(self, checker: ProviderHealthChecker):
        class BadProvider:
            name = "bad"
            # Missing complete method

        result = await checker.check_llm(BadProvider())

        assert result.status == HealthStatus.UNHEALTHY
        assert "complete" in result.error

    @pytest.mark.asyncio
    async def test_check_chat_healthy(self, checker: ProviderHealthChecker):
        provider = MockChatProvider()

        result = await checker.check_chat(provider)

        assert result.status == HealthStatus.HEALTHY
        assert result.provider_name == "mock_chat"
        assert result.provider_type == "chat"

    @pytest.mark.asyncio
    async def test_check_chat_missing_method(self, checker: ProviderHealthChecker):
        class BadProvider:
            name = "bad"
            # Missing send_message method

        result = await checker.check_chat(BadProvider())

        assert result.status == HealthStatus.UNHEALTHY
        assert "send_message" in result.error

    def test_get_metrics(self, checker: ProviderHealthChecker):
        metrics = checker.get_metrics("stt", "test_provider")

        assert metrics.provider_name == "test_provider"
        assert metrics.provider_type == "stt"
        assert metrics.total_requests == 0

    def test_get_all_metrics(self, checker: ProviderHealthChecker):
        checker.get_metrics("stt", "stt1")
        checker.get_metrics("llm", "llm1")

        all_metrics = checker.get_all_metrics()

        assert len(all_metrics) == 2


# =============================================================================
# Enhanced ProviderRegistry Tests
# =============================================================================


class TestEnhancedProviderRegistry:
    """Tests for enhanced ProviderRegistry with health checks."""

    @pytest.fixture
    def registry(self) -> ProviderRegistry:
        return ProviderRegistry(enable_metrics=True)

    def test_register_with_priority(self, registry: ProviderRegistry):
        provider1 = MockSTTProvider("stt1")
        provider2 = MockSTTProvider("stt2")

        registry.register_stt("stt1", provider1, priority=10)
        registry.register_stt("stt2", provider2, priority=20)

        providers = registry.list_providers()

        assert providers["stt"]["configs"]["stt1"]["priority"] == 10
        assert providers["stt"]["configs"]["stt2"]["priority"] == 20

    def test_register_with_enabled_flag(self, registry: ProviderRegistry):
        provider = MockSTTProvider()

        registry.register_stt("test", provider, enabled=False)

        with pytest.raises(ValueError, match="disabled"):
            registry.get_stt("test")

    def test_enable_disable_provider(self, registry: ProviderRegistry):
        provider = MockSTTProvider()
        registry.register_stt("test", provider)

        # Disable
        registry.disable_provider("stt", "test")
        with pytest.raises(ValueError, match="disabled"):
            registry.get_stt("test")

        # Re-enable
        registry.enable_provider("stt", "test")
        assert registry.get_stt("test") == provider

    def test_unregister_provider(self, registry: ProviderRegistry):
        provider = MockSTTProvider()
        registry.register_stt("test", provider)

        registry.unregister_stt("test")

        assert "test" not in registry.list_stt_providers()

    def test_validation_on_register(self, registry: ProviderRegistry):
        class BadProvider:
            pass  # Missing name and methods

        with pytest.raises(ValueError, match="name"):
            registry.register_stt("bad", BadProvider())

    @pytest.mark.asyncio
    async def test_check_health(self, registry: ProviderRegistry):
        registry.register_stt("stt", MockSTTProvider())
        registry.register_llm("llm", MockLLMProvider())
        registry.register_chat("chat", MockChatProvider())

        results = await registry.check_health()

        assert len(results) == 3
        assert all(r.status == HealthStatus.HEALTHY for r in results)

    @pytest.mark.asyncio
    async def test_check_health_concurrent(self, registry: ProviderRegistry):
        registry.register_stt("stt", MockSTTProvider())
        registry.register_llm("llm", MockLLMProvider())
        registry.register_chat("chat", MockChatProvider())

        results = await registry.check_health_concurrent()

        assert len(results) == 3

    def test_get_health_summary(self, registry: ProviderRegistry):
        summary = registry.get_health_summary()

        assert "total_providers" in summary
        assert "healthy" in summary
        assert "degraded" in summary
        assert "unhealthy" in summary
        assert "overall_status" in summary

    def test_record_request_metrics(self, registry: ProviderRegistry):
        registry.register_stt("test", MockSTTProvider())

        registry.record_request("stt", "test", True, 100.0)
        registry.record_request("stt", "test", True, 200.0)
        registry.record_request("stt", "test", False, 50.0, "Timeout")

        metrics = registry.get_metrics("stt", "test")

        assert metrics.total_requests == 3
        assert metrics.successful_requests == 2
        assert metrics.failed_requests == 1

    def test_get_stt_with_fallback(self, registry: ProviderRegistry):
        provider1 = MockSTTProvider("primary")
        provider2 = MockSTTProvider("fallback")

        registry.register_stt("primary", provider1, priority=10)
        registry.register_stt("fallback", provider2, priority=5)

        # Should return primary (higher priority)
        result = registry.get_stt_with_fallback()
        assert result.name == "primary"

    def test_clear_registry(self, registry: ProviderRegistry):
        registry.register_stt("stt", MockSTTProvider())
        registry.register_llm("llm", MockLLMProvider())
        registry.register_chat("chat", MockChatProvider())

        registry.clear()

        assert len(registry.list_stt_providers()) == 0
        assert len(registry.list_llm_providers()) == 0
        assert len(registry.list_chat_providers()) == 0

    def test_list_providers_includes_configs(self, registry: ProviderRegistry):
        registry.register_stt("test", MockSTTProvider(), priority=5)

        providers = registry.list_providers()

        assert "configs" in providers["stt"]
        assert providers["stt"]["configs"]["test"]["priority"] == 5
        assert providers["stt"]["configs"]["test"]["enabled"] is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestProviderHealthIntegration:
    """Integration tests for provider health system."""

    @pytest.mark.asyncio
    async def test_full_health_check_workflow(self):
        registry = ProviderRegistry()

        # Register multiple providers
        registry.register_stt("gemini", MockSTTProvider("gemini"), priority=10)
        registry.register_stt("deepgram", MockSTTProvider("deepgram"), priority=5)
        registry.register_llm("openai", MockLLMProvider("openai"), priority=10)
        registry.register_llm("anthropic", MockLLMProvider("anthropic"), priority=5)
        registry.register_chat("zulip", MockChatProvider("zulip"))

        # Run health checks
        results = await registry.check_health()
        assert len(results) == 5
        assert all(r.status == HealthStatus.HEALTHY for r in results)

        # Get summary
        summary = registry.get_health_summary()
        assert summary["healthy"] == 5
        assert summary["overall_status"] == "healthy"

        # Record some requests
        registry.record_request("stt", "gemini", True, 100.0)
        registry.record_request("stt", "gemini", True, 150.0)
        registry.record_request("llm", "openai", True, 500.0)

        # Check metrics
        all_metrics = registry.get_all_metrics()
        assert len(all_metrics) == 2  # Only providers with recorded requests

    @pytest.mark.asyncio
    async def test_fallback_with_unhealthy_provider(self):
        registry = ProviderRegistry()

        # Register providers
        registry.register_stt("primary", MockSTTProvider("primary"), priority=10)
        registry.register_stt("fallback", MockSTTProvider("fallback"), priority=5)

        # Get with fallback (both healthy, should return primary)
        provider = registry.get_stt_with_fallback()
        assert provider.name == "primary"

        # Disable primary
        registry.disable_provider("stt", "primary")

        # Should now return fallback
        provider = registry.get_stt_with_fallback()
        assert provider.name == "fallback"
