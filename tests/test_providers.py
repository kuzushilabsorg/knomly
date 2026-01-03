"""
Tests for Knomly providers.

Tests provider registration, protocols, and implementations.
"""
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from knomly.providers import ProviderRegistry
from knomly.providers.stt import STTProvider, TranscriptionResult
from knomly.providers.llm import LLMConfig, LLMProvider, LLMResponse, Message
from knomly.providers.chat import ChatProvider, MessageResult


class MockSTTProvider:
    """Mock STT provider for testing."""

    @property
    def name(self) -> str:
        return "mock_stt"

    async def transcribe(
        self,
        audio_bytes: bytes,
        mime_type: str = "audio/ogg",
        language_hint: str | None = None,
    ) -> TranscriptionResult:
        return TranscriptionResult(
            original_text="Mocked transcription",
            english_text="Mocked transcription",
            detected_language="en",
            language_name="English",
            confidence=0.95,
            provider=self.name,
        )


class MockLLMProvider:
    """Mock LLM provider for testing."""

    @property
    def name(self) -> str:
        return "mock_llm"

    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        return LLMResponse(
            content='{"today_items": ["task"], "blockers": []}',
            model="mock-model",
            provider=self.name,
        )


class MockChatProvider:
    """Mock Chat provider for testing."""

    @property
    def name(self) -> str:
        return "mock_chat"

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        return MessageResult(
            success=True,
            message_id=12345,
            stream=stream,
            topic=topic,
        )


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def test_registry_register_stt(self):
        """Registry should register STT providers."""
        registry = ProviderRegistry()
        provider = MockSTTProvider()

        registry.register_stt("mock", provider)

        assert "mock" in registry._stt_providers
        assert registry.get_stt("mock") is provider

    def test_registry_auto_default(self):
        """First registered provider should become default."""
        registry = ProviderRegistry()
        provider = MockSTTProvider()

        registry.register_stt("mock", provider)

        assert registry._default_stt == "mock"
        assert registry.get_stt() is provider

    def test_registry_set_default_stt(self):
        """set_default_stt should change the default."""
        registry = ProviderRegistry()
        provider1 = MockSTTProvider()
        provider2 = MockSTTProvider()

        registry.register_stt("first", provider1)
        registry.register_stt("second", provider2)
        registry.set_default_stt("second")

        assert registry.get_stt() is provider2

    def test_registry_invalid_default_raises(self):
        """set_default_stt should raise for unknown provider."""
        registry = ProviderRegistry()

        with pytest.raises(ValueError):
            registry.set_default_stt("nonexistent")

    def test_registry_get_unknown_raises(self):
        """get_stt should raise for unknown provider."""
        registry = ProviderRegistry()
        registry.register_stt("mock", MockSTTProvider())

        with pytest.raises(ValueError):
            registry.get_stt("nonexistent")

    def test_registry_get_no_default_raises(self):
        """get_stt should raise when no providers registered."""
        registry = ProviderRegistry()

        with pytest.raises(ValueError):
            registry.get_stt()

    def test_registry_register_llm(self):
        """Registry should register LLM providers."""
        registry = ProviderRegistry()
        provider = MockLLMProvider()

        registry.register_llm("mock", provider)

        assert registry.get_llm("mock") is provider

    def test_registry_register_chat(self):
        """Registry should register Chat providers."""
        registry = ProviderRegistry()
        provider = MockChatProvider()

        registry.register_chat("mock", provider)

        assert registry.get_chat("mock") is provider

    def test_registry_list_providers(self):
        """list_providers should return all registered providers."""
        registry = ProviderRegistry()
        registry.register_stt("stt1", MockSTTProvider())
        registry.register_llm("llm1", MockLLMProvider())
        registry.register_chat("chat1", MockChatProvider())

        listing = registry.list_providers()

        assert "stt1" in listing["stt"]["registered"]
        assert "llm1" in listing["llm"]["registered"]
        assert "chat1" in listing["chat"]["registered"]

    def test_registry_shorthand_properties(self):
        """Registry stt/llm/chat properties should return defaults."""
        registry = ProviderRegistry()
        stt = MockSTTProvider()
        llm = MockLLMProvider()
        chat = MockChatProvider()

        registry.register_stt("s", stt)
        registry.register_llm("l", llm)
        registry.register_chat("c", chat)

        assert registry.stt is stt
        assert registry.llm is llm
        assert registry.chat is chat


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""

    def test_transcription_result_creation(self):
        """TranscriptionResult should store all fields."""
        result = TranscriptionResult(
            original_text="Original",
            english_text="English",
            detected_language="hi",
            language_name="Hindi",
            confidence=0.9,
            provider="test",
        )

        assert result.original_text == "Original"
        assert result.english_text == "English"
        assert result.detected_language == "hi"
        assert result.confidence == 0.9

    def test_transcription_result_to_dict(self):
        """to_dict should serialize the result."""
        result = TranscriptionResult(
            original_text="Test",
            english_text="Test",
            provider="test",
        )

        data = result.to_dict()
        assert "original_text" in data
        assert "english_text" in data
        assert "provider" in data


class TestLLMTypes:
    """Tests for LLM types."""

    def test_message_factory_methods(self):
        """Message class should have factory methods."""
        system = Message.system("You are helpful")
        user = Message.user("Hello")
        assistant = Message.assistant("Hi there")

        assert system.role.value == "system"
        assert user.role.value == "user"
        assert assistant.role.value == "assistant"

    def test_message_to_dict(self):
        """Message.to_dict should return API format."""
        msg = Message.user("Test content")
        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Test content"

    def test_llm_config_defaults(self):
        """LLMConfig should have sensible defaults."""
        config = LLMConfig()

        assert config.temperature == 0.7
        assert config.max_tokens == 1024
        assert config.model is None  # Use provider default

    def test_llm_response_token_properties(self):
        """LLMResponse should have token counting properties."""
        response = LLMResponse(
            content="Test",
            usage={"input_tokens": 10, "output_tokens": 20},
        )

        assert response.input_tokens == 10
        assert response.output_tokens == 20
        assert response.total_tokens == 30


class TestMessageResult:
    """Tests for Chat MessageResult."""

    def test_message_result_success(self):
        """MessageResult should represent successful send."""
        result = MessageResult(
            success=True,
            message_id=12345,
            stream="test",
            topic="topic",
        )

        assert result.success is True
        assert result.message_id == 12345
        assert result.error is None

    def test_message_result_failure(self):
        """MessageResult should represent failed send."""
        result = MessageResult(
            success=False,
            error="Connection refused",
        )

        assert result.success is False
        assert result.error == "Connection refused"
