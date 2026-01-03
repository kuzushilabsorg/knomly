"""
Knomly Providers

Swappable service providers for STT, LLM, and Chat.

Provider Types:
- STT: GeminiSTTProvider, DeepgramSTTProvider, WhisperSTTProvider
- LLM: OpenAILLMProvider, AnthropicLLMProvider, GeminiLLMProvider
- Chat: ZulipChatProvider

Features:
- ProviderRegistry for centralized management
- Health checks with automatic fallback
- Usage metrics tracking
- Provider validation on registration
"""

# Chat Providers
from .chat import (
    BaseChatProvider,
    ChatProvider,
    MessageResult,
    ZulipChatProvider,
)
from .health import (
    HealthCheckResult,
    HealthStatus,
    ProviderHealthChecker,
    ProviderMetrics,
    get_health_checker,
    set_health_checker,
)

# LLM Providers
from .llm import (
    AnthropicLLMProvider,
    BaseLLMProvider,
    GeminiFlashProvider,
    GeminiLLMProvider,
    GeminiProProvider,
    LLMConfig,
    LLMProvider,
    LLMResponse,
    Message,
    MessageRole,
    OpenAILLMProvider,
)
from .registry import (
    ProviderConfig,
    ProviderRegistry,
    get_registry,
    reset_registry,
    set_registry,
)

# STT Providers
from .stt import (
    BaseSTTProvider,
    DeepgramStreamingSTTProvider,
    DeepgramSTTProvider,
    GeminiSTTProvider,
    STTProvider,
    TranscriptionResult,
    WhisperSTTProvider,
)

__all__ = [
    "AnthropicLLMProvider",
    "BaseChatProvider",
    "BaseLLMProvider",
    "BaseSTTProvider",
    # Chat - Protocol and Base
    "ChatProvider",
    "DeepgramSTTProvider",
    "DeepgramStreamingSTTProvider",
    "GeminiFlashProvider",
    "GeminiLLMProvider",
    "GeminiProProvider",
    # STT - Implementations
    "GeminiSTTProvider",
    "HealthCheckResult",
    # Health
    "HealthStatus",
    "LLMConfig",
    # LLM - Protocol and Base
    "LLMProvider",
    "LLMResponse",
    "Message",
    "MessageResult",
    "MessageRole",
    # LLM - Implementations
    "OpenAILLMProvider",
    # Registry
    "ProviderConfig",
    "ProviderHealthChecker",
    "ProviderMetrics",
    "ProviderRegistry",
    # STT - Protocol and Base
    "STTProvider",
    "TranscriptionResult",
    "WhisperSTTProvider",
    # Chat - Implementations
    "ZulipChatProvider",
    "get_health_checker",
    "get_registry",
    "reset_registry",
    "set_health_checker",
    "set_registry",
]
