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

from .registry import (
    ProviderConfig,
    ProviderRegistry,
    get_registry,
    set_registry,
    reset_registry,
)

from .health import (
    HealthCheckResult,
    HealthStatus,
    ProviderHealthChecker,
    ProviderMetrics,
    get_health_checker,
    set_health_checker,
)

# STT Providers
from .stt import (
    BaseSTTProvider,
    DeepgramSTTProvider,
    DeepgramStreamingSTTProvider,
    GeminiSTTProvider,
    STTProvider,
    TranscriptionResult,
    WhisperSTTProvider,
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

# Chat Providers
from .chat import (
    BaseChatProvider,
    ChatProvider,
    MessageResult,
    ZulipChatProvider,
)

__all__ = [
    # Registry
    "ProviderConfig",
    "ProviderRegistry",
    "get_registry",
    "set_registry",
    "reset_registry",
    # Health
    "HealthStatus",
    "HealthCheckResult",
    "ProviderMetrics",
    "ProviderHealthChecker",
    "get_health_checker",
    "set_health_checker",
    # STT - Protocol and Base
    "STTProvider",
    "BaseSTTProvider",
    "TranscriptionResult",
    # STT - Implementations
    "GeminiSTTProvider",
    "DeepgramSTTProvider",
    "DeepgramStreamingSTTProvider",
    "WhisperSTTProvider",
    # LLM - Protocol and Base
    "LLMProvider",
    "BaseLLMProvider",
    "LLMConfig",
    "LLMResponse",
    "Message",
    "MessageRole",
    # LLM - Implementations
    "OpenAILLMProvider",
    "AnthropicLLMProvider",
    "GeminiLLMProvider",
    "GeminiFlashProvider",
    "GeminiProProvider",
    # Chat - Protocol and Base
    "ChatProvider",
    "BaseChatProvider",
    "MessageResult",
    # Chat - Implementations
    "ZulipChatProvider",
]
