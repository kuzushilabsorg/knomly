"""
LLM Provider Protocol for Knomly.

Defines the interface for Large Language Model providers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """
    A message in the LLM conversation.

    Attributes:
        role: Role of the message sender
        content: Text content of the message
        name: Optional name for the sender
    """

    role: MessageRole
    content: str
    name: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        return result

    @classmethod
    def system(cls, content: str) -> "Message":
        """Create a system message."""
        return cls(role=MessageRole.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> "Message":
        """Create a user message."""
        return cls(role=MessageRole.USER, content=content)

    @classmethod
    def assistant(cls, content: str) -> "Message":
        """Create an assistant message."""
        return cls(role=MessageRole.ASSISTANT, content=content)


@dataclass
class LLMResponse:
    """
    Response from LLM completion.

    Attributes:
        content: The generated text content
        model: Model used for generation
        usage: Token usage statistics
        finish_reason: Why generation stopped
        provider: Name of the provider
    """

    content: str
    model: str = ""
    usage: Dict[str, int] = field(default_factory=dict)
    finish_reason: str = "stop"
    provider: str = ""

    @property
    def input_tokens(self) -> int:
        return self.usage.get("input_tokens", 0)

    @property
    def output_tokens(self) -> int:
        return self.usage.get("output_tokens", 0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "usage": self.usage,
            "finish_reason": self.finish_reason,
            "provider": self.provider,
        }


@dataclass
class LLMConfig:
    """
    Configuration for LLM requests.

    Attributes:
        model: Model identifier (e.g., "gpt-4o-mini")
        temperature: Sampling temperature (0.0 to 2.0)
        max_tokens: Maximum tokens to generate
        top_p: Nucleus sampling parameter
        response_format: Format specification (e.g., "json")
    """

    model: Optional[str] = None  # Use provider default if None
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 1.0
    response_format: Optional[str] = None  # "json" for JSON mode


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for LLM providers.

    Implementations must provide:
    - complete(): Generate text from messages
    - name: Provider identifier

    Supported providers:
    - OpenAI (gpt-4o-mini, gpt-4o)
    - Anthropic (claude-3-5-haiku, claude-3-5-sonnet)
    - Google (gemini-2.0-flash)
    """

    @property
    def name(self) -> str:
        """Provider name for logging and configuration."""
        ...

    async def complete(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion from messages.

        Args:
            messages: List of conversation messages
            config: Optional configuration overrides

        Returns:
            LLMResponse with generated content
        """
        ...


class BaseLLMProvider(ABC):
    """
    Base class for LLM provider implementations.

    Provides common functionality and enforces interface.
    """

    def __init__(self, default_model: str = ""):
        self.default_model = default_model

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion from messages."""
        pass

    async def complete_json(
        self,
        messages: List[Message],
        config: Optional[LLMConfig] = None,
    ) -> Dict[str, Any]:
        """
        Generate a JSON completion from messages.

        Convenience method that sets response_format and parses JSON.
        """
        import json

        if config is None:
            config = LLMConfig()
        config.response_format = "json"

        response = await self.complete(messages, config)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]
            return json.loads(content.strip())

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', model='{self.default_model}')"
