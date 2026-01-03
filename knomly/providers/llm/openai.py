"""
OpenAI LLM Provider for Knomly.

Uses OpenAI's GPT models for text generation.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import BaseLLMProvider, LLMConfig, LLMResponse, Message

logger = logging.getLogger(__name__)


class OpenAILLMProvider(BaseLLMProvider):
    """
    OpenAI-based LLM provider.

    Uses OpenAI's Chat Completions API with:
    - GPT-4o-mini (default, fast and cost-effective)
    - GPT-4o (more capable)
    - GPT-4 Turbo (for complex tasks)

    Requirements:
    - openai package
    - OPENAI_API_KEY environment variable
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        organization: str | None = None,
    ):
        """
        Initialize OpenAI LLM provider.

        Args:
            api_key: OpenAI API key
            model: Default model to use
            organization: Optional OpenAI organization ID
        """
        super().__init__(default_model=model)
        self._api_key = api_key
        self._organization = organization
        self._client = None  # Lazy initialization

    @property
    def name(self) -> str:
        return "openai"

    def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI

                self._client = AsyncOpenAI(
                    api_key=self._api_key,
                    organization=self._organization,
                )
            except ImportError:
                raise ImportError(
                    "openai package is required for OpenAI LLM. " "Install with: pip install openai"
                )
        return self._client

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Convert Message objects to OpenAI format."""
        return [msg.to_dict() for msg in messages]

    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI.

        Args:
            messages: List of conversation messages
            config: Optional configuration overrides

        Returns:
            LLMResponse with generated content
        """
        if config is None:
            config = LLMConfig()

        try:
            client = self._get_client()

            # Build request parameters
            params: dict[str, Any] = {
                "model": config.model or self.default_model,
                "messages": self._convert_messages(messages),
                "temperature": config.temperature,
                "max_tokens": config.max_tokens,
                "top_p": config.top_p,
            }

            # Add response format if specified
            if config.response_format == "json":
                params["response_format"] = {"type": "json_object"}

            # Make API call
            response = await client.chat.completions.create(**params)

            # Extract content
            choice = response.choices[0]
            content = choice.message.content or ""

            # Build usage dict
            usage = {}
            if response.usage:
                usage = {
                    "input_tokens": response.usage.prompt_tokens,
                    "output_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=choice.finish_reason or "stop",
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"OpenAI completion error: {e}", exc_info=True)
            raise


class AnthropicLLMProvider(BaseLLMProvider):
    """
    Anthropic-based LLM provider.

    Uses Anthropic's Claude models:
    - claude-3-5-haiku (fast, cost-effective)
    - claude-3-5-sonnet (balanced)
    - claude-3-opus (most capable)

    Requirements:
    - anthropic package
    - ANTHROPIC_API_KEY environment variable
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-haiku-latest",
    ):
        """
        Initialize Anthropic LLM provider.

        Args:
            api_key: Anthropic API key
            model: Default model to use
        """
        super().__init__(default_model=model)
        self._api_key = api_key
        self._client = None

    @property
    def name(self) -> str:
        return "anthropic"

    def _get_client(self):
        """Lazy initialization of Anthropic client."""
        if self._client is None:
            try:
                from anthropic import AsyncAnthropic

                self._client = AsyncAnthropic(api_key=self._api_key)
            except ImportError:
                raise ImportError(
                    "anthropic package is required for Anthropic LLM. "
                    "Install with: pip install anthropic"
                )
        return self._client

    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """
        Generate a completion using Anthropic.

        Args:
            messages: List of conversation messages
            config: Optional configuration overrides

        Returns:
            LLMResponse with generated content
        """
        if config is None:
            config = LLMConfig()

        try:
            client = self._get_client()

            # Separate system message from conversation
            system_prompt = ""
            conversation = []
            for msg in messages:
                if msg.role.value == "system":
                    system_prompt = msg.content
                else:
                    conversation.append(
                        {
                            "role": msg.role.value,
                            "content": msg.content,
                        }
                    )

            # Make API call
            response = await client.messages.create(
                model=config.model or self.default_model,
                max_tokens=config.max_tokens,
                system=system_prompt,
                messages=conversation,
            )

            # Extract content
            content = ""
            if response.content:
                content = response.content[0].text

            # Build usage dict
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }

            return LLMResponse(
                content=content,
                model=response.model,
                usage=usage,
                finish_reason=response.stop_reason or "end_turn",
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"Anthropic completion error: {e}", exc_info=True)
            raise
