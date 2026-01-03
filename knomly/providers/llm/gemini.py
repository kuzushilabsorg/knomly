"""
Gemini LLM Provider for Knomly.

Uses Google's Gemini models for text generation.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .base import BaseLLMProvider, LLMConfig, LLMResponse, Message, MessageRole

logger = logging.getLogger(__name__)


class GeminiLLMProvider(BaseLLMProvider):
    """
    Google Gemini LLM provider.

    Uses Google's Gemini models for:
    - Text generation (gemini-2.0-flash-exp, gemini-1.5-pro)
    - Multi-modal understanding (with images)
    - JSON mode for structured outputs

    Requirements:
    - google-generativeai package
    - GEMINI_API_KEY environment variable

    Features:
    - High-speed inference (Flash models)
    - Long context windows (up to 1M tokens)
    - Multi-modal capabilities
    - JSON mode support
    - Safety settings customization
    """

    # Default safety settings (relaxed for production use)
    DEFAULT_SAFETY_SETTINGS = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.0-flash-exp",
        safety_settings: list[dict[str, str]] | None = None,
    ):
        """
        Initialize Gemini LLM provider.

        Args:
            api_key: Google AI API key
            model: Default model to use (gemini-2.0-flash-exp, gemini-1.5-pro, etc.)
            safety_settings: Custom safety settings (uses relaxed defaults if None)
        """
        super().__init__(default_model=model)
        self._api_key = api_key
        self._safety_settings = safety_settings or self.DEFAULT_SAFETY_SETTINGS
        self._genai = None  # Lazy initialization
        self._model_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "gemini"

    def _configure_genai(self):
        """Configure the generativeai module."""
        if self._genai is None:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self._api_key)
                self._genai = genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required for Gemini LLM. "
                    "Install with: pip install google-generativeai"
                )
        return self._genai

    def _get_model(
        self,
        model_name: str,
        system_instruction: str | None = None,
        json_mode: bool = False,
    ):
        """Get or create a Gemini model instance."""
        genai = self._configure_genai()

        # Create cache key
        cache_key = f"{model_name}:{system_instruction or ''}:{json_mode}"

        if cache_key not in self._model_cache:
            # Build generation config
            generation_config = {}
            if json_mode:
                generation_config["response_mime_type"] = "application/json"

            # Build model config
            model_config = {
                "safety_settings": self._safety_settings,
            }

            if generation_config:
                model_config["generation_config"] = generation_config

            if system_instruction:
                model_config["system_instruction"] = system_instruction

            self._model_cache[cache_key] = genai.GenerativeModel(
                model_name,
                **model_config,
            )

        return self._model_cache[cache_key]

    def _convert_messages(self, messages: list[Message]) -> tuple[str, list[dict[str, Any]]]:
        """
        Convert Message objects to Gemini format.

        Gemini uses a different format:
        - System messages become system_instruction
        - user/assistant become user/model roles
        - Content is in parts array

        Returns:
            Tuple of (system_instruction, conversation_history)
        """
        system_instruction = ""
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                # Accumulate system messages
                if system_instruction:
                    system_instruction += "\n\n"
                system_instruction += msg.content
            elif msg.role == MessageRole.USER:
                conversation.append(
                    {
                        "role": "user",
                        "parts": [msg.content],
                    }
                )
            elif msg.role == MessageRole.ASSISTANT:
                conversation.append(
                    {
                        "role": "model",
                        "parts": [msg.content],
                    }
                )

        return system_instruction, conversation

    def _extract_usage(self, response: Any) -> dict[str, int]:
        """Extract token usage from Gemini response."""
        usage = {}
        try:
            if hasattr(response, "usage_metadata"):
                metadata = response.usage_metadata
                usage = {
                    "input_tokens": getattr(metadata, "prompt_token_count", 0),
                    "output_tokens": getattr(metadata, "candidates_token_count", 0),
                    "total_tokens": getattr(metadata, "total_token_count", 0),
                }
        except Exception as e:
            logger.debug(f"Could not extract usage metadata: {e}")
        return usage

    def _extract_finish_reason(self, response: Any) -> str:
        """Extract finish reason from Gemini response."""
        try:
            if response.candidates:
                reason = response.candidates[0].finish_reason
                # Map Gemini finish reasons to standard format
                reason_map = {
                    1: "stop",  # STOP
                    2: "length",  # MAX_TOKENS
                    3: "safety",  # SAFETY
                    4: "recitation",  # RECITATION
                    5: "other",  # OTHER
                }
                if isinstance(reason, int):
                    return reason_map.get(reason, "stop")
                return str(reason).lower()
        except (AttributeError, IndexError):
            pass
        return "stop"

    async def complete(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> LLMResponse:
        """
        Generate a completion using Gemini.

        Args:
            messages: List of conversation messages
            config: Optional configuration overrides

        Returns:
            LLMResponse with generated content
        """
        if config is None:
            config = LLMConfig()

        try:
            # Convert messages to Gemini format
            system_instruction, conversation = self._convert_messages(messages)

            # Determine if JSON mode
            json_mode = config.response_format == "json"

            # Get model instance
            model_name = config.model or self.default_model
            model = self._get_model(model_name, system_instruction, json_mode)

            # Build generation config
            generation_config = {
                "temperature": config.temperature,
                "max_output_tokens": config.max_tokens,
                "top_p": config.top_p,
            }

            # Generate response
            logger.debug(
                f"Gemini completion: model={model_name}, "
                f"messages={len(conversation)}, json_mode={json_mode}"
            )

            # Use async generation
            response = await model.generate_content_async(
                conversation,
                generation_config=generation_config,
            )

            # Extract content
            content = ""
            try:
                content = response.text
            except ValueError:
                # Handle blocked responses
                logger.warning("Gemini response was blocked or empty")
                if response.candidates:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text"):
                            content += part.text

            # Extract metadata
            usage = self._extract_usage(response)
            finish_reason = self._extract_finish_reason(response)

            return LLMResponse(
                content=content,
                model=model_name,
                usage=usage,
                finish_reason=finish_reason,
                provider=self.name,
            )

        except Exception as e:
            logger.error(f"Gemini completion error: {e}", exc_info=True)
            raise

    async def complete_json(
        self,
        messages: list[Message],
        config: LLMConfig | None = None,
    ) -> dict[str, Any]:
        """
        Generate a JSON completion using Gemini.

        Gemini has native JSON mode support for structured outputs.

        Args:
            messages: List of conversation messages
            config: Optional configuration overrides

        Returns:
            Parsed JSON response
        """
        if config is None:
            config = LLMConfig()
        config.response_format = "json"

        response = await self.complete(messages, config)

        # Parse JSON response
        content = response.content.strip()

        # Handle markdown code blocks
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]

        return json.loads(content.strip())

    async def count_tokens(self, messages: list[Message]) -> int:
        """
        Count tokens for a list of messages.

        Args:
            messages: List of messages to count tokens for

        Returns:
            Total token count
        """
        try:
            genai = self._configure_genai()

            # Convert messages
            system_instruction, conversation = self._convert_messages(messages)

            # Get model for token counting
            model = genai.GenerativeModel(self.default_model)

            # Build content for counting
            content = []
            if system_instruction:
                content.append({"role": "user", "parts": [system_instruction]})
            content.extend(conversation)

            # Count tokens
            result = model.count_tokens(content)
            return result.total_tokens

        except Exception as e:
            logger.error(f"Token counting error: {e}")
            raise


class GeminiFlashProvider(GeminiLLMProvider):
    """
    Convenience class for Gemini 2.0 Flash.

    Optimized for speed and cost-effectiveness.
    """

    def __init__(self, api_key: str):
        super().__init__(api_key, model="gemini-2.0-flash-exp")


class GeminiProProvider(GeminiLLMProvider):
    """
    Convenience class for Gemini 1.5 Pro.

    Best for complex tasks requiring reasoning.
    """

    def __init__(self, api_key: str):
        super().__init__(api_key, model="gemini-1.5-pro")


__all__ = [
    "GeminiFlashProvider",
    "GeminiLLMProvider",
    "GeminiProProvider",
]
