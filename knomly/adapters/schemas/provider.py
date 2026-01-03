"""
Provider Definition Schema.

JSON-serializable schema for defining service providers (STT, LLM, TTS, Chat)
that can be stored in databases and instantiated at runtime.

Design Principle:
    Provider configuration is separated from provider implementation.
    A ProviderDefinition specifies WHAT provider to use and HOW to configure it,
    but the actual class instantiation is handled by a ServiceFactory.

Registry Pattern:
    Similar to SERVICE_CONFIG patterns in other frameworks, we provide the
    same data in a JSON-serializable format that can be stored in a database.
    The ServiceFactory then uses a registry to resolve classes.

Usage:
    # From JSON (database)
    provider_def = ProviderDefinition.model_validate({
        "provider_type": "stt",
        "provider_code": "deepgram",
        "params": {
            "model": "nova-2",
            "language": "en",
        },
        "auth_secret_key": "deepgram_api_key",
    })

    # To provider instance (via factory)
    factory = GenericServiceFactory(registry=SERVICE_REGISTRY)
    stt = factory.create_service(provider_def, secrets={"deepgram_api_key": "..."})
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class ProviderDefinition(BaseModel):
    """
    Configuration for a single service provider.

    This schema captures everything needed to instantiate a provider:
    1. What type of provider (stt, llm, tts, chat)
    2. Which provider implementation (deepgram, openai, elevenlabs)
    3. Configuration parameters (model, language, etc.)
    4. How to authenticate (secret key reference)

    The actual class mapping (provider_code → Python class) is handled
    by the ServiceFactory's registry, not stored here.

    Attributes:
        provider_type: Service category (stt, llm, tts, chat, vad)
        provider_code: Provider implementation (deepgram, openai, etc.)
        params: Provider-specific configuration
        auth_secret_key: Key in secrets dict for authentication
        enabled: Whether this provider is active
        priority: For fallback ordering (higher = preferred)

    Example:
        {
            "provider_type": "llm",
            "provider_code": "openai",
            "params": {
                "model": "gpt-4o",
                "temperature": 0.7,
                "max_tokens": 4096
            },
            "auth_secret_key": "openai_api_key"
        }
    """

    provider_type: Literal["stt", "llm", "tts", "chat", "vad"] = Field(
        ...,
        description="Type of service provider",
    )
    provider_code: str = Field(
        ...,
        description="Provider implementation code (e.g., 'deepgram', 'openai')",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Provider-specific configuration parameters",
    )
    auth_secret_key: str | None = Field(
        default=None,
        description="Key in secrets dict for authentication",
    )
    enabled: bool = Field(default=True, description="Whether provider is active")
    priority: int = Field(default=0, description="Fallback priority (higher = preferred)")

    class Config:
        populate_by_name = True

    def get_auth(self, secrets: dict[str, str]) -> str | None:
        """
        Get authentication value from secrets.

        Args:
            secrets: Dict of secret values (e.g., from ToolContext.secrets)

        Returns:
            Auth value if found, None otherwise
        """
        if self.auth_secret_key is None:
            return None
        return secrets.get(self.auth_secret_key)

    @classmethod
    def stt(
        cls,
        provider_code: str,
        model: str | None = None,
        language: str = "en",
        auth_secret_key: str | None = None,
        **params: Any,
    ) -> "ProviderDefinition":
        """Factory for STT provider definition."""
        all_params = {"language": language, **params}
        if model:
            all_params["model"] = model

        return cls(
            provider_type="stt",
            provider_code=provider_code,
            params=all_params,
            auth_secret_key=auth_secret_key,
        )

    @classmethod
    def llm(
        cls,
        provider_code: str,
        model: str = "gpt-4o",
        temperature: float = 0.7,
        auth_secret_key: str | None = None,
        **params: Any,
    ) -> "ProviderDefinition":
        """Factory for LLM provider definition."""
        return cls(
            provider_type="llm",
            provider_code=provider_code,
            params={
                "model": model,
                "temperature": temperature,
                **params,
            },
            auth_secret_key=auth_secret_key,
        )

    @classmethod
    def tts(
        cls,
        provider_code: str,
        voice_id: str | None = None,
        model: str | None = None,
        auth_secret_key: str | None = None,
        **params: Any,
    ) -> "ProviderDefinition":
        """Factory for TTS provider definition."""
        all_params = {**params}
        if voice_id:
            all_params["voice_id"] = voice_id
        if model:
            all_params["model"] = model

        return cls(
            provider_type="tts",
            provider_code=provider_code,
            params=all_params,
            auth_secret_key=auth_secret_key,
        )

    @classmethod
    def chat(
        cls,
        provider_code: str,
        auth_secret_key: str | None = None,
        **params: Any,
    ) -> "ProviderDefinition":
        """Factory for Chat provider definition."""
        return cls(
            provider_type="chat",
            provider_code=provider_code,
            params=params,
            auth_secret_key=auth_secret_key,
        )
