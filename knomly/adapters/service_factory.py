"""
Generic Service Factory.

Creates service instances (STT, LLM, TTS, Chat) from ProviderDefinition
using a registry pattern.

Design Principle:
    The factory is configuration-driven. The registry maps provider codes
    to class configurations, and the factory handles instantiation.

Registry Pattern:
    Similar to SERVICE_CONFIG patterns in other frameworks, we make the
    registry injectable, allowing different configurations per deployment.

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    GenericServiceFactory                      │
    │                                                               │
    │   Input: ProviderDefinition + secrets                        │
    │                                                               │
    │   1. Look up provider_code in registry                       │
    │   2. Get class and configuration                             │
    │   3. Handle authentication from secrets                      │
    │   4. Build constructor arguments                             │
    │   5. Instantiate and return                                  │
    │                                                               │
    │   Output: Service instance (STT, LLM, TTS, Chat)             │
    └──────────────────────────────────────────────────────────────┘

Usage:
    # Define registry
    registry = DictServiceRegistry({
        "stt": {
            "deepgram": {
                "class": DeepgramSTT,
                "auth": {"arg": "api_key"},
                "params_class": DeepgramOptions,
                "params_arg": "options",
                "direct_args": ["model"],
            },
        },
        "llm": {
            "openai": {
                "class": OpenAILLM,
                "auth": {"arg": "api_key"},
                "direct_args": ["model", "temperature"],
            },
        },
    })

    # Create factory
    factory = GenericServiceFactory(registry)

    # Create services
    stt = factory.create_service(
        ProviderDefinition.stt("deepgram", model="nova-2"),
        secrets={"deepgram_api_key": "..."},
    )
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, TypeVar

from .base import ServiceRegistry, DictServiceRegistry

if TYPE_CHECKING:
    from .schemas import ProviderDefinition

logger = logging.getLogger(__name__)

T = TypeVar("T")


class ServiceClassMapping:
    """
    Configuration for a single service class.

    This dataclass-like structure holds all information needed
    to instantiate a provider class.

    Attributes:
        cls: The Python class to instantiate
        auth: Authentication configuration
            - "arg": Constructor argument name for auth
            - "env_var": Environment variable (optional fallback)
        params_class: Nested params class (e.g., DeepgramLiveOptions)
        params_arg: Constructor argument name for params instance
        direct_args: List of params passed directly to constructor
        extra_args: Static extra arguments to always pass

    Example:
        ServiceClassMapping(
            cls=DeepgramSTT,
            auth={"arg": "api_key"},
            params_class=DeepgramLiveOptions,
            params_arg="options",
            direct_args=["model", "language"],
        )
    """

    def __init__(
        self,
        cls: type,
        auth: dict[str, str] | None = None,
        params_class: type | None = None,
        params_arg: str | None = None,
        direct_args: list[str] | None = None,
        extra_args: dict[str, Any] | None = None,
    ):
        """
        Initialize mapping.

        Args:
            cls: The class to instantiate
            auth: Auth configuration {"arg": "api_key", "env_var": "..."}
            params_class: Nested params class
            params_arg: Constructor arg name for params
            direct_args: Args passed directly from params
            extra_args: Static extra args
        """
        self.cls = cls
        self.auth = auth or {}
        self.params_class = params_class
        self.params_arg = params_arg
        self.direct_args = direct_args or []
        self.extra_args = extra_args or {}

    def to_dict(self) -> dict[str, Any]:
        """Convert to dict for registry storage."""
        result: dict[str, Any] = {"class": self.cls}
        if self.auth:
            result["auth"] = self.auth
        if self.params_class:
            result["params_class"] = self.params_class
        if self.params_arg:
            result["params_arg"] = self.params_arg
        if self.direct_args:
            result["direct_args"] = self.direct_args
        if self.extra_args:
            result["extra_args"] = self.extra_args
        return result


class GenericServiceFactory:
    """
    Generic factory for creating service instances from configuration.

    This factory:
    1. Looks up provider_code in the registry
    2. Extracts authentication from secrets
    3. Builds constructor arguments from params
    4. Instantiates the service class

    The registry pattern allows different configurations per deployment
    without changing the factory code.

    Usage:
        registry = DictServiceRegistry({...})
        factory = GenericServiceFactory(registry)

        stt = factory.create_service(
            ProviderDefinition.stt("deepgram", model="nova-2"),
            secrets={"deepgram_api_key": "..."},
        )
    """

    def __init__(
        self,
        registry: ServiceRegistry | None = None,
    ):
        """
        Initialize factory.

        Args:
            registry: Service registry mapping provider codes to classes
        """
        self._registry = registry or DictServiceRegistry({})

    @property
    def registry(self) -> ServiceRegistry:
        """Get the service registry."""
        return self._registry

    def create_service(
        self,
        definition: "ProviderDefinition",
        secrets: dict[str, str] | None = None,
        **extra_kwargs: Any,
    ) -> Any | None:
        """
        Create a service instance from definition.

        Args:
            definition: Provider configuration
            secrets: Credential map for authentication
            **extra_kwargs: Additional constructor arguments

        Returns:
            Service instance or None if provider not supported
        """
        secrets = secrets or {}

        # Look up in registry
        config = self._registry.get_config(
            definition.provider_type,
            definition.provider_code,
        )

        if config is None:
            logger.warning(
                f"[service_factory] Unknown provider: "
                f"{definition.provider_type}/{definition.provider_code}"
            )
            return None

        service_class = config.get("class")
        if service_class is None:
            logger.error(
                f"[service_factory] No class defined for: "
                f"{definition.provider_type}/{definition.provider_code}"
            )
            return None

        try:
            # Build constructor arguments
            constructor_args = self._build_constructor_args(
                config=config,
                definition=definition,
                secrets=secrets,
            )

            # Add extra runtime arguments
            constructor_args.update(extra_kwargs)

            # Add static extra args from config
            static_extra = config.get("extra_args", {})
            constructor_args.update(static_extra)

            logger.info(
                f"[service_factory] Creating {definition.provider_type} service: "
                f"{service_class.__name__}"
            )
            logger.debug(f"[service_factory] Args: {list(constructor_args.keys())}")

            # Instantiate
            return service_class(**constructor_args)

        except Exception as e:
            logger.error(
                f"[service_factory] Failed to create {service_class.__name__}: {e}",
                exc_info=True,
            )
            return None

    def _build_constructor_args(
        self,
        config: dict[str, Any],
        definition: "ProviderDefinition",
        secrets: dict[str, str],
    ) -> dict[str, Any]:
        """
        Build constructor arguments from config and definition.

        Args:
            config: Registry config for this provider
            definition: Provider definition with params
            secrets: Credentials map

        Returns:
            Dict of constructor arguments
        """
        constructor_args: dict[str, Any] = {}
        params_data = definition.params.copy()

        # 1. Handle authentication
        auth_config = config.get("auth", {})
        if auth_config:
            auth_arg = auth_config.get("arg")
            if auth_arg:
                # Try to get from secrets using definition's auth_secret_key
                auth_value = None
                if definition.auth_secret_key:
                    auth_value = secrets.get(definition.auth_secret_key)

                # Fallback to env var if specified
                if not auth_value:
                    env_var = auth_config.get("env_var")
                    if env_var:
                        import os
                        auth_value = os.getenv(env_var)

                if auth_value:
                    constructor_args[auth_arg] = auth_value

        # 2. Handle direct arguments (passed to constructor)
        direct_args = config.get("direct_args", [])
        for arg_name in direct_args:
            if arg_name in params_data:
                constructor_args[arg_name] = params_data.pop(arg_name)

        # 3. Handle nested params class
        params_class = config.get("params_class")
        params_arg = config.get("params_arg")

        if params_class and params_arg:
            # Build params instance from remaining params
            try:
                params_instance = params_class(**params_data)
                constructor_args[params_arg] = params_instance
            except Exception as e:
                logger.warning(
                    f"[service_factory] Failed to create params: {e}"
                )
                # Fall back to passing params directly
                constructor_args.update(params_data)
        else:
            # No params class - pass remaining params directly
            constructor_args.update(params_data)

        return constructor_args

    def list_providers(self, service_type: str) -> list[str]:
        """List available provider codes for a service type."""
        return self._registry.list_providers(service_type)

    def supports_provider(
        self,
        service_type: str,
        provider_code: str,
    ) -> bool:
        """Check if a provider is supported."""
        config = self._registry.get_config(service_type, provider_code)
        return config is not None


# =============================================================================
# Convenience: Create factory from Knomly's existing providers
# =============================================================================


def create_knomly_service_registry() -> DictServiceRegistry:
    """
    Create a service registry for Knomly's built-in providers.

    This maps provider codes to Knomly's existing provider classes.
    Extend this registry for additional providers.

    Returns:
        Configured DictServiceRegistry
    """
    # Import providers lazily to avoid circular imports
    from knomly.providers.stt.deepgram import DeepgramSTTProvider
    from knomly.providers.stt.gemini import GeminiSTTProvider
    from knomly.providers.stt.whisper import WhisperSTTProvider
    from knomly.providers.llm.openai import OpenAILLMProvider
    from knomly.providers.llm.gemini import GeminiLLMProvider
    from knomly.providers.chat.zulip import ZulipChatProvider

    return DictServiceRegistry({
        "stt": {
            "deepgram": {
                "class": DeepgramSTTProvider,
                "auth": {"arg": "api_key", "env_var": "DEEPGRAM_API_KEY"},
                "direct_args": ["model", "language"],
            },
            "gemini": {
                "class": GeminiSTTProvider,
                "auth": {"arg": "api_key", "env_var": "GEMINI_API_KEY"},
                "direct_args": ["model", "language"],
            },
            "whisper": {
                "class": WhisperSTTProvider,
                "auth": {"arg": "api_key", "env_var": "OPENAI_API_KEY"},
                "direct_args": ["model", "language"],
            },
        },
        "llm": {
            "openai": {
                "class": OpenAILLMProvider,
                "auth": {"arg": "api_key", "env_var": "OPENAI_API_KEY"},
                "direct_args": ["model", "temperature", "max_tokens"],
            },
            "gemini": {
                "class": GeminiLLMProvider,
                "auth": {"arg": "api_key", "env_var": "GEMINI_API_KEY"},
                "direct_args": ["model", "temperature", "max_tokens"],
            },
        },
        "chat": {
            "zulip": {
                "class": ZulipChatProvider,
                "auth": {"arg": "api_key"},
                "direct_args": ["email", "site"],
            },
        },
    })


def create_knomly_service_factory() -> GenericServiceFactory:
    """
    Create a GenericServiceFactory with Knomly's providers.

    Returns:
        Configured GenericServiceFactory
    """
    return GenericServiceFactory(registry=create_knomly_service_registry())
