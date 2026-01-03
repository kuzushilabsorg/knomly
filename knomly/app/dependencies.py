"""
Dependency Injection for Knomly.

Provides singleton instances of services and pipeline.

Architecture (v3):
    The dependency layer now supports TWO modes:
    1. Static Mode (legacy): Uses PipelineFactory for hardcoded processor chains
    2. Dynamic Mode (v3): Uses PipelineResolver for database-driven config

    The Twilio webhook checks for resolver first, falls back to static pipeline.

Thread Safety:
    Global instances are protected by locks to prevent race conditions
    during concurrent initialization (e.g., multiple webhook requests).
"""
from __future__ import annotations

import logging
import os
import threading
from functools import lru_cache
from pathlib import Path
from typing import Optional

from knomly.config import ConfigurationService
from knomly.config.schemas import AppSettings
from knomly.pipeline import Pipeline
from knomly.pipeline.builder import create_standup_pipeline
from knomly.providers import ProviderRegistry
from knomly.providers.chat.zulip import ZulipChatProvider
from knomly.providers.llm.openai import AnthropicLLMProvider, OpenAILLMProvider
from knomly.providers.stt.gemini import GeminiSTTProvider

logger = logging.getLogger(__name__)

# Thread locks for safe singleton initialization
_providers_lock = threading.Lock()
_config_service_lock = threading.Lock()
_pipeline_lock = threading.Lock()
_resolver_lock = threading.Lock()


@lru_cache()
def get_settings() -> AppSettings:
    """
    Get application settings from environment.

    Uses lru_cache for singleton pattern.
    """
    return AppSettings(
        # Service
        service_name=os.getenv("KNOMLY_SERVICE_NAME", "knomly"),
        environment=os.getenv("KNOMLY_ENVIRONMENT", "development"),
        debug=os.getenv("KNOMLY_DEBUG", "false").lower() == "true",
        # MongoDB
        mongodb_url=os.getenv("KNOMLY_MONGODB_URL", "mongodb://localhost:27017"),
        mongodb_database=os.getenv("KNOMLY_MONGODB_DATABASE", "knomly"),
        # Provider API keys
        gemini_api_key=os.getenv("KNOMLY_GEMINI_API_KEY", ""),
        openai_api_key=os.getenv("KNOMLY_OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("KNOMLY_ANTHROPIC_API_KEY"),
        # Zulip
        zulip_site=os.getenv("KNOMLY_ZULIP_SITE", ""),
        zulip_bot_email=os.getenv("KNOMLY_ZULIP_BOT_EMAIL", ""),
        zulip_api_key=os.getenv("KNOMLY_ZULIP_API_KEY", ""),
        # Twilio
        twilio_account_sid=os.getenv("KNOMLY_TWILIO_ACCOUNT_SID", ""),
        twilio_auth_token=os.getenv("KNOMLY_TWILIO_AUTH_TOKEN", ""),
        twilio_whatsapp_number=os.getenv("KNOMLY_TWILIO_WHATSAPP_NUMBER", ""),
        # Provider selection
        default_stt_provider=os.getenv("KNOMLY_DEFAULT_STT_PROVIDER", "gemini"),
        default_llm_provider=os.getenv("KNOMLY_DEFAULT_LLM_PROVIDER", "openai"),
        default_chat_provider=os.getenv("KNOMLY_DEFAULT_CHAT_PROVIDER", "zulip"),
    )


# Global instances (initialized on first access)
_providers: Optional[ProviderRegistry] = None
_config_service: Optional[ConfigurationService] = None
_pipeline: Optional[Pipeline] = None


def get_providers() -> ProviderRegistry:
    """
    Get the provider registry with configured providers.

    Initializes providers on first call using double-checked locking
    for thread safety.
    """
    global _providers
    # Fast path: already initialized
    if _providers is not None:
        return _providers

    # Slow path: need to initialize with lock
    with _providers_lock:
        # Double-check after acquiring lock
        if _providers is not None:
            return _providers

        settings = get_settings()
        registry = ProviderRegistry()

        # Register STT providers
        # Use .get_secret_value() to extract the actual string from SecretStr
        gemini_key = settings.gemini_api_key.get_secret_value()
        if gemini_key:
            registry.register_stt(
                "gemini",
                GeminiSTTProvider(api_key=gemini_key),
            )

        # Register LLM providers
        openai_key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if openai_key:
            registry.register_llm(
                "openai",
                OpenAILLMProvider(api_key=openai_key),
            )
        anthropic_key = settings.anthropic_api_key.get_secret_value() if settings.anthropic_api_key else ""
        if anthropic_key:
            registry.register_llm(
                "anthropic",
                AnthropicLLMProvider(api_key=anthropic_key),
            )

        # Register Chat providers
        zulip_key = settings.zulip_api_key.get_secret_value()
        if settings.zulip_bot_email and zulip_key:
            registry.register_chat(
                "zulip",
                ZulipChatProvider(
                    site=settings.zulip_site,
                    bot_email=settings.zulip_bot_email,
                    api_key=zulip_key,
                ),
            )

        # Set defaults
        if settings.default_stt_provider in registry._stt_providers:
            registry.set_default_stt(settings.default_stt_provider)
        if settings.default_llm_provider in registry._llm_providers:
            registry.set_default_llm(settings.default_llm_provider)
        if settings.default_chat_provider in registry._chat_providers:
            registry.set_default_chat(settings.default_chat_provider)

        # Assign to global only after fully initialized
        _providers = registry

    return _providers


def get_config_service() -> ConfigurationService:
    """
    Get the configuration service.

    Initializes MongoDB connection on first call using double-checked
    locking for thread safety.
    """
    global _config_service
    # Fast path
    if _config_service is not None:
        return _config_service

    # Slow path with lock
    with _config_service_lock:
        if _config_service is not None:
            return _config_service

        settings = get_settings()
        service = ConfigurationService(
            mongodb_url=settings.mongodb_url.get_secret_value(),
            database_name=settings.mongodb_database,
        )
        _config_service = service

    return _config_service


def get_pipeline() -> Pipeline:
    """
    Get the configured standup pipeline.

    Creates pipeline on first call using double-checked locking
    for thread safety.
    """
    global _pipeline
    # Fast path
    if _pipeline is not None:
        return _pipeline

    # Slow path with lock
    with _pipeline_lock:
        if _pipeline is not None:
            return _pipeline

        settings = get_settings()
        pipeline = create_standup_pipeline(settings)
        _pipeline = pipeline

    return _pipeline


async def initialize_services() -> None:
    """
    Initialize all services on application startup.

    Called from FastAPI lifespan.
    """
    # Initialize providers
    get_providers()

    # Initialize config service and connect to MongoDB
    config_service = get_config_service()
    await config_service.connect()

    # Seed default prompts
    await config_service.seed_default_prompts()


async def shutdown_services() -> None:
    """
    Cleanup all services on application shutdown.

    Called from FastAPI lifespan.
    """
    global _config_service
    if _config_service:
        await _config_service.close()
        _config_service = None


# =============================================================================
# v3 Dynamic Configuration (PipelineResolver)
# =============================================================================

# Global resolver instance
_resolver: Optional["PipelineResolver"] = None


def get_resolver() -> "PipelineResolver":
    """
    Get the PipelineResolver for dynamic configuration.

    The resolver loads tool definitions and pipeline configs from:
    - File system (development): config/ directory
    - Database (production): MongoDB collections

    Uses double-checked locking for thread safety.

    Returns:
        PipelineResolver instance

    Example:
        resolver = get_resolver()
        packet = await resolver.resolve_for_user(user_id)
        tools = await resolver.build_tools_only(packet, secrets)
    """
    global _resolver

    # Fast path
    if _resolver is not None:
        return _resolver

    # Slow path with lock
    with _resolver_lock:
        if _resolver is not None:
            return _resolver

        from knomly.runtime import PipelineResolver, FileDefinitionLoader, MemoryDefinitionLoader
        from knomly.adapters.base import ToolBuilder
        from knomly.adapters.openapi_adapter import OpenAPIToolAdapter
        from knomly.adapters.service_factory import create_knomly_service_factory

        settings = get_settings()

        # Determine loader based on environment
        config_dir = os.getenv("KNOMLY_CONFIG_DIR", "config")
        config_path = Path(config_dir)

        if config_path.exists() and (config_path / "pipelines").exists():
            logger.info(f"[resolver] Using FileDefinitionLoader: {config_path}")
            loader = FileDefinitionLoader(config_path)
        else:
            logger.info("[resolver] Using MemoryDefinitionLoader (no config dir found)")
            loader = MemoryDefinitionLoader()
            # Could seed default config here

        # Create tool builder with adapters
        tool_builder = ToolBuilder(
            adapters={
                "openapi": OpenAPIToolAdapter(),
                # Add more adapters as needed:
                # "native": NativeToolAdapter(),
                # "mcp": MCPToolAdapter(),
            },
            loader=loader,
        )

        # Create service factory
        service_factory = create_knomly_service_factory()

        # Create resolver
        resolver = PipelineResolver(
            loader=loader,
            service_factory=service_factory,
            tool_builder=tool_builder,
        )

        logger.info("[resolver] PipelineResolver initialized")
        _resolver = resolver

    return _resolver


def get_user_secrets(user_id: str) -> dict[str, str]:
    """
    Get secrets for a user from environment or vault.

    This is the secret resolution layer. In production, this would
    fetch from a vault service. For development, we use environment
    variables.

    Args:
        user_id: User identifier

    Returns:
        Dict of secret key -> value mappings
    """
    settings = get_settings()

    # Helper to safely extract secret value
    def _get_secret(secret_str) -> str:
        if secret_str is None:
            return ""
        if hasattr(secret_str, "get_secret_value"):
            return secret_str.get_secret_value()
        return str(secret_str)

    # For now, return global secrets from environment
    # In production, fetch user-specific secrets from vault
    return {
        # OpenAI / LLM
        "openai_api_key": _get_secret(settings.openai_api_key),
        "anthropic_api_key": _get_secret(settings.anthropic_api_key),
        "gemini_api_key": _get_secret(settings.gemini_api_key),
        # Integrations
        "plane_api_key": os.getenv("KNOMLY_PLANE_API_KEY", ""),
        "zulip_api_key": _get_secret(settings.zulip_api_key),
        # STT/TTS
        "deepgram_api_key": os.getenv("KNOMLY_DEEPGRAM_API_KEY", ""),
        "elevenlabs_api_key": os.getenv("KNOMLY_ELEVENLABS_API_KEY", ""),
    }


async def resolve_tools_for_user(user_id: str) -> list:
    """
    Convenience function to resolve tools for a user.

    Combines resolver + secret loading in one call.

    Args:
        user_id: User identifier

    Returns:
        List of live Tool instances ready for use

    Example:
        tools = await resolve_tools_for_user("user-123")
        for tool in tools:
            result = await tool.execute(params)
    """
    resolver = get_resolver()
    secrets = get_user_secrets(user_id)

    # Get tool definitions for user
    tool_defs = await resolver.resolve_tools_for_user(user_id)

    if not tool_defs:
        logger.debug(f"[resolver] No tools defined for user={user_id}")
        return []

    # Build the tools with user secrets
    from knomly.tools.factory import ToolContext

    context = ToolContext(
        user_id=user_id,
        secrets=secrets,
    )

    # Use tool builder to create live tools
    if resolver._tool_builder:
        return await resolver._tool_builder.build_tools(tool_defs, context)

    return []


# Type hint for forward reference
if True:  # Avoid circular import at module level
    from knomly.runtime import PipelineResolver
