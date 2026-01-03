"""
Dependency Injection for Knomly.

Provides singleton instances of services and pipeline.

Architecture (v3):
    The dependency layer now supports TWO modes:
    1. Static Mode (legacy): Uses PipelineFactory for hardcoded processor chains
    2. Dynamic Mode (v3): Uses PipelineResolver for database-driven config

    The Twilio webhook checks for resolver first, falls back to static pipeline.
"""
from __future__ import annotations

import logging
import os
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

    Initializes providers on first call.
    """
    global _providers
    if _providers is None:
        settings = get_settings()
        _providers = ProviderRegistry()

        # Register STT providers
        if settings.gemini_api_key:
            _providers.register_stt(
                "gemini",
                GeminiSTTProvider(api_key=settings.gemini_api_key),
            )

        # Register LLM providers
        if settings.openai_api_key:
            _providers.register_llm(
                "openai",
                OpenAILLMProvider(api_key=settings.openai_api_key),
            )
        if settings.anthropic_api_key:
            _providers.register_llm(
                "anthropic",
                AnthropicLLMProvider(api_key=settings.anthropic_api_key),
            )

        # Register Chat providers
        if settings.zulip_bot_email and settings.zulip_api_key:
            _providers.register_chat(
                "zulip",
                ZulipChatProvider(
                    site=settings.zulip_site,
                    bot_email=settings.zulip_bot_email,
                    api_key=settings.zulip_api_key,
                ),
            )

        # Set defaults
        if settings.default_stt_provider in _providers._stt_providers:
            _providers.set_default_stt(settings.default_stt_provider)
        if settings.default_llm_provider in _providers._llm_providers:
            _providers.set_default_llm(settings.default_llm_provider)
        if settings.default_chat_provider in _providers._chat_providers:
            _providers.set_default_chat(settings.default_chat_provider)

    return _providers


def get_config_service() -> ConfigurationService:
    """
    Get the configuration service.

    Initializes MongoDB connection on first call.
    """
    global _config_service
    if _config_service is None:
        settings = get_settings()
        _config_service = ConfigurationService(
            mongodb_url=settings.mongodb_url,
            database_name=settings.mongodb_database,
        )
    return _config_service


def get_pipeline() -> Pipeline:
    """
    Get the configured standup pipeline.

    Creates pipeline on first call.
    """
    global _pipeline
    if _pipeline is None:
        settings = get_settings()
        _pipeline = create_standup_pipeline(settings)
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

    Returns:
        PipelineResolver instance

    Example:
        resolver = get_resolver()
        packet = await resolver.resolve_for_user(user_id)
        tools = await resolver.build_tools_only(packet, secrets)
    """
    global _resolver

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
    _resolver = PipelineResolver(
        loader=loader,
        service_factory=service_factory,
        tool_builder=tool_builder,
    )

    logger.info("[resolver] PipelineResolver initialized")

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

    # For now, return global secrets from environment
    # In production, fetch user-specific secrets from vault
    return {
        # OpenAI / LLM
        "openai_api_key": settings.openai_api_key or "",
        "anthropic_api_key": settings.anthropic_api_key or "",
        "gemini_api_key": settings.gemini_api_key or "",
        # Integrations
        "plane_api_key": os.getenv("KNOMLY_PLANE_API_KEY", ""),
        "zulip_api_key": settings.zulip_api_key or "",
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
