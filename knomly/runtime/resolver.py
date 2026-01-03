"""
Pipeline Resolver.

The "Ignition System" that connects webhooks to dynamic pipeline configuration.

Design Principle:
    The Resolver is the single entry point for loading configuration.
    It abstracts away WHERE config comes from (file, database, API)
    and provides a consistent interface for the webhook layer.

Flow:
    1. Webhook extracts user_id from request
    2. Resolver.resolve_for_user(user_id) loads PipelinePacket
    3. Resolver.build_pipeline(packet, secrets) creates live Pipeline
    4. Webhook executes pipeline with initial frame

Caching:
    - First check cache (Redis/memory)
    - On miss, load from primary source (DB/file)
    - Store in cache for future requests

Usage:
    # Setup (at application startup)
    resolver = PipelineResolver(
        loader=MongoDefinitionLoader(db),
        service_factory=GenericServiceFactory(registry),
        tool_builder=ToolBuilder(adapters),
        cache=RedisPipelineCache(redis),  # optional
    )

    # In webhook handler
    @router.post("/webhook/twilio")
    async def handle(request: Request):
        user_id = extract_user_id(request)

        # Load config (cached or fresh)
        packet = await resolver.resolve_for_user(user_id)

        # Get user secrets (from vault or request)
        secrets = await get_secrets_for_user(user_id)

        # Build and execute
        pipeline = await resolver.build_pipeline(packet, secrets)
        result = await pipeline.execute(initial_frame)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from knomly.adapters.schemas import PipelinePacket, ToolDefinition
    from knomly.adapters.base import DefinitionLoader, ToolBuilder
    from knomly.adapters.service_factory import GenericServiceFactory
    from knomly.pipeline import Pipeline
    from knomly.tools.base import Tool
    from knomly.tools.factory import ToolContext

logger = logging.getLogger(__name__)


class PipelineCache(Protocol):
    """Protocol for pipeline packet caching."""

    async def get(self, key: str) -> "PipelinePacket | None":
        """Get cached packet by key."""
        ...

    async def set(self, key: str, packet: "PipelinePacket", ttl_seconds: int = 3600) -> None:
        """Store packet in cache."""
        ...


class InMemoryPipelineCache:
    """Simple in-memory cache for testing/development."""

    def __init__(self):
        self._cache: dict[str, "PipelinePacket"] = {}

    async def get(self, key: str) -> "PipelinePacket | None":
        return self._cache.get(key)

    async def set(self, key: str, packet: "PipelinePacket", ttl_seconds: int = 3600) -> None:
        self._cache[key] = packet

    def clear(self) -> None:
        self._cache.clear()


class PipelineResolver:
    """
    Resolves and builds pipelines from dynamic configuration.

    This is the central coordination point that:
    1. Loads PipelinePacket from storage (database/file)
    2. Caches resolved packets for performance
    3. Builds live Pipeline instances from packets
    4. Handles inheritance/defaults

    The resolver is stateless regarding user data - all user-specific
    information flows through the ToolContext/secrets at build time.

    Example:
        resolver = PipelineResolver(
            loader=MongoDefinitionLoader(db),
            service_factory=create_knomly_service_factory(),
            tool_builder=ToolBuilder({"openapi": OpenAPIToolAdapter()}),
        )

        # Resolve configuration
        packet = await resolver.resolve_for_user("user-123")

        # Build live pipeline
        pipeline = await resolver.build_pipeline(
            packet,
            secrets={"openai_api_key": "...", "plane_api_key": "..."},
        )

        # Execute
        result = await pipeline.execute(initial_frame)
    """

    def __init__(
        self,
        *,
        loader: "DefinitionLoader",
        service_factory: "GenericServiceFactory | None" = None,
        tool_builder: "ToolBuilder | None" = None,
        cache: PipelineCache | None = None,
        default_packet: "PipelinePacket | None" = None,
    ):
        """
        Initialize resolver.

        Args:
            loader: Loads definitions from storage
            service_factory: Builds providers from definitions
            tool_builder: Builds tools from definitions
            cache: Optional cache for resolved packets
            default_packet: Fallback packet if none found
        """
        self._loader = loader
        self._service_factory = service_factory
        self._tool_builder = tool_builder
        self._cache = cache or InMemoryPipelineCache()
        self._default_packet = default_packet

    async def resolve_for_user(
        self,
        user_id: str,
        *,
        session_id: str | None = None,
        use_cache: bool = True,
        **context: Any,
    ) -> "PipelinePacket":
        """
        Resolve pipeline configuration for a user.

        Args:
            user_id: User/tenant identifier
            session_id: Optional session ID (for session-specific config)
            use_cache: Whether to use cache
            **context: Additional context (avatar_id, etc.)

        Returns:
            PipelinePacket with full configuration

        Raises:
            ValueError: If no config found and no default
        """
        cache_key = self._build_cache_key(user_id, session_id)

        # 1. Check cache
        if use_cache:
            cached = await self._cache.get(cache_key)
            if cached is not None:
                logger.debug(f"[resolver] Cache hit: {cache_key}")
                return cached

        # 2. Load from primary source
        logger.info(f"[resolver] Loading config for user={user_id}")

        packet = await self._loader.get_pipeline_for_session(
            session_id=session_id or f"default-{user_id}",
            user_id=user_id,
            **context,
        )

        # 3. Apply defaults if needed
        if packet is None:
            if self._default_packet is not None:
                logger.info("[resolver] Using default packet")
                packet = self._default_packet
            else:
                raise ValueError(
                    f"No pipeline configuration found for user '{user_id}' "
                    "and no default configured"
                )

        # 4. Cache for future requests
        await self._cache.set(cache_key, packet)

        logger.info(
            f"[resolver] Resolved config | "
            f"user={user_id} | "
            f"tools={len(packet.tools)} | "
            f"providers={self._count_providers(packet)}"
        )

        return packet

    async def resolve_tools_for_user(
        self,
        user_id: str,
    ) -> list["ToolDefinition"]:
        """
        Resolve just tool definitions for a user.

        Useful when you want tools without full pipeline config.

        Args:
            user_id: User/tenant identifier

        Returns:
            List of ToolDefinitions
        """
        return await self._loader.get_tools_for_user(user_id)

    async def build_pipeline(
        self,
        packet: "PipelinePacket",
        secrets: dict[str, str],
        *,
        extra_tools: list["Tool"] | None = None,
    ) -> "Pipeline":
        """
        Build a live Pipeline from a resolved packet.

        This is where configuration becomes execution:
        1. Build providers from ProviderDefinitions
        2. Build tools from ToolDefinitions
        3. Construct processors
        4. Assemble pipeline

        Args:
            packet: Resolved PipelinePacket
            secrets: User's credentials (API keys)
            extra_tools: Additional static tools to include

        Returns:
            Executable Pipeline
        """
        from knomly.pipeline import PipelineBuilder, PipelineContext
        from knomly.providers import ProviderRegistry
        from knomly.tools.factory import ToolContext

        logger.info(
            f"[resolver] Building pipeline | "
            f"session={packet.session.session_id} | "
            f"user={packet.session.user_id}"
        )

        # 1. Build providers
        providers = ProviderRegistry()

        if packet.providers.stt and self._service_factory:
            stt = self._service_factory.create_service(packet.providers.stt, secrets)
            if stt:
                providers.register_stt(packet.providers.stt.provider_code, stt)

        if packet.providers.llm and self._service_factory:
            llm = self._service_factory.create_service(packet.providers.llm, secrets)
            if llm:
                providers.register_llm(packet.providers.llm.provider_code, llm)

        if packet.providers.chat and self._service_factory:
            chat = self._service_factory.create_service(packet.providers.chat, secrets)
            if chat:
                providers.register_chat(packet.providers.chat.provider_code, chat)

        # 2. Build tools
        tools: list["Tool"] = []

        if self._tool_builder and packet.tools:
            tool_context = ToolContext(
                user_id=packet.session.user_id,
                secrets=secrets,
                metadata=packet.session.metadata,
            )

            tools = await self._tool_builder.build_tools(
                packet.get_enabled_tools(),
                tool_context,
            )

        if extra_tools:
            tools.extend(extra_tools)

        # 3. Build pipeline context
        context = PipelineContext(
            session_id=packet.session.session_id,
            user_id=packet.session.user_id,
            providers=providers,
            metadata={
                "system_prompt": packet.agent.system_prompt,
                "welcome_message": packet.agent.welcome_message,
                "voice_id": packet.agent.voice_id,
                "locale": packet.session.locale,
                **packet.session.metadata,
            },
        )

        # 4. Build pipeline
        #    Note: The actual processor chain depends on your use case.
        #    This is a minimal example - you'd customize for your workflow.
        builder = PipelineBuilder(context=context)

        logger.info(
            f"[resolver] Pipeline built | "
            f"providers={providers} | "
            f"tools={len(tools)}"
        )

        return builder.build()

    async def build_tools_only(
        self,
        packet: "PipelinePacket",
        secrets: dict[str, str],
    ) -> list["Tool"]:
        """
        Build just the tools from a packet.

        Useful when integrating with existing pipeline code
        that handles provider setup separately.

        Args:
            packet: Resolved PipelinePacket
            secrets: User's credentials

        Returns:
            List of live Tool instances
        """
        from knomly.tools.factory import ToolContext

        if not self._tool_builder:
            return []

        tool_context = ToolContext(
            user_id=packet.session.user_id,
            secrets=secrets,
            metadata=packet.session.metadata,
        )

        return await self._tool_builder.build_tools(
            packet.get_enabled_tools(),
            tool_context,
        )

    def _build_cache_key(
        self,
        user_id: str,
        session_id: str | None,
    ) -> str:
        """Build cache key from identifiers."""
        if session_id:
            return f"pipeline:{user_id}:{session_id}"
        return f"pipeline:{user_id}:default"

    def _count_providers(self, packet: "PipelinePacket") -> int:
        """Count configured providers in packet."""
        count = 0
        if packet.providers.stt:
            count += 1
        if packet.providers.llm:
            count += 1
        if packet.providers.tts:
            count += 1
        if packet.providers.chat:
            count += 1
        return count


# =============================================================================
# Convenience: Create resolver with common setup
# =============================================================================


def create_resolver(
    *,
    loader: "DefinitionLoader | None" = None,
    config_dir: str | None = None,
    cache: PipelineCache | None = None,
) -> PipelineResolver:
    """
    Create a PipelineResolver with common setup.

    Args:
        loader: Custom loader (uses FileDefinitionLoader if None)
        config_dir: Directory for file-based config
        cache: Optional cache implementation

    Returns:
        Configured PipelineResolver
    """
    from knomly.adapters.service_factory import create_knomly_service_factory
    from knomly.adapters.openapi_adapter import OpenAPIToolAdapter
    from knomly.adapters.base import ToolBuilder
    from .loaders import FileDefinitionLoader, MemoryDefinitionLoader

    # Default loader
    if loader is None:
        if config_dir:
            loader = FileDefinitionLoader(config_dir)
        else:
            loader = MemoryDefinitionLoader()

    # Setup tool builder
    tool_builder = ToolBuilder(
        adapters={
            "openapi": OpenAPIToolAdapter(),
            # Add more adapters as needed
        },
        loader=loader,
    )

    return PipelineResolver(
        loader=loader,
        service_factory=create_knomly_service_factory(),
        tool_builder=tool_builder,
        cache=cache,
    )
