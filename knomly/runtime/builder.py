"""
Runtime Builder.

Provides a simplified builder for constructing pipelines at runtime
from PipelinePacket configurations.

This is a thin wrapper that delegates to PipelineResolver.build_pipeline()
but can be used standalone when you already have a resolved packet.

Usage:
    # With resolver (recommended)
    resolver = PipelineResolver(loader=..., service_factory=..., tool_builder=...)
    packet = await resolver.resolve_for_user(user_id)
    pipeline = await resolver.build_pipeline(packet, secrets)

    # Standalone builder (when packet is already resolved)
    builder = RuntimeBuilder(service_factory=..., tool_builder=...)
    pipeline = await builder.build(packet, secrets)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knomly.adapters.base import ToolBuilder
    from knomly.adapters.schemas import PipelinePacket
    from knomly.adapters.service_factory import GenericServiceFactory
    from knomly.pipeline import Pipeline
    from knomly.providers.registry import ProviderRegistry
    from knomly.tools.base import Tool

logger = logging.getLogger(__name__)


class RuntimeBuilder:
    """
    Builds live Pipeline instances from PipelinePacket configurations.

    This is a stateless builder that converts resolved configuration
    into executable pipeline components.

    Example:
        builder = RuntimeBuilder(
            service_factory=create_knomly_service_factory(),
            tool_builder=ToolBuilder(adapters),
        )

        pipeline = await builder.build(packet, secrets)
        result = await pipeline.execute(initial_frame)

    Note:
        For most use cases, prefer using PipelineResolver which handles
        loading, caching, and building in one interface.
    """

    def __init__(
        self,
        *,
        service_factory: GenericServiceFactory | None = None,
        tool_builder: ToolBuilder | None = None,
    ):
        """
        Initialize builder.

        Args:
            service_factory: Factory for creating provider instances
            tool_builder: Builder for creating tool instances
        """
        self._service_factory = service_factory
        self._tool_builder = tool_builder

    async def build(
        self,
        packet: PipelinePacket,
        secrets: dict[str, str],
        *,
        extra_tools: list[Tool] | None = None,
    ) -> Pipeline:
        """
        Build a live Pipeline from a PipelinePacket.

        Args:
            packet: Resolved pipeline configuration
            secrets: User credentials (API keys)
            extra_tools: Additional tools to include

        Returns:
            Executable Pipeline instance
        """
        from knomly.pipeline import PipelineBuilder, PipelineContext

        logger.info(
            f"[runtime_builder] Building pipeline | "
            f"session={packet.session.session_id} | "
            f"user={packet.session.user_id}"
        )

        # 1. Build providers
        providers = await self._build_providers(packet, secrets)

        # 2. Build tools
        tools = await self._build_tools(packet, secrets)
        if extra_tools:
            tools.extend(extra_tools)

        # 3. Build context
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
        builder = PipelineBuilder(context=context)

        logger.info(
            f"[runtime_builder] Pipeline built | providers={providers} | tools={len(tools)}"
        )

        return builder.build()

    async def build_tools_only(
        self,
        packet: PipelinePacket,
        secrets: dict[str, str],
    ) -> list[Tool]:
        """
        Build just the tools from a packet.

        Useful for integrating with existing pipeline code.

        Args:
            packet: Resolved pipeline configuration
            secrets: User credentials

        Returns:
            List of live Tool instances
        """
        return await self._build_tools(packet, secrets)

    async def build_providers_only(
        self,
        packet: PipelinePacket,
        secrets: dict[str, str],
    ) -> ProviderRegistry:
        """
        Build just the providers from a packet.

        Useful for integrating with existing pipeline code.

        Args:
            packet: Resolved pipeline configuration
            secrets: User credentials

        Returns:
            ProviderRegistry with configured providers
        """
        return await self._build_providers(packet, secrets)

    async def _build_providers(
        self,
        packet: PipelinePacket,
        secrets: dict[str, str],
    ) -> ProviderRegistry:
        """Build provider registry from packet configuration."""
        from knomly.providers import ProviderRegistry

        providers = ProviderRegistry()

        if not self._service_factory:
            logger.warning("[runtime_builder] No service factory, skipping providers")
            return providers

        # STT Provider
        if packet.providers.stt:
            stt = self._service_factory.create_service(packet.providers.stt, secrets)
            if stt:
                providers.register_stt(packet.providers.stt.provider_code, stt)
                logger.debug(
                    f"[runtime_builder] Registered STT: {packet.providers.stt.provider_code}"
                )

        # LLM Provider
        if packet.providers.llm:
            llm = self._service_factory.create_service(packet.providers.llm, secrets)
            if llm:
                providers.register_llm(packet.providers.llm.provider_code, llm)
                logger.debug(
                    f"[runtime_builder] Registered LLM: {packet.providers.llm.provider_code}"
                )

        # TTS Provider
        if packet.providers.tts:
            tts = self._service_factory.create_service(packet.providers.tts, secrets)
            if tts:
                providers.register_tts(packet.providers.tts.provider_code, tts)
                logger.debug(
                    f"[runtime_builder] Registered TTS: {packet.providers.tts.provider_code}"
                )

        # Chat Provider
        if packet.providers.chat:
            chat = self._service_factory.create_service(packet.providers.chat, secrets)
            if chat:
                providers.register_chat(packet.providers.chat.provider_code, chat)
                logger.debug(
                    f"[runtime_builder] Registered Chat: {packet.providers.chat.provider_code}"
                )

        return providers

    async def _build_tools(
        self,
        packet: PipelinePacket,
        secrets: dict[str, str],
    ) -> list[Tool]:
        """Build tools from packet configuration."""
        from knomly.tools.factory import ToolContext

        if not self._tool_builder:
            logger.warning("[runtime_builder] No tool builder, skipping tools")
            return []

        if not packet.tools:
            return []

        tool_context = ToolContext(
            user_id=packet.session.user_id,
            secrets=secrets,
            metadata=packet.session.metadata,
        )

        enabled_tools = packet.get_enabled_tools()
        tools = await self._tool_builder.build_tools(enabled_tools, tool_context)

        logger.info(
            f"[runtime_builder] Built {len(tools)} tools from {len(enabled_tools)} definitions"
        )
        return tools
