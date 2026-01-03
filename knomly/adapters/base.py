"""
Adapter Protocols.

Abstract base classes and protocols for the adapter layer.
These define the contracts that implementations must follow.

Design Principle:
    Protocols define WHAT, implementations define HOW.
    This allows the core framework to remain database-agnostic
    while downstream projects provide concrete implementations.

Protocols:
    - ToolAdapter: Convert ToolDefinition → live Tool
    - ServiceFactory: Convert ProviderDefinition → live Provider
    - DefinitionLoader: Load definitions from storage
    - ServiceRegistry: Map provider codes to classes

Usage:
    # In core framework (this file)
    class ToolAdapter(Protocol):
        async def build_tool(self, definition, context): ...

    # In implementation layer
    class OpenAPIToolAdapter:
        async def build_tool(self, definition, context):
            # Fetch OpenAPI spec, build tool
            ...

    # In downstream project
    class MongoDefinitionLoader:
        async def get_tools(self, user_id):
            # Query MongoDB for tools
            ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Protocol, Sequence, TypeVar, runtime_checkable

if TYPE_CHECKING:
    from .schemas import ToolDefinition, ProviderDefinition, PipelinePacket
    from knomly.tools.base import Tool
    from knomly.tools.factory import ToolContext


T = TypeVar("T")


# =============================================================================
# Tool Adapter Protocol
# =============================================================================


@runtime_checkable
class ToolAdapter(Protocol):
    """
    Protocol for converting ToolDefinition to live Tool instances.

    A ToolAdapter knows how to take a JSON-serializable ToolDefinition
    and create a functioning Tool that can be used by an agent.

    Implementations:
        - OpenAPIToolAdapter: Builds tools from OpenAPI operations
        - NativeToolAdapter: Imports Python classes by name
        - MCPToolAdapter: Builds MCP-protocol tools

    Example:
        class OpenAPIToolAdapter:
            async def build_tool(
                self,
                definition: ToolDefinition,
                context: ToolContext,
            ) -> Tool:
                spec = await self._fetch_spec(definition.source_config["spec_url"])
                toolkit = OpenAPIToolkit.from_spec(
                    spec,
                    auth={"api_key": context.get_secret(
                        definition.source_config["auth_secret_key"]
                    )},
                )
                return toolkit.get_tool(definition.source_config["operation_id"])
    """

    async def build_tool(
        self,
        definition: "ToolDefinition",
        context: "ToolContext",
    ) -> "Tool":
        """
        Build a live Tool from a definition.

        Args:
            definition: JSON-serializable tool definition
            context: User-specific context with credentials

        Returns:
            Live Tool instance ready for use

        Raises:
            ValueError: If definition is invalid
            RuntimeError: If tool cannot be built
        """
        ...

    def supports_source(self, source: str) -> bool:
        """
        Check if this adapter handles a given source type.

        Args:
            source: Source type (openapi, native, mcp, function)

        Returns:
            True if this adapter handles the source
        """
        ...


class BaseToolAdapter(ABC):
    """
    Abstract base class for tool adapters.

    Provides common functionality for tool adapters.
    """

    @property
    @abstractmethod
    def source_type(self) -> str:
        """The source type this adapter handles."""
        ...

    def supports_source(self, source: str) -> bool:
        """Check if this adapter handles a given source type."""
        return source == self.source_type

    @abstractmethod
    async def build_tool(
        self,
        definition: "ToolDefinition",
        context: "ToolContext",
    ) -> "Tool":
        """Build a live Tool from a definition."""
        ...

    async def build_tools(
        self,
        definitions: Sequence["ToolDefinition"],
        context: "ToolContext",
    ) -> list["Tool"]:
        """
        Build multiple tools from definitions.

        Args:
            definitions: List of tool definitions
            context: User-specific context

        Returns:
            List of live Tool instances
        """
        tools = []
        for defn in definitions:
            if defn.enabled and self.supports_source(defn.source):
                tool = await self.build_tool(defn, context)
                tools.append(tool)
        return tools


# =============================================================================
# Service Factory Protocol
# =============================================================================


@runtime_checkable
class ServiceFactory(Protocol[T]):
    """
    Protocol for creating service instances from configuration.

    A ServiceFactory takes a ProviderDefinition and creates the
    corresponding provider instance (STT, LLM, TTS, Chat).

    The factory uses a registry to map provider codes to classes,
    allowing the registry to be configured at runtime.

    Example:
        factory = GenericServiceFactory(registry=MY_REGISTRY)

        # Create from definition
        stt = factory.create_service(
            ProviderDefinition.stt("deepgram", model="nova-2"),
            secrets={"deepgram_api_key": "..."},
        )
    """

    def create_service(
        self,
        definition: "ProviderDefinition",
        secrets: dict[str, str] | None = None,
        **kwargs: Any,
    ) -> T | None:
        """
        Create a service instance from definition.

        Args:
            definition: Provider configuration
            secrets: Credential map for authentication
            **kwargs: Additional constructor arguments

        Returns:
            Service instance or None if provider not supported
        """
        ...


# =============================================================================
# Service Registry Protocol
# =============================================================================


class ServiceClassMapping(Protocol):
    """
    Mapping from provider code to class configuration.

    This is the runtime registry that maps provider_code (e.g., "deepgram")
    to the information needed to instantiate it:
    - class: The Python class
    - auth: How to authenticate
    - params_class: Nested params class (if any)
    - direct_args: Args passed directly to constructor

    Example:
        SERVICE_REGISTRY = {
            "stt": {
                "deepgram": {
                    "class": DeepgramSTTService,
                    "auth": {"arg": "api_key", "env_var": "DEEPGRAM_API_KEY"},
                    "params_class": DeepgramLiveOptions,
                    "params_arg": "options",
                    "direct_args": ["model"],
                },
            },
        }
    """

    def get_config(
        self,
        service_type: str,
        provider_code: str,
    ) -> dict[str, Any] | None:
        """
        Get configuration for a provider.

        Args:
            service_type: Type of service (stt, llm, tts, chat)
            provider_code: Provider code (deepgram, openai, etc.)

        Returns:
            Config dict or None if not found
        """
        ...

    def list_providers(self, service_type: str) -> list[str]:
        """List available provider codes for a service type."""
        ...


class DictServiceRegistry:
    """
    Simple dict-based service registry.

    Uses a nested dict structure:
        {
            "stt": {
                "deepgram": {"class": ..., "auth": ...},
                "whisper": {"class": ..., "auth": ...},
            },
            "llm": {
                "openai": {"class": ..., "auth": ...},
                "anthropic": {"class": ..., "auth": ...},
            },
        }

    Example:
        registry = DictServiceRegistry({
            "stt": {
                "deepgram": {
                    "class": DeepgramSTT,
                    "auth": {"arg": "api_key"},
                },
            },
        })
    """

    def __init__(self, config: dict[str, dict[str, dict[str, Any]]]):
        """
        Initialize with config dict.

        Args:
            config: Nested dict of service type → provider code → config
        """
        self._config = config

    def get_config(
        self,
        service_type: str,
        provider_code: str,
    ) -> dict[str, Any] | None:
        """Get configuration for a provider."""
        type_config = self._config.get(service_type, {})
        return type_config.get(provider_code)

    def list_providers(self, service_type: str) -> list[str]:
        """List available provider codes for a service type."""
        return list(self._config.get(service_type, {}).keys())

    def register(
        self,
        service_type: str,
        provider_code: str,
        config: dict[str, Any],
    ) -> None:
        """Register a new provider."""
        if service_type not in self._config:
            self._config[service_type] = {}
        self._config[service_type][provider_code] = config

    def unregister(self, service_type: str, provider_code: str) -> None:
        """Unregister a provider."""
        if service_type in self._config:
            self._config[service_type].pop(provider_code, None)


# Alias for backwards compatibility
ServiceRegistry = DictServiceRegistry


# =============================================================================
# Definition Loader Protocol
# =============================================================================


@runtime_checkable
class DefinitionLoader(Protocol):
    """
    Protocol for loading definitions from storage.

    A DefinitionLoader abstracts the database layer, allowing
    the core framework to remain database-agnostic.

    Implementations:
        - MongoDefinitionLoader: Load from MongoDB
        - PostgresDefinitionLoader: Load from PostgreSQL
        - FileDefinitionLoader: Load from YAML/JSON files

    Example:
        class MongoDefinitionLoader:
            def __init__(self, db):
                self._db = db

            async def get_tools_for_user(self, user_id):
                docs = await self._db.tools.find({
                    "$or": [
                        {"user_id": user_id},
                        {"global": True},
                    ],
                    "enabled": True,
                }).to_list()
                return [ToolDefinition.model_validate(d) for d in docs]
    """

    async def get_tools_for_user(
        self,
        user_id: str,
    ) -> list["ToolDefinition"]:
        """
        Get tool definitions available to a user.

        Args:
            user_id: User/tenant identifier

        Returns:
            List of enabled tool definitions
        """
        ...

    async def get_pipeline_for_session(
        self,
        session_id: str,
        user_id: str,
        **context: Any,
    ) -> "PipelinePacket | None":
        """
        Get pipeline configuration for a session.

        Args:
            session_id: Session identifier
            user_id: User/tenant identifier
            **context: Additional context (e.g., avatar_id)

        Returns:
            PipelinePacket if found, None otherwise
        """
        ...

    async def get_provider_definition(
        self,
        provider_type: str,
        provider_code: str,
        user_id: str | None = None,
    ) -> "ProviderDefinition | None":
        """
        Get provider definition.

        Args:
            provider_type: Service type (stt, llm, tts, chat)
            provider_code: Provider code (deepgram, openai, etc.)
            user_id: Optional user for user-specific config

        Returns:
            ProviderDefinition if found, None otherwise
        """
        ...


# =============================================================================
# Tool Builder (combines Adapter + Loader)
# =============================================================================


class ToolBuilder:
    """
    High-level tool builder that combines adapters and loaders.

    This is the main entry point for building tools from database
    definitions at runtime.

    Usage:
        builder = ToolBuilder(
            adapters={
                "openapi": OpenAPIToolAdapter(),
                "native": NativeToolAdapter(),
            },
            loader=MongoDefinitionLoader(db),
        )

        # Build all tools for a user
        tools = await builder.build_tools_for_user("user-123", context)
    """

    def __init__(
        self,
        adapters: dict[str, ToolAdapter],
        loader: DefinitionLoader | None = None,
    ):
        """
        Initialize tool builder.

        Args:
            adapters: Map of source type to adapter
            loader: Optional definition loader for database access
        """
        self._adapters = adapters
        self._loader = loader

    def get_adapter(self, source: str) -> ToolAdapter | None:
        """Get adapter for a source type."""
        return self._adapters.get(source)

    async def build_tool(
        self,
        definition: "ToolDefinition",
        context: "ToolContext",
    ) -> "Tool | None":
        """
        Build a single tool from definition.

        Args:
            definition: Tool definition
            context: User context with credentials

        Returns:
            Live Tool or None if no adapter found
        """
        adapter = self.get_adapter(definition.source)
        if adapter is None:
            return None
        return await adapter.build_tool(definition, context)

    async def build_tools(
        self,
        definitions: Sequence["ToolDefinition"],
        context: "ToolContext",
    ) -> list["Tool"]:
        """
        Build multiple tools from definitions.

        Args:
            definitions: List of tool definitions
            context: User context with credentials

        Returns:
            List of live Tools (skips unsupported sources)
        """
        tools = []
        for defn in definitions:
            if not defn.enabled:
                continue
            tool = await self.build_tool(defn, context)
            if tool is not None:
                tools.append(tool)
        return tools

    async def build_tools_for_user(
        self,
        user_id: str,
        context: "ToolContext",
    ) -> list["Tool"]:
        """
        Build all tools available to a user.

        Requires a loader to be configured.

        Args:
            user_id: User/tenant identifier
            context: User context with credentials

        Returns:
            List of live Tools

        Raises:
            RuntimeError: If no loader configured
        """
        if self._loader is None:
            raise RuntimeError("No loader configured for ToolBuilder")

        definitions = await self._loader.get_tools_for_user(user_id)
        return await self.build_tools(definitions, context)
