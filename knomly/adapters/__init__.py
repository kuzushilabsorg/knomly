"""
Knomly Adapters Layer.

This layer provides portable, JSON-serializable schemas for tools, providers,
and pipelines. It enables database-driven configuration where tool definitions
and pipeline configs are loaded at runtime rather than hardcoded.

Design Principle:
    "Configuration determines Execution."

    Instead of writing Python code for each tool/provider, you define them
    as JSON/YAML documents stored in a database. At runtime, adapters
    convert these definitions into live Tool/Provider instances.

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    CORE FRAMEWORK                            │
    │                                                              │
    │   Schemas (JSON-serializable):                              │
    │   ├── ToolDefinition      - Tool name, params, metadata     │
    │   ├── ProviderDefinition  - Provider type, code, params     │
    │   └── PipelinePacket      - Full pipeline config            │
    │                                                              │
    │   Protocols:                                                 │
    │   ├── ToolAdapter         - Schema → Live Tool              │
    │   ├── ServiceFactory      - Config → Provider Instance      │
    │   └── DefinitionLoader    - Database → Schemas              │
    └─────────────────────────────────────────────────────────────┘
                               │
                               ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 IMPLEMENTATION LAYER                         │
    │                                                              │
    │   Loaders (database-specific):                              │
    │   ├── MongoDefinitionLoader                                 │
    │   └── RedisPipelineCache                                    │
    │                                                              │
    │   Concrete Factories:                                        │
    │   ├── SERVICE_CONFIG mapping                                │
    │   └── Domain-specific adapters                              │
    └─────────────────────────────────────────────────────────────┘

Comparison to Pipecat:
    Pipecat's adapters/schemas/ provides FunctionSchema and ToolsSchema.
    We extend this with ProviderDefinition and PipelinePacket for
    full database-driven pipeline configuration.

Usage:
    # Define tool in JSON (stored in database)
    tool_def = ToolDefinition(
        name="create_task",
        description="Create a task in the project",
        parameters={
            "type": "object",
            "properties": {
                "project": {"type": "string"},
                "title": {"type": "string"},
            },
            "required": ["project", "title"],
        },
        source="openapi",
        source_config={"spec_url": "...", "operation_id": "createTask"},
    )

    # Convert to live tool at runtime
    adapter = OpenAPIToolAdapter()
    tool = await adapter.build_tool(tool_def, context)

    # Or load from database
    loader = MongoDefinitionLoader(db)
    tool_defs = await loader.get_tools_for_user(user_id)
"""

from .schemas import (
    ToolDefinition,
    ToolParameter,
    ProviderDefinition,
    PipelinePacket,
    SessionContext,
    AgentContext,
    PipelineProviderConfig,
)

from .base import (
    ToolAdapter,
    ServiceFactory,
    DefinitionLoader,
    ServiceRegistry,
)

from .service_factory import (
    GenericServiceFactory,
    ServiceClassMapping,
)

from .openapi_adapter import (
    OpenAPIToolAdapter,
    OpenAPISpecImporter,
    SpecCache,
    get_openapi_adapter,
    import_openapi_tools,
)

__all__ = [
    # Schemas
    "ToolDefinition",
    "ToolParameter",
    "ProviderDefinition",
    "PipelinePacket",
    "SessionContext",
    "AgentContext",
    "PipelineProviderConfig",
    # Protocols
    "ToolAdapter",
    "ServiceFactory",
    "DefinitionLoader",
    "ServiceRegistry",
    # Implementations
    "GenericServiceFactory",
    "ServiceClassMapping",
    # OpenAPI Adapter
    "OpenAPIToolAdapter",
    "OpenAPISpecImporter",
    "SpecCache",
    "get_openapi_adapter",
    "import_openapi_tools",
]
