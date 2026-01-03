"""
PipelineResolver Usage Examples.

This module demonstrates how to use the dynamic configuration system
to load pipeline configurations from files or databases at runtime.

Architecture:
    ┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
    │   Webhook       │ ──▶  │ PipelineResolver │ ──▶  │   Pipeline      │
    │   (Request)     │      │ (Load + Build)   │      │   (Execute)     │
    └─────────────────┘      └──────────────────┘      └─────────────────┘
           │                         │
           │                         ▼
           │                 ┌──────────────────┐
           │                 │ DefinitionLoader │
           │                 │ (File/Mongo/API) │
           │                 └──────────────────┘

The key insight: Webhooks no longer need static builders.
Configuration lives in JSON/DB and is loaded dynamically.
"""

import asyncio
from pathlib import Path


# =============================================================================
# Example 1: Basic Setup with FileDefinitionLoader (Development)
# =============================================================================

async def example_basic_file_loader():
    """
    Simplest setup: Load configs from JSON files.

    Directory structure:
        config/
        ├── tools/
        │   └── plane.json          # Plane API tools
        ├── pipelines/
        │   ├── default.json        # Default pipeline config
        │   └── user-123.json       # User-specific overrides
        └── providers/
            └── default.json        # Provider definitions
    """
    from knomly.runtime import PipelineResolver, FileDefinitionLoader

    # 1. Create loader pointing to config directory
    loader = FileDefinitionLoader("config/")

    # 2. Create resolver with loader
    resolver = PipelineResolver(loader=loader)

    # 3. Resolve configuration for a user
    packet = await resolver.resolve_for_user(
        user_id="user-123",
        session_id="session-abc",
    )

    print(f"Loaded pipeline for: {packet.session.user_id}")
    print(f"System prompt: {packet.agent.system_prompt[:50]}...")
    print(f"Tools: {[t.name for t in packet.tools]}")

    return packet


# =============================================================================
# Example 2: Memory Loader for Testing
# =============================================================================

async def example_memory_loader_for_testing():
    """
    In-memory loader for unit tests - no files needed.
    """
    from knomly.runtime import PipelineResolver, MemoryDefinitionLoader
    from knomly.adapters.schemas import (
        PipelinePacket,
        SessionContext,
        AgentContext,
        PipelineProviderConfig,
        ProviderDefinition,
        ToolDefinition,
    )

    # 1. Create in-memory loader
    loader = MemoryDefinitionLoader()

    # 2. Add tools programmatically
    await loader.add_tool(
        ToolDefinition(
            name="create_task",
            description="Create a task in the project",
            parameters={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Task title"},
                    "description": {"type": "string"},
                },
                "required": ["title"],
            },
            source="openapi",
            source_config={
                "spec_url": "https://api.plane.so/openapi.json",
                "operation_id": "createIssue",
                "auth_type": "bearer",
                "auth_secret_key": "plane_api_key",
            },
        )
    )

    # 3. Add default pipeline
    await loader.set_default_pipeline(
        PipelinePacket(
            session=SessionContext(
                session_id="default",
                user_id="default",
            ),
            agent=AgentContext(
                system_prompt="You are a helpful project manager assistant.",
                welcome_message="Hello! I can help you manage tasks.",
            ),
            providers=PipelineProviderConfig(
                llm=ProviderDefinition.llm("openai", model="gpt-4o"),
                stt=ProviderDefinition.stt("deepgram", model="nova-2"),
            ),
            tools=[],  # Will be populated from loader
        )
    )

    # 4. Create resolver and use it
    resolver = PipelineResolver(loader=loader)

    packet = await resolver.resolve_for_user("any-user")
    tools = await resolver.resolve_tools_for_user("any-user")

    print(f"Tools available: {[t.name for t in tools]}")

    return resolver


# =============================================================================
# Example 3: Full Webhook Integration Pattern
# =============================================================================

async def example_webhook_integration():
    """
    How to wire PipelineResolver into a webhook handler.

    This is the "Cutover" pattern - replacing static builders with dynamic resolution.
    """
    from knomly.runtime import PipelineResolver, FileDefinitionLoader
    from knomly.adapters import GenericServiceFactory
    from knomly.adapters.base import ToolBuilder
    from knomly.adapters.openapi_adapter import OpenAPIToolAdapter

    # ==========================================================================
    # Application Startup (run once)
    # ==========================================================================

    # 1. Create loader (file-based for dev, MongoDB for prod)
    loader = FileDefinitionLoader("config/")

    # 2. Create tool builder with adapters
    tool_builder = ToolBuilder(
        adapters={
            "openapi": OpenAPIToolAdapter(),
            # Add more: "native": NativeToolAdapter(), "mcp": MCPToolAdapter()
        },
        loader=loader,
    )

    # 3. Create service factory for providers
    from knomly.adapters.service_factory import create_knomly_service_factory
    service_factory = create_knomly_service_factory()

    # 4. Create resolver with all dependencies
    resolver = PipelineResolver(
        loader=loader,
        service_factory=service_factory,
        tool_builder=tool_builder,
    )

    # ==========================================================================
    # Per-Request Handling (runs for each webhook)
    # ==========================================================================

    async def handle_webhook(user_id: str, session_id: str, secrets: dict):
        """
        Handle a webhook request with dynamic pipeline resolution.

        Args:
            user_id: Extracted from phone number or auth token
            session_id: Generated or from request
            secrets: User's API keys from vault
        """
        # 1. Resolve configuration (loads from file/DB, uses cache)
        packet = await resolver.resolve_for_user(
            user_id=user_id,
            session_id=session_id,
        )

        # 2. Build live pipeline with user's credentials
        pipeline = await resolver.build_pipeline(
            packet=packet,
            secrets=secrets,  # {"plane_api_key": "...", "openai_api_key": "..."}
        )

        # 3. Execute (would normally be in background task)
        # result = await pipeline.execute(initial_frame)

        return pipeline

    # Simulate webhook handling
    pipeline = await handle_webhook(
        user_id="user-123",
        session_id="session-abc",
        secrets={
            "plane_api_key": "sk-test-plane-key",
            "openai_api_key": "sk-test-openai-key",
        },
    )

    print(f"Built pipeline for session: {pipeline}")


# =============================================================================
# Example 4: Building Tools Only (Integration with Existing Pipeline)
# =============================================================================

async def example_tools_only():
    """
    When you only need tools (not full pipeline).

    Useful for integrating with existing pipeline code.
    """
    from knomly.runtime import PipelineResolver, MemoryDefinitionLoader
    from knomly.adapters.schemas import ToolDefinition
    from knomly.adapters.base import ToolBuilder
    from knomly.adapters.openapi_adapter import OpenAPIToolAdapter

    # Setup
    loader = MemoryDefinitionLoader()
    await loader.add_tool(
        ToolDefinition(
            name="list_issues",
            description="List all issues",
            parameters={"type": "object", "properties": {}},
            source="openapi",
            source_config={
                "spec_url": "https://api.plane.so/openapi.json",
                "operation_id": "listIssues",
                "auth_secret_key": "plane_api_key",
            },
            read_only=True,  # Mark as read-only
        )
    )

    tool_builder = ToolBuilder(
        adapters={"openapi": OpenAPIToolAdapter()},
        loader=loader,
    )

    resolver = PipelineResolver(
        loader=loader,
        tool_builder=tool_builder,
    )

    # Resolve just tool definitions (no building yet)
    tool_defs = await resolver.resolve_tools_for_user("user-123")
    print(f"Tool definitions: {[t.name for t in tool_defs]}")

    # Or resolve full packet and build tools from it
    # This would require a default pipeline to be set


# =============================================================================
# Example 5: Custom Loader Implementation (MongoDB)
# =============================================================================

class MongoDefinitionLoader:
    """
    Example MongoDB loader implementation.

    Implements the DefinitionLoader protocol for production use.
    """

    def __init__(self, db):
        """
        Args:
            db: Motor AsyncIOMotorDatabase instance
        """
        self.db = db

    async def get_tools_for_user(self, user_id: str):
        """Load tools from MongoDB."""
        from knomly.adapters.schemas import ToolDefinition

        # Query tools collection
        cursor = self.db.tools.find({
            "$or": [
                {"user_id": user_id},
                {"user_id": {"$exists": False}},  # Global tools
            ],
            "enabled": True,
        })

        tools = []
        async for doc in cursor:
            tools.append(ToolDefinition.model_validate(doc))

        return tools

    async def get_pipeline_for_session(
        self,
        session_id: str,
        user_id: str,
        **context,
    ):
        """Load pipeline config from MongoDB."""
        from knomly.adapters.schemas import PipelinePacket

        # Try user-specific pipeline
        doc = await self.db.pipelines.find_one({"user_id": user_id})

        if doc is None:
            # Fall back to default
            doc = await self.db.pipelines.find_one({"user_id": "default"})

        if doc is None:
            return None

        # Update session context
        doc["session"]["session_id"] = session_id
        doc["session"]["user_id"] = user_id

        return PipelinePacket.model_validate(doc)

    async def get_provider_definition(
        self,
        provider_type: str,
        provider_code: str,
        user_id: str | None = None,
    ):
        """Load provider config from MongoDB."""
        from knomly.adapters.schemas import ProviderDefinition

        doc = await self.db.providers.find_one({
            "provider_type": provider_type,
            "provider_code": provider_code,
        })

        if doc:
            return ProviderDefinition.model_validate(doc)
        return None


async def example_mongodb_production():
    """
    Production setup with MongoDB.

    Requires motor[asyncio] package.
    """
    # from motor.motor_asyncio import AsyncIOMotorClient
    #
    # client = AsyncIOMotorClient("mongodb://localhost:27017")
    # db = client.knomly
    #
    # loader = MongoDefinitionLoader(db)
    # resolver = PipelineResolver(
    #     loader=loader,
    #     service_factory=...,
    #     tool_builder=...,
    # )
    pass


# =============================================================================
# Example 6: Cache Control
# =============================================================================

async def example_cache_control():
    """
    Control caching behavior for different scenarios.
    """
    from knomly.runtime import PipelineResolver, MemoryDefinitionLoader
    from knomly.runtime.resolver import InMemoryPipelineCache

    loader = MemoryDefinitionLoader()
    cache = InMemoryPipelineCache()

    resolver = PipelineResolver(
        loader=loader,
        cache=cache,
    )

    # Normal resolution (uses cache)
    # packet = await resolver.resolve_for_user("user-1", use_cache=True)

    # Force reload from loader (bypasses cache)
    # packet = await resolver.resolve_for_user("user-1", use_cache=False)

    # Clear cache entirely
    cache.clear()

    print("Cache cleared")


# =============================================================================
# Example JSON Config Files
# =============================================================================

EXAMPLE_PIPELINE_JSON = """
{
    "session": {
        "session_id": "placeholder",
        "user_id": "placeholder",
        "locale": "en-US"
    },
    "agent": {
        "system_prompt": "You are a helpful project manager assistant...",
        "welcome_message": "Hello! I can help you manage tasks.",
        "voice_id": "alloy"
    },
    "providers": {
        "stt": {
            "provider_type": "stt",
            "provider_code": "deepgram",
            "params": {"model": "nova-2", "language": "en"},
            "auth_secret_key": "deepgram_api_key"
        },
        "llm": {
            "provider_type": "llm",
            "provider_code": "openai",
            "params": {"model": "gpt-4o", "temperature": 0.7},
            "auth_secret_key": "openai_api_key"
        }
    },
    "tools": [
        {
            "name": "create_issue",
            "description": "Create a new issue in the project",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "priority": {"type": "string", "enum": ["low", "medium", "high"]}
                },
                "required": ["title"]
            },
            "source": "openapi",
            "source_config": {
                "spec_url": "https://api.plane.so/openapi.json",
                "operation_id": "createIssue",
                "auth_type": "bearer",
                "auth_secret_key": "plane_api_key"
            }
        }
    ]
}
"""

EXAMPLE_TOOLS_JSON = """
[
    {
        "name": "list_issues",
        "description": "List all issues in the project",
        "parameters": {"type": "object", "properties": {}},
        "source": "openapi",
        "source_config": {
            "spec_url": "https://api.plane.so/openapi.json",
            "operation_id": "listIssues",
            "auth_secret_key": "plane_api_key"
        },
        "read_only": true,
        "tags": ["plane", "issues"]
    },
    {
        "name": "update_issue",
        "description": "Update an existing issue",
        "parameters": {
            "type": "object",
            "properties": {
                "issue_id": {"type": "string"},
                "title": {"type": "string"},
                "status": {"type": "string"}
            },
            "required": ["issue_id"]
        },
        "source": "openapi",
        "source_config": {
            "spec_url": "https://api.plane.so/openapi.json",
            "operation_id": "updateIssue",
            "auth_secret_key": "plane_api_key"
        },
        "destructive": true,
        "requires_confirmation": true
    }
]
"""


# =============================================================================
# Main: Run Examples
# =============================================================================

async def main():
    """Run examples."""
    print("=" * 60)
    print("Example 1: Basic File Loader")
    print("=" * 60)
    # await example_basic_file_loader()  # Needs config/ directory

    print("\n" + "=" * 60)
    print("Example 2: Memory Loader for Testing")
    print("=" * 60)
    await example_memory_loader_for_testing()

    print("\n" + "=" * 60)
    print("Example 4: Tools Only")
    print("=" * 60)
    await example_tools_only()

    print("\n" + "=" * 60)
    print("Example 6: Cache Control")
    print("=" * 60)
    await example_cache_control()


if __name__ == "__main__":
    asyncio.run(main())
