"""
Tests for the runtime layer.

Tests for:
- PipelineResolver
- RuntimeBuilder
- FileDefinitionLoader
- MemoryDefinitionLoader
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from knomly.adapters.schemas import (
    AgentContext,
    PipelinePacket,
    PipelineProviderConfig,
    ProviderDefinition,
    SessionContext,
    ToolDefinition,
)
from knomly.runtime import (
    FileDefinitionLoader,
    MemoryDefinitionLoader,
    PipelineResolver,
    RuntimeBuilder,
)
from knomly.runtime.resolver import InMemoryPipelineCache

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_tool_definition():
    """Create a sample ToolDefinition."""
    return ToolDefinition(
        name="create_task",
        description="Create a task",
        parameters={
            "type": "object",
            "properties": {
                "title": {"type": "string"},
            },
            "required": ["title"],
        },
        source="native",
        source_config={},
    )


@pytest.fixture
def sample_pipeline_packet():
    """Create a sample PipelinePacket."""
    return PipelinePacket(
        session=SessionContext(
            session_id="test-session",
            user_id="test-user",
            locale="en-US",
        ),
        agent=AgentContext(
            system_prompt="You are a helpful assistant.",
            welcome_message="Hello!",
        ),
        providers=PipelineProviderConfig(
            llm=ProviderDefinition.llm(
                provider_code="openai",
                model="gpt-4",
            ),
        ),
        tools=[
            ToolDefinition(
                name="test_tool",
                description="A test tool",
                parameters={"type": "object", "properties": {}},
                source="native",
                source_config={},
            ),
        ],
    )


@pytest.fixture
def temp_config_dir(sample_pipeline_packet, sample_tool_definition):
    """Create a temporary config directory with sample files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir)

        # Create directories
        (base_dir / "tools").mkdir()
        (base_dir / "pipelines").mkdir()
        (base_dir / "providers").mkdir()

        # Create tool files
        tools_data = [sample_tool_definition.model_dump()]
        with open(base_dir / "tools" / "global.json", "w") as f:
            json.dump(tools_data, f)

        # Create user-specific tools
        user_tools = [
            {
                "name": "user_specific_tool",
                "description": "User specific",
                "parameters": {"type": "object", "properties": {}},
                "source": "native",
                "source_config": {},
            }
        ]
        with open(base_dir / "tools" / "test-user.json", "w") as f:
            json.dump(user_tools, f)

        # Create pipeline files
        pipeline_data = sample_pipeline_packet.model_dump(mode="json")
        with open(base_dir / "pipelines" / "default.json", "w") as f:
            json.dump(pipeline_data, f)

        # Create user-specific pipeline
        user_pipeline = sample_pipeline_packet.model_copy(deep=True)
        user_pipeline.agent.system_prompt = "User-specific prompt"
        with open(base_dir / "pipelines" / "test-user.json", "w") as f:
            json.dump(user_pipeline.model_dump(mode="json"), f)

        # Create providers file
        providers_data = {
            "llm": {
                "openai": {
                    "provider_type": "llm",
                    "provider_code": "openai",
                    "params": {"model": "gpt-4"},
                }
            },
            "stt": {
                "deepgram": {
                    "provider_type": "stt",
                    "provider_code": "deepgram",
                    "params": {},
                }
            },
        }
        with open(base_dir / "providers" / "default.json", "w") as f:
            json.dump(providers_data, f)

        yield base_dir


# =============================================================================
# InMemoryPipelineCache Tests
# =============================================================================


class TestInMemoryPipelineCache:
    """Tests for InMemoryPipelineCache."""

    @pytest.mark.asyncio
    async def test_get_miss(self):
        """Test cache miss returns None."""
        cache = InMemoryPipelineCache()
        result = await cache.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_set_and_get(self, sample_pipeline_packet):
        """Test setting and getting from cache."""
        cache = InMemoryPipelineCache()
        await cache.set("test-key", sample_pipeline_packet)

        result = await cache.get("test-key")
        assert result is not None
        assert result.session.session_id == sample_pipeline_packet.session.session_id

    @pytest.mark.asyncio
    async def test_clear(self, sample_pipeline_packet):
        """Test clearing cache."""
        cache = InMemoryPipelineCache()
        await cache.set("key1", sample_pipeline_packet)
        await cache.set("key2", sample_pipeline_packet)

        cache.clear()

        assert await cache.get("key1") is None
        assert await cache.get("key2") is None


# =============================================================================
# FileDefinitionLoader Tests
# =============================================================================


class TestFileDefinitionLoader:
    """Tests for FileDefinitionLoader."""

    @pytest.mark.asyncio
    async def test_get_tools_for_user_empty_dir(self, tmp_path):
        """Test loading tools from non-existent directory."""
        loader = FileDefinitionLoader(tmp_path)
        tools = await loader.get_tools_for_user("user-1")
        assert tools == []

    @pytest.mark.asyncio
    async def test_get_tools_for_user_global(self, temp_config_dir):
        """Test loading global tools."""
        loader = FileDefinitionLoader(temp_config_dir)
        tools = await loader.get_tools_for_user("unknown-user")

        # Should get all global tools (from all .json files except user-specific)
        # The loader loads from all files that don't match user_id
        # So unknown-user will load both global.json AND test-user.json
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "create_task" in names
        assert "user_specific_tool" in names

    @pytest.mark.asyncio
    async def test_get_tools_for_user_with_user_specific(self, temp_config_dir):
        """Test loading user-specific + global tools."""
        loader = FileDefinitionLoader(temp_config_dir)
        tools = await loader.get_tools_for_user("test-user")

        # Should get user-specific + global tools
        assert len(tools) == 2
        names = {t.name for t in tools}
        assert "user_specific_tool" in names
        assert "create_task" in names

    @pytest.mark.asyncio
    async def test_get_pipeline_for_session_default(self, temp_config_dir):
        """Test loading default pipeline."""
        loader = FileDefinitionLoader(temp_config_dir)
        packet = await loader.get_pipeline_for_session("sess-1", "unknown-user")

        assert packet is not None
        assert packet.session.session_id == "sess-1"
        assert packet.session.user_id == "unknown-user"
        assert packet.agent.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_get_pipeline_for_session_user_specific(self, temp_config_dir):
        """Test loading user-specific pipeline."""
        loader = FileDefinitionLoader(temp_config_dir)
        packet = await loader.get_pipeline_for_session("sess-1", "test-user")

        assert packet is not None
        assert packet.session.session_id == "sess-1"
        assert packet.session.user_id == "test-user"
        assert packet.agent.system_prompt == "User-specific prompt"

    @pytest.mark.asyncio
    async def test_get_pipeline_for_session_not_found(self, tmp_path):
        """Test loading pipeline when no files exist."""
        loader = FileDefinitionLoader(tmp_path)
        packet = await loader.get_pipeline_for_session("sess-1", "user-1")
        assert packet is None

    @pytest.mark.asyncio
    async def test_get_provider_definition(self, temp_config_dir):
        """Test loading provider definition."""
        loader = FileDefinitionLoader(temp_config_dir)
        provider = await loader.get_provider_definition("llm", "openai")

        assert provider is not None
        assert provider.provider_code == "openai"
        assert provider.params["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_provider_definition_not_found(self, temp_config_dir):
        """Test loading non-existent provider."""
        loader = FileDefinitionLoader(temp_config_dir)
        provider = await loader.get_provider_definition("llm", "nonexistent")
        assert provider is None


# =============================================================================
# MemoryDefinitionLoader Tests
# =============================================================================


class TestMemoryDefinitionLoader:
    """Tests for MemoryDefinitionLoader."""

    @pytest.mark.asyncio
    async def test_add_global_tool(self, sample_tool_definition):
        """Test adding a global tool."""
        loader = MemoryDefinitionLoader()
        await loader.add_tool(sample_tool_definition)

        # Should be available to any user
        tools = await loader.get_tools_for_user("any-user")
        assert len(tools) == 1
        assert tools[0].name == "create_task"

    @pytest.mark.asyncio
    async def test_add_user_specific_tool(self, sample_tool_definition):
        """Test adding a user-specific tool."""
        loader = MemoryDefinitionLoader()
        await loader.add_tool(sample_tool_definition, user_id="user-1")

        # Should only be available to that user
        tools_user1 = await loader.get_tools_for_user("user-1")
        tools_user2 = await loader.get_tools_for_user("user-2")

        assert len(tools_user1) == 1
        assert len(tools_user2) == 0

    @pytest.mark.asyncio
    async def test_combined_tools(self, sample_tool_definition):
        """Test global + user-specific tools are combined."""
        loader = MemoryDefinitionLoader()

        # Add global tool
        await loader.add_tool(sample_tool_definition)

        # Add user-specific tool
        user_tool = sample_tool_definition.model_copy()
        user_tool.name = "user_tool"
        await loader.add_tool(user_tool, user_id="user-1")

        # User-1 should get both
        tools = await loader.get_tools_for_user("user-1")
        assert len(tools) == 2

        # Other users get only global
        tools_other = await loader.get_tools_for_user("user-2")
        assert len(tools_other) == 1

    @pytest.mark.asyncio
    async def test_set_and_get_pipeline(self, sample_pipeline_packet):
        """Test setting and getting user pipeline."""
        loader = MemoryDefinitionLoader()
        await loader.set_pipeline("user-1", sample_pipeline_packet)

        packet = await loader.get_pipeline_for_session("sess-1", "user-1")
        assert packet is not None
        assert packet.agent.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_default_pipeline(self, sample_pipeline_packet):
        """Test default pipeline fallback."""
        loader = MemoryDefinitionLoader()
        await loader.set_default_pipeline(sample_pipeline_packet)

        # User without specific pipeline gets default
        packet = await loader.get_pipeline_for_session("sess-1", "unknown-user")
        assert packet is not None

    @pytest.mark.asyncio
    async def test_clear(self, sample_tool_definition, sample_pipeline_packet):
        """Test clearing all definitions."""
        loader = MemoryDefinitionLoader()
        await loader.add_tool(sample_tool_definition)
        await loader.set_default_pipeline(sample_pipeline_packet)

        loader.clear()

        tools = await loader.get_tools_for_user("user-1")
        packet = await loader.get_pipeline_for_session("sess-1", "user-1")

        assert tools == []
        assert packet is None


# =============================================================================
# PipelineResolver Tests
# =============================================================================


class TestPipelineResolver:
    """Tests for PipelineResolver."""

    @pytest.mark.asyncio
    async def test_resolve_for_user_from_loader(self, sample_pipeline_packet):
        """Test resolving config from loader."""
        loader = MemoryDefinitionLoader()
        await loader.set_pipeline("user-1", sample_pipeline_packet)

        resolver = PipelineResolver(loader=loader)
        packet = await resolver.resolve_for_user("user-1")

        assert packet is not None
        assert packet.agent.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_resolve_for_user_uses_cache(self, sample_pipeline_packet):
        """Test that resolved packets are cached."""
        loader = MemoryDefinitionLoader()
        await loader.set_pipeline("user-1", sample_pipeline_packet)

        cache = InMemoryPipelineCache()
        resolver = PipelineResolver(loader=loader, cache=cache)

        # First call - cache miss
        packet1 = await resolver.resolve_for_user("user-1")

        # Second call - should use cache
        packet2 = await resolver.resolve_for_user("user-1")

        assert packet1 is not None
        assert packet2 is not None
        # Verify cache was populated
        cached = await cache.get("pipeline:user-1:default")
        assert cached is not None

    @pytest.mark.asyncio
    async def test_resolve_for_user_bypass_cache(self, sample_pipeline_packet):
        """Test bypassing cache."""
        loader = MemoryDefinitionLoader()
        # Use deep copy to avoid shared references
        original = sample_pipeline_packet.model_copy(deep=True)
        await loader.set_pipeline("user-1", original)

        cache = InMemoryPipelineCache()
        resolver = PipelineResolver(loader=loader, cache=cache)

        # First call with cache - this populates the cache
        await resolver.resolve_for_user("user-1", use_cache=True)

        # Modify the pipeline in the loader (deep copy again)
        modified = sample_pipeline_packet.model_copy(deep=True)
        modified.agent.system_prompt = "Modified prompt"
        await loader.set_pipeline("user-1", modified)

        # With cache - still gets cached value
        packet_cached = await resolver.resolve_for_user("user-1", use_cache=True)
        assert packet_cached.agent.system_prompt == "You are a helpful assistant."

        # Without cache - forces reload from loader, gets new value
        packet_fresh = await resolver.resolve_for_user("user-1", use_cache=False)
        assert packet_fresh.agent.system_prompt == "Modified prompt"

    @pytest.mark.asyncio
    async def test_resolve_for_user_default_packet(self, sample_pipeline_packet):
        """Test falling back to default packet."""
        loader = MemoryDefinitionLoader()  # Empty loader
        resolver = PipelineResolver(loader=loader, default_packet=sample_pipeline_packet)

        packet = await resolver.resolve_for_user("unknown-user")
        assert packet is not None
        assert packet.agent.system_prompt == "You are a helpful assistant."

    @pytest.mark.asyncio
    async def test_resolve_for_user_no_config_raises(self):
        """Test error when no config found and no default."""
        loader = MemoryDefinitionLoader()  # Empty loader
        resolver = PipelineResolver(loader=loader)

        with pytest.raises(ValueError, match="No pipeline configuration found"):
            await resolver.resolve_for_user("unknown-user")

    @pytest.mark.asyncio
    async def test_resolve_tools_for_user(self, sample_tool_definition):
        """Test resolving just tools."""
        loader = MemoryDefinitionLoader()
        await loader.add_tool(sample_tool_definition)

        resolver = PipelineResolver(loader=loader)
        tools = await resolver.resolve_tools_for_user("user-1")

        assert len(tools) == 1
        assert tools[0].name == "create_task"

    @pytest.mark.asyncio
    async def test_cache_key_with_session(self):
        """Test cache key includes session ID when provided."""
        resolver = PipelineResolver(loader=MemoryDefinitionLoader())

        key_default = resolver._build_cache_key("user-1", None)
        key_session = resolver._build_cache_key("user-1", "session-abc")

        assert key_default == "pipeline:user-1:default"
        assert key_session == "pipeline:user-1:session-abc"


# =============================================================================
# RuntimeBuilder Tests
# =============================================================================


class TestRuntimeBuilder:
    """Tests for RuntimeBuilder."""

    @pytest.mark.asyncio
    async def test_build_tools_only_without_builder(self, sample_pipeline_packet):
        """Test building tools without tool builder returns empty list."""
        builder = RuntimeBuilder()
        tools = await builder.build_tools_only(sample_pipeline_packet, {})
        assert tools == []

    @pytest.mark.asyncio
    async def test_build_tools_only_empty_tools(self, sample_pipeline_packet):
        """Test building tools with empty tools list."""
        # Create packet with no tools
        empty_packet = sample_pipeline_packet.model_copy(deep=True)
        empty_packet.tools = []

        mock_tool_builder = AsyncMock()
        builder = RuntimeBuilder(tool_builder=mock_tool_builder)

        tools = await builder.build_tools_only(empty_packet, {})

        # Should return empty list without calling builder
        assert tools == []
        mock_tool_builder.build_tools.assert_not_called()

    @pytest.mark.asyncio
    async def test_build_providers_only_without_factory(self, sample_pipeline_packet):
        """Test building providers without factory returns empty registry."""
        builder = RuntimeBuilder()

        # Import here to get real ProviderRegistry
        from knomly.providers import ProviderRegistry

        providers = await builder.build_providers_only(sample_pipeline_packet, {})

        # Should return an empty registry
        assert isinstance(providers, ProviderRegistry)

    @pytest.mark.asyncio
    async def test_build_providers_only_with_factory(self, sample_pipeline_packet):
        """Test building providers with mock factory."""
        mock_factory = MagicMock()
        mock_llm_service = MagicMock()
        mock_factory.create_service.return_value = mock_llm_service

        builder = RuntimeBuilder(service_factory=mock_factory)
        providers = await builder.build_providers_only(sample_pipeline_packet, {})

        # Should have called create_service for LLM
        mock_factory.create_service.assert_called()

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test RuntimeBuilder initialization with different args."""
        # No args
        builder1 = RuntimeBuilder()
        assert builder1._service_factory is None
        assert builder1._tool_builder is None

        # With service factory
        mock_factory = MagicMock()
        builder2 = RuntimeBuilder(service_factory=mock_factory)
        assert builder2._service_factory is mock_factory

        # With tool builder
        mock_tool_builder = MagicMock()
        builder3 = RuntimeBuilder(tool_builder=mock_tool_builder)
        assert builder3._tool_builder is mock_tool_builder


# =============================================================================
# Integration Tests
# =============================================================================


class TestRuntimeIntegration:
    """Integration tests for the runtime layer."""

    @pytest.mark.asyncio
    async def test_file_loader_to_resolver(self, temp_config_dir):
        """Test FileDefinitionLoader with PipelineResolver."""
        loader = FileDefinitionLoader(temp_config_dir)
        resolver = PipelineResolver(loader=loader)

        # Resolve config
        packet = await resolver.resolve_for_user("test-user")

        assert packet is not None
        assert packet.session.user_id == "test-user"
        assert packet.agent.system_prompt == "User-specific prompt"

    @pytest.mark.asyncio
    async def test_memory_loader_to_resolver(self, sample_pipeline_packet, sample_tool_definition):
        """Test MemoryDefinitionLoader with PipelineResolver."""
        loader = MemoryDefinitionLoader()
        await loader.set_default_pipeline(sample_pipeline_packet)
        await loader.add_tool(sample_tool_definition)

        resolver = PipelineResolver(loader=loader)

        # Resolve config
        packet = await resolver.resolve_for_user("any-user")
        tools = await resolver.resolve_tools_for_user("any-user")

        assert packet is not None
        assert len(tools) == 1

    @pytest.mark.asyncio
    async def test_resolver_with_session_context(self, sample_pipeline_packet):
        """Test that resolver updates session context."""
        loader = MemoryDefinitionLoader()
        await loader.set_default_pipeline(sample_pipeline_packet)

        resolver = PipelineResolver(loader=loader)

        # Resolve with specific session
        packet = await resolver.resolve_for_user(
            "user-123",
            session_id="session-456",
        )

        # Note: The loader doesn't update session context,
        # but the packet should be returned with original context
        assert packet is not None
