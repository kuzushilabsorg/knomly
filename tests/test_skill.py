"""
Tests for Skill Abstraction (Phase 2.5).

Tests cover:
- Skill creation and configuration
- Tool bundling and access
- OpenAPI-based skill generation
- Skill registry management
- Lifecycle (initialize/cleanup)
"""

import pytest
from unittest.mock import AsyncMock, MagicMock

from knomly.tools.skill import (
    Skill,
    SkillProtocol,
    SkillRegistry,
    create_skill_from_openapi,
)
from knomly.tools.base import Tool, ToolResult


# =============================================================================
# Test Fixtures
# =============================================================================


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", description: str = "A mock tool"):
        self._name = name
        self._description = description
        self.initialize_called = False
        self.cleanup_called = False

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.success("Mock executed")

    async def initialize(self) -> None:
        self.initialize_called = True

    async def cleanup(self) -> None:
        self.cleanup_called = True


@pytest.fixture
def mock_tools():
    """Create a set of mock tools."""
    return [
        MockTool("create_task", "Create a new task"),
        MockTool("query_tasks", "Query existing tasks"),
        MockTool("update_task", "Update a task"),
    ]


@pytest.fixture
def simple_openapi_spec():
    """Simple OpenAPI spec for testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Test API",
            "description": "A test API for skills",
            "version": "1.0.0",
        },
        "servers": [{"url": "https://api.test.com"}],
        "paths": {
            "/items": {
                "get": {
                    "operationId": "listItems",
                    "summary": "List all items",
                    "responses": {"200": {"description": "Success"}},
                },
                "post": {
                    "operationId": "createItem",
                    "summary": "Create an item",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {"type": "string"},
                                    },
                                    "required": ["name"],
                                }
                            }
                        },
                    },
                    "responses": {"201": {"description": "Created"}},
                },
            },
            "/items/{id}": {
                "get": {
                    "operationId": "getItem",
                    "summary": "Get item by ID",
                    "parameters": [
                        {
                            "name": "id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {"200": {"description": "Success"}},
                },
            },
        },
    }


# =============================================================================
# Skill Protocol Tests
# =============================================================================


class TestSkillProtocol:
    """Tests for SkillProtocol compliance."""

    def test_skill_implements_protocol(self, mock_tools):
        """Skill should implement SkillProtocol."""
        skill = Skill(
            name="test",
            description="Test skill",
            tools=mock_tools,
        )

        # Should have all required properties
        assert hasattr(skill, "name")
        assert hasattr(skill, "description")
        assert hasattr(skill, "get_tools")
        assert hasattr(skill, "initialize")
        assert hasattr(skill, "cleanup")

    def test_protocol_methods_are_accessible(self, mock_tools):
        """Protocol methods should be callable."""
        skill = Skill(
            name="test",
            description="Test skill",
            tools=mock_tools,
        )

        assert skill.name == "test"
        assert skill.description == "Test skill"
        assert len(skill.get_tools()) == 3


# =============================================================================
# Skill Creation Tests
# =============================================================================


class TestSkillCreation:
    """Tests for Skill creation and configuration."""

    def test_create_empty_skill(self):
        """Create skill with no tools."""
        skill = Skill(name="empty", description="Empty skill")

        assert skill.name == "empty"
        assert skill.description == "Empty skill"
        assert len(skill.tools) == 0
        assert skill.get_tools() == ()

    def test_create_skill_with_tools(self, mock_tools):
        """Create skill with tools."""
        skill = Skill(
            name="plane",
            description="Plane tools",
            tools=mock_tools,
        )

        assert skill.name == "plane"
        assert len(skill.get_tools()) == 3

    def test_create_skill_with_auth(self, mock_tools):
        """Create skill with authentication."""
        skill = Skill(
            name="plane",
            description="Plane tools",
            tools=mock_tools,
            auth={"Authorization": "Bearer xyz"},
        )

        assert skill.auth == {"Authorization": "Bearer xyz"}

    def test_create_skill_with_metadata(self, mock_tools):
        """Create skill with metadata."""
        skill = Skill(
            name="plane",
            description="Plane tools",
            tools=mock_tools,
            metadata={"version": "1.0", "api_version": "v1"},
        )

        assert skill.metadata["version"] == "1.0"
        assert skill.metadata["api_version"] == "v1"

    def test_from_tools_factory(self, mock_tools):
        """Test from_tools factory method."""
        skill = Skill.from_tools(
            name="plane",
            tools=mock_tools,
            description="Plane project management",
            auth={"X-API-Key": "test-key"},
        )

        assert skill.name == "plane"
        assert skill.description == "Plane project management"
        assert len(skill.get_tools()) == 3
        assert skill.auth == {"X-API-Key": "test-key"}


# =============================================================================
# Tool Access Tests
# =============================================================================


class TestToolAccess:
    """Tests for tool access methods."""

    def test_get_tools_returns_tuple(self, mock_tools):
        """get_tools should return immutable sequence."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        tools = skill.get_tools()

        assert isinstance(tools, tuple)
        assert len(tools) == 3

    def test_get_tool_by_name(self, mock_tools):
        """get_tool should find tool by name."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        tool = skill.get_tool("create_task")

        assert tool is not None
        assert tool.name == "create_task"

    def test_get_tool_not_found(self, mock_tools):
        """get_tool should return None for unknown tool."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        tool = skill.get_tool("nonexistent")

        assert tool is None

    def test_add_tool(self):
        """add_tool should add tool and return self."""
        skill = Skill(name="test", description="Test")
        tool = MockTool("new_tool")

        result = skill.add_tool(tool)

        assert result is skill  # Returns self for chaining
        assert len(skill.get_tools()) == 1
        assert skill.get_tool("new_tool") is tool

    def test_add_tool_chaining(self):
        """add_tool should support chaining."""
        skill = (
            Skill(name="test", description="Test")
            .add_tool(MockTool("tool1"))
            .add_tool(MockTool("tool2"))
            .add_tool(MockTool("tool3"))
        )

        assert len(skill.get_tools()) == 3


# =============================================================================
# OpenAPI Factory Tests
# =============================================================================


class TestOpenAPISkillCreation:
    """Tests for creating skills from OpenAPI specs."""

    def test_from_openapi_creates_skill(self, simple_openapi_spec):
        """from_openapi should create skill with tools."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
        )

        assert skill.name == "test_api"
        assert len(skill.get_tools()) == 3  # listItems, createItem, getItem

    def test_from_openapi_extracts_description(self, simple_openapi_spec):
        """from_openapi should extract description from spec."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
        )

        assert "Test API" in skill.description
        assert "A test API for skills" in skill.description

    def test_from_openapi_custom_description(self, simple_openapi_spec):
        """from_openapi should allow custom description."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
            description="Custom description",
        )

        assert skill.description == "Custom description"

    def test_from_openapi_with_auth(self, simple_openapi_spec):
        """from_openapi should store auth headers."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
            auth={"Authorization": "Bearer token"},
        )

        assert skill.auth == {"Authorization": "Bearer token"}

    def test_from_openapi_filter_operations(self, simple_openapi_spec):
        """from_openapi should filter operations."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
            operations=["listItems", "createItem"],
        )

        assert len(skill.get_tools()) == 2
        assert skill.get_tool("listItems") is not None
        assert skill.get_tool("createItem") is not None
        assert skill.get_tool("getItem") is None

    def test_from_openapi_with_metadata(self, simple_openapi_spec):
        """from_openapi should store metadata."""
        skill = Skill.from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
            metadata={"environment": "production"},
        )

        assert skill.metadata["environment"] == "production"

    def test_convenience_function(self, simple_openapi_spec):
        """create_skill_from_openapi should work."""
        skill = create_skill_from_openapi(
            name="test_api",
            spec=simple_openapi_spec,
            auth={"X-API-Key": "test"},
        )

        assert skill.name == "test_api"
        assert len(skill.get_tools()) == 3


# =============================================================================
# Lifecycle Tests
# =============================================================================


class TestSkillLifecycle:
    """Tests for skill lifecycle management."""

    @pytest.mark.asyncio
    async def test_initialize_calls_tool_initialize(self, mock_tools):
        """initialize should call initialize on tools."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        await skill.initialize()

        for tool in mock_tools:
            assert tool.initialize_called

    @pytest.mark.asyncio
    async def test_initialize_only_once(self, mock_tools):
        """initialize should only run once."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        await skill.initialize()

        # Reset flags
        for tool in mock_tools:
            tool.initialize_called = False

        # Call again
        await skill.initialize()

        # Should not have called again
        for tool in mock_tools:
            assert not tool.initialize_called

    @pytest.mark.asyncio
    async def test_cleanup_calls_tool_cleanup(self, mock_tools):
        """cleanup should call cleanup on tools."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        await skill.initialize()
        await skill.cleanup()

        for tool in mock_tools:
            assert tool.cleanup_called

    @pytest.mark.asyncio
    async def test_cleanup_without_initialize(self, mock_tools):
        """cleanup without initialize should do nothing."""
        skill = Skill(name="test", description="Test", tools=mock_tools)

        await skill.cleanup()

        for tool in mock_tools:
            assert not tool.cleanup_called

    @pytest.mark.asyncio
    async def test_cleanup_handles_errors(self):
        """cleanup should handle errors gracefully."""
        # Create tool that raises on cleanup
        tool = MockTool("failing_tool")
        tool.cleanup = AsyncMock(side_effect=Exception("Cleanup failed"))

        skill = Skill(name="test", description="Test", tools=[tool])
        await skill.initialize()

        # Should not raise
        await skill.cleanup()


# =============================================================================
# Skill Registry Tests
# =============================================================================


class TestSkillRegistry:
    """Tests for SkillRegistry."""

    def test_create_empty_registry(self):
        """Create empty registry."""
        registry = SkillRegistry()

        assert len(registry) == 0

    def test_register_skill(self, mock_tools):
        """Register a skill."""
        registry = SkillRegistry()
        skill = Skill(name="plane", description="Plane", tools=mock_tools)

        result = registry.register(skill)

        assert result is registry  # Returns self for chaining
        assert len(registry) == 1
        assert "plane" in registry

    def test_register_duplicate_raises(self, mock_tools):
        """Registering duplicate name should raise."""
        registry = SkillRegistry()
        skill1 = Skill(name="plane", description="First", tools=mock_tools)
        skill2 = Skill(name="plane", description="Second", tools=mock_tools)

        registry.register(skill1)

        with pytest.raises(ValueError, match="already registered"):
            registry.register(skill2)

    def test_get_skill(self, mock_tools):
        """Get skill by name."""
        registry = SkillRegistry()
        skill = Skill(name="plane", description="Plane", tools=mock_tools)
        registry.register(skill)

        result = registry.get_skill("plane")

        assert result is skill

    def test_get_skill_not_found(self):
        """Get non-existent skill returns None."""
        registry = SkillRegistry()

        result = registry.get_skill("nonexistent")

        assert result is None

    def test_get_skills(self, mock_tools):
        """Get all skills."""
        registry = SkillRegistry()
        skill1 = Skill(name="plane", description="Plane", tools=mock_tools[:1])
        skill2 = Skill(name="zulip", description="Zulip", tools=mock_tools[1:2])

        registry.register(skill1)
        registry.register(skill2)

        skills = registry.get_skills()

        assert len(skills) == 2

    def test_get_all_tools(self, mock_tools):
        """Get all tools across skills."""
        registry = SkillRegistry()
        skill1 = Skill(name="plane", description="Plane", tools=mock_tools[:2])
        skill2 = Skill(name="zulip", description="Zulip", tools=mock_tools[2:])

        registry.register(skill1)
        registry.register(skill2)

        all_tools = registry.get_all_tools()

        assert len(all_tools) == 3

    def test_get_tool_by_name(self, mock_tools):
        """Find tool by name across skills."""
        registry = SkillRegistry()
        skill1 = Skill(name="plane", description="Plane", tools=mock_tools[:2])
        skill2 = Skill(name="zulip", description="Zulip", tools=mock_tools[2:])

        registry.register(skill1)
        registry.register(skill2)

        tool = registry.get_tool("create_task")

        assert tool is not None
        assert tool.name == "create_task"

    def test_get_tool_not_found(self, mock_tools):
        """Find non-existent tool returns None."""
        registry = SkillRegistry()
        skill = Skill(name="plane", description="Plane", tools=mock_tools)
        registry.register(skill)

        tool = registry.get_tool("nonexistent")

        assert tool is None

    @pytest.mark.asyncio
    async def test_initialize_all(self, mock_tools):
        """initialize_all should initialize all skills."""
        registry = SkillRegistry()

        tools1 = [MockTool("t1"), MockTool("t2")]
        tools2 = [MockTool("t3")]

        skill1 = Skill(name="s1", description="S1", tools=tools1)
        skill2 = Skill(name="s2", description="S2", tools=tools2)

        registry.register(skill1)
        registry.register(skill2)

        await registry.initialize_all()

        for tool in tools1 + tools2:
            assert tool.initialize_called

    @pytest.mark.asyncio
    async def test_cleanup_all(self, mock_tools):
        """cleanup_all should cleanup all skills."""
        registry = SkillRegistry()

        tools1 = [MockTool("t1"), MockTool("t2")]
        tools2 = [MockTool("t3")]

        skill1 = Skill(name="s1", description="S1", tools=tools1)
        skill2 = Skill(name="s2", description="S2", tools=tools2)

        registry.register(skill1)
        registry.register(skill2)

        await registry.initialize_all()
        await registry.cleanup_all()

        for tool in tools1 + tools2:
            assert tool.cleanup_called

    def test_registry_chaining(self, mock_tools):
        """Registry should support method chaining."""
        registry = (
            SkillRegistry()
            .register(Skill(name="s1", description="S1", tools=mock_tools[:1]))
            .register(Skill(name="s2", description="S2", tools=mock_tools[1:2]))
        )

        assert len(registry) == 2


# =============================================================================
# Integration Tests
# =============================================================================


class TestSkillIntegration:
    """Integration tests for skill usage patterns."""

    def test_skill_in_agent_workflow(self, mock_tools):
        """Test skill usage in typical agent workflow."""
        # Create skills
        plane_skill = Skill.from_tools(
            name="plane",
            tools=mock_tools[:2],
            description="Plane project management",
        )

        zulip_skill = Skill.from_tools(
            name="zulip",
            tools=mock_tools[2:],
            description="Zulip communication",
        )

        # Register in registry
        registry = SkillRegistry()
        registry.register(plane_skill)
        registry.register(zulip_skill)

        # Get all tools for agent
        all_tools = registry.get_all_tools()

        assert len(all_tools) == 3

        # Find specific tool
        create_task = registry.get_tool("create_task")
        assert create_task is not None

    @pytest.mark.asyncio
    async def test_skill_lifecycle_in_workflow(self):
        """Test complete lifecycle in workflow."""
        tools = [MockTool("t1"), MockTool("t2")]
        skill = Skill(name="test", description="Test", tools=tools)

        # Simulate workflow
        await skill.initialize()

        # Use tools...
        tool = skill.get_tool("t1")
        result = await tool.execute({})
        assert result.success

        # Cleanup
        await skill.cleanup()

        for tool in tools:
            assert tool.initialize_called
            assert tool.cleanup_called

    def test_skill_with_openapi_and_custom_tools(self, simple_openapi_spec):
        """Test combining OpenAPI tools with custom tools."""
        # Create OpenAPI skill
        api_skill = Skill.from_openapi(
            name="api",
            spec=simple_openapi_spec,
        )

        # Add a custom tool
        custom_tool = MockTool("custom_handler", "Custom handler")
        api_skill.add_tool(custom_tool)

        # Should have OpenAPI tools + custom tool
        assert len(api_skill.get_tools()) == 4  # 3 OpenAPI + 1 custom
        assert api_skill.get_tool("custom_handler") is not None
        assert api_skill.get_tool("listItems") is not None
