"""
Tests for Tool Factory Pattern (v2.1 Multi-Tenancy).

Tests cover:
- ToolContext creation and access
- StaticToolFactory
- CompositeToolFactory
- ConditionalToolFactory
- Context extraction from frames
- PlaneToolFactory (mocked)
"""

import pytest
from unittest.mock import MagicMock, patch

from knomly.tools.factory import (
    ToolContext,
    ToolFactory,
    StaticToolFactory,
    CompositeToolFactory,
    ConditionalToolFactory,
    extract_tool_context_from_frame,
)
from knomly.tools.base import Tool, ToolResult


# =============================================================================
# Mock Tool for Testing
# =============================================================================


class MockTool(Tool):
    """Mock tool for testing factories."""

    def __init__(self, name: str = "mock_tool"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A mock tool"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.success("Mock executed")


# =============================================================================
# ToolContext Tests
# =============================================================================


class TestToolContext:
    """Tests for ToolContext dataclass."""

    def test_create_basic_context(self):
        """Create context with minimal fields."""
        ctx = ToolContext(user_id="user-123")

        assert ctx.user_id == "user-123"
        assert ctx.secrets == {}
        assert ctx.metadata == {}

    def test_create_context_with_secrets(self):
        """Create context with secrets."""
        ctx = ToolContext(
            user_id="user-123",
            secrets={"api_key": "secret-key", "token": "bearer-token"},
        )

        assert ctx.secrets["api_key"] == "secret-key"
        assert ctx.secrets["token"] == "bearer-token"

    def test_context_is_immutable(self):
        """Context should be frozen (immutable)."""
        ctx = ToolContext(user_id="user-123")

        with pytest.raises(Exception):  # FrozenInstanceError
            ctx.user_id = "changed"

    def test_get_secret(self):
        """Get secret by key."""
        ctx = ToolContext(
            user_id="user-123",
            secrets={"api_key": "secret"},
        )

        assert ctx.get_secret("api_key") == "secret"
        assert ctx.get_secret("missing") is None
        assert ctx.get_secret("missing", "default") == "default"

    def test_require_secret_success(self):
        """Require secret when present."""
        ctx = ToolContext(
            user_id="user-123",
            secrets={"api_key": "secret"},
        )

        assert ctx.require_secret("api_key") == "secret"

    def test_require_secret_missing(self):
        """Require secret when missing raises."""
        ctx = ToolContext(user_id="user-123")

        with pytest.raises(KeyError, match="api_key"):
            ctx.require_secret("api_key")


# =============================================================================
# StaticToolFactory Tests
# =============================================================================


class TestStaticToolFactory:
    """Tests for StaticToolFactory."""

    def test_returns_static_tools(self):
        """Factory returns the static tools."""
        tools = [MockTool("tool1"), MockTool("tool2")]
        factory = StaticToolFactory(tools)

        ctx = ToolContext(user_id="user-123")
        result = factory.build_tools(ctx)

        assert len(result) == 2
        assert result[0].name == "tool1"
        assert result[1].name == "tool2"

    def test_ignores_context(self):
        """Factory ignores context and returns same tools."""
        tools = [MockTool("tool1")]
        factory = StaticToolFactory(tools)

        ctx1 = ToolContext(user_id="user-1")
        ctx2 = ToolContext(user_id="user-2", secrets={"key": "value"})

        result1 = factory.build_tools(ctx1)
        result2 = factory.build_tools(ctx2)

        assert result1 == result2

    def test_empty_tools(self):
        """Factory with no tools returns empty sequence."""
        factory = StaticToolFactory([])
        ctx = ToolContext(user_id="user-123")

        result = factory.build_tools(ctx)

        assert result == ()


# =============================================================================
# CompositeToolFactory Tests
# =============================================================================


class TestCompositeToolFactory:
    """Tests for CompositeToolFactory."""

    def test_combines_factories(self):
        """Composite combines tools from multiple factories."""
        factory1 = StaticToolFactory([MockTool("tool1")])
        factory2 = StaticToolFactory([MockTool("tool2"), MockTool("tool3")])

        composite = CompositeToolFactory([factory1, factory2])
        ctx = ToolContext(user_id="user-123")

        result = composite.build_tools(ctx)

        assert len(result) == 3
        names = [t.name for t in result]
        assert names == ["tool1", "tool2", "tool3"]

    def test_add_factory_chaining(self):
        """add_factory supports chaining."""
        composite = (
            CompositeToolFactory([])
            .add_factory(StaticToolFactory([MockTool("tool1")]))
            .add_factory(StaticToolFactory([MockTool("tool2")]))
        )

        ctx = ToolContext(user_id="user-123")
        result = composite.build_tools(ctx)

        assert len(result) == 2


# =============================================================================
# ConditionalToolFactory Tests
# =============================================================================


class TestConditionalToolFactory:
    """Tests for ConditionalToolFactory."""

    def test_builds_when_condition_true(self):
        """Factory builds tools when condition is True."""
        inner = StaticToolFactory([MockTool("admin_tool")])
        factory = ConditionalToolFactory(
            inner=inner,
            condition=lambda ctx: ctx.metadata.get("is_admin", False),
        )

        ctx = ToolContext(
            user_id="user-123",
            metadata={"is_admin": True},
        )
        result = factory.build_tools(ctx)

        assert len(result) == 1
        assert result[0].name == "admin_tool"

    def test_empty_when_condition_false(self):
        """Factory returns empty when condition is False."""
        inner = StaticToolFactory([MockTool("admin_tool")])
        factory = ConditionalToolFactory(
            inner=inner,
            condition=lambda ctx: ctx.metadata.get("is_admin", False),
        )

        ctx = ToolContext(
            user_id="user-123",
            metadata={"is_admin": False},
        )
        result = factory.build_tools(ctx)

        assert result == ()

    def test_condition_based_on_secrets(self):
        """Condition can check secrets."""
        inner = StaticToolFactory([MockTool("premium_tool")])
        factory = ConditionalToolFactory(
            inner=inner,
            condition=lambda ctx: "premium_key" in ctx.secrets,
        )

        # Without premium key
        ctx1 = ToolContext(user_id="user-1")
        assert factory.build_tools(ctx1) == ()

        # With premium key
        ctx2 = ToolContext(
            user_id="user-2",
            secrets={"premium_key": "xxx"},
        )
        assert len(factory.build_tools(ctx2)) == 1


# =============================================================================
# Context Extraction Tests
# =============================================================================


class TestExtractToolContextFromFrame:
    """Tests for extract_tool_context_from_frame helper."""

    def test_extract_from_frame_ignores_secrets_in_metadata(self):
        """
        SECURITY (v3.2): Secrets in frame.metadata are IGNORED.

        Frames may be serialized/logged, so secrets should never be in metadata.
        Use secret_provider callback instead.
        """
        frame = MagicMock()
        frame.metadata = {
            "user_id": "user-123",
            "secrets": {"api_key": "secret"},  # This should be IGNORED
            "extra_data": "value",
        }
        frame.sender_phone = "+1234567890"

        ctx = extract_tool_context_from_frame(frame)

        assert ctx.user_id == "user-123"
        assert ctx.secrets == {}  # Secrets NOT extracted from frame
        assert ctx.metadata == {"extra_data": "value"}

    def test_extract_with_secret_provider(self):
        """Secrets come from secret_provider callback, not frame."""
        frame = MagicMock()
        frame.metadata = {"user_id": "user-123"}
        frame.sender_phone = None

        def secret_provider(user_id: str) -> dict[str, str]:
            assert user_id == "user-123"
            return {"api_key": "secret-from-callback"}

        ctx = extract_tool_context_from_frame(frame, secret_provider=secret_provider)

        assert ctx.secrets == {"api_key": "secret-from-callback"}

    def test_extract_secret_provider_receives_correct_user_id(self):
        """Secret provider is called with the correct user_id."""
        frame = MagicMock()
        frame.metadata = {}
        frame.sender_phone = "+1234567890"

        calls = []

        def secret_provider(user_id: str) -> dict[str, str]:
            calls.append(user_id)
            return {}

        extract_tool_context_from_frame(frame, secret_provider=secret_provider)

        assert calls == ["+1234567890"]  # Falls back to sender_phone

    def test_extract_secret_provider_failure_returns_empty_secrets(self):
        """If secret_provider fails, return empty secrets (don't crash)."""
        frame = MagicMock()
        frame.metadata = {"user_id": "user-123"}
        frame.sender_phone = None

        def failing_provider(user_id: str) -> dict[str, str]:
            raise RuntimeError("Vault connection failed")

        ctx = extract_tool_context_from_frame(frame, secret_provider=failing_provider)

        assert ctx.secrets == {}  # Graceful degradation

    def test_extract_uses_sender_phone_as_fallback(self):
        """Uses sender_phone if user_id not in metadata."""
        frame = MagicMock()
        frame.metadata = {}
        frame.sender_phone = "+1234567890"

        ctx = extract_tool_context_from_frame(frame)

        assert ctx.user_id == "+1234567890"

    def test_extract_anonymous_fallback(self):
        """Uses 'anonymous' if no user identifier."""
        frame = MagicMock()
        frame.metadata = {}
        frame.sender_phone = None

        ctx = extract_tool_context_from_frame(frame)

        assert ctx.user_id == "anonymous"

    def test_extract_without_secrets(self):
        """Works when no secrets in metadata."""
        frame = MagicMock()
        frame.metadata = {"user_id": "user-123"}
        frame.sender_phone = None

        ctx = extract_tool_context_from_frame(frame)

        assert ctx.secrets == {}

    def test_extract_excludes_secrets_from_metadata(self):
        """Secrets and user_id are not duplicated in metadata."""
        frame = MagicMock()
        frame.metadata = {
            "user_id": "user-123",
            "secrets": {"api_key": "secret"},
            "intent": "task",
        }
        frame.sender_phone = None

        ctx = extract_tool_context_from_frame(frame)

        assert "user_id" not in ctx.metadata
        assert "secrets" not in ctx.metadata
        assert ctx.metadata["intent"] == "task"


# =============================================================================
# PlaneToolFactory Tests (Mocked)
# =============================================================================


class TestPlaneToolFactory:
    """Tests for PlaneToolFactory with mocked dependencies."""

    def test_returns_empty_without_api_key(self):
        """Factory returns empty when no API key."""
        from knomly.tools.plane import PlaneToolFactory

        factory = PlaneToolFactory(workspace_slug="test")
        ctx = ToolContext(user_id="user-123")

        result = factory.build_tools(ctx)

        assert result == ()

    def test_returns_empty_without_workspace(self):
        """Factory returns empty when no workspace."""
        from knomly.tools.plane import PlaneToolFactory

        factory = PlaneToolFactory()  # No workspace
        ctx = ToolContext(
            user_id="user-123",
            secrets={"plane_api_key": "key"},
        )

        result = factory.build_tools(ctx)

        assert result == ()

    @patch("knomly.integrations.plane.PlaneClient")
    @patch("knomly.integrations.plane.cache.PlaneEntityCache")
    def test_builds_tools_with_credentials(self, mock_cache, mock_client):
        """Factory builds tools when credentials present."""
        from knomly.tools.plane import PlaneToolFactory

        factory = PlaneToolFactory(
            base_url="https://api.plane.so/api/v1",
            workspace_slug="test-workspace",
        )
        ctx = ToolContext(
            user_id="user-123",
            secrets={"plane_api_key": "user-api-key"},
        )

        result = factory.build_tools(ctx)

        # Should return 2 tools (create + query)
        assert len(result) == 2

        # Verify client was created with user's key
        mock_client.assert_called_once()
        config_arg = mock_client.call_args[0][0]
        assert config_arg.api_key == "user-api-key"

    @patch("knomly.integrations.plane.PlaneClient")
    @patch("knomly.integrations.plane.cache.PlaneEntityCache")
    def test_respects_include_query_tool(self, mock_cache, mock_client):
        """Factory respects include_query_tool setting."""
        from knomly.tools.plane import PlaneToolFactory

        factory = PlaneToolFactory(
            workspace_slug="test",
            include_query_tool=False,
        )
        ctx = ToolContext(
            user_id="user-123",
            secrets={"plane_api_key": "key"},
        )

        result = factory.build_tools(ctx)

        assert len(result) == 1  # Only create tool


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestToolFactoryProtocol:
    """Tests verifying ToolFactory protocol compliance."""

    def test_static_factory_is_tool_factory(self):
        """StaticToolFactory satisfies ToolFactory protocol."""
        factory = StaticToolFactory([])

        assert isinstance(factory, ToolFactory)

    def test_composite_factory_is_tool_factory(self):
        """CompositeToolFactory satisfies ToolFactory protocol."""
        factory = CompositeToolFactory([])

        assert isinstance(factory, ToolFactory)

    def test_conditional_factory_is_tool_factory(self):
        """ConditionalToolFactory satisfies ToolFactory protocol."""
        factory = ConditionalToolFactory(
            inner=StaticToolFactory([]),
            condition=lambda ctx: True,
        )

        assert isinstance(factory, ToolFactory)


# =============================================================================
# Integration Tests
# =============================================================================


class TestFactoryIntegration:
    """Integration tests for factory pattern."""

    def test_multi_tenant_simulation(self):
        """Simulate multi-tenant tool building."""
        # Different users get different tools based on context
        admin_factory = ConditionalToolFactory(
            inner=StaticToolFactory([MockTool("admin_tool")]),
            condition=lambda ctx: ctx.metadata.get("role") == "admin",
        )

        premium_factory = ConditionalToolFactory(
            inner=StaticToolFactory([MockTool("premium_tool")]),
            condition=lambda ctx: "premium_key" in ctx.secrets,
        )

        base_factory = StaticToolFactory([MockTool("base_tool")])

        composite = CompositeToolFactory([
            base_factory,
            admin_factory,
            premium_factory,
        ])

        # Regular user: only base tools
        regular_ctx = ToolContext(user_id="regular")
        regular_tools = composite.build_tools(regular_ctx)
        assert len(regular_tools) == 1
        assert regular_tools[0].name == "base_tool"

        # Admin user: base + admin tools
        admin_ctx = ToolContext(
            user_id="admin",
            metadata={"role": "admin"},
        )
        admin_tools = composite.build_tools(admin_ctx)
        assert len(admin_tools) == 2

        # Premium user: base + premium tools
        premium_ctx = ToolContext(
            user_id="premium",
            secrets={"premium_key": "xxx"},
        )
        premium_tools = composite.build_tools(premium_ctx)
        assert len(premium_tools) == 2

        # Admin + Premium: all tools
        super_ctx = ToolContext(
            user_id="super",
            secrets={"premium_key": "xxx"},
            metadata={"role": "admin"},
        )
        super_tools = composite.build_tools(super_ctx)
        assert len(super_tools) == 3
