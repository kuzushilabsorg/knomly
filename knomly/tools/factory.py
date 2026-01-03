"""
Tool Factory Pattern (v2.1 Multi-Tenancy Fix).

Solves the "Static Tool Trap" identified in architectural review:
Tools were instantiated at startup with static credentials, making
the system single-tenant.

The Fix:
    Tools are now built per-request via a Factory that receives
    user-specific context (credentials, metadata).

Design Principle:
    "Context determines Execution."

    Execution is determined by the Frame's context, not by
    deployment-time environment variables.

Architecture:
    ┌──────────────────────────────────────────────────────────┐
    │                    Request Flow                          │
    │                                                          │
    │   Frame (with user context)                              │
    │          ↓                                               │
    │   AgentBridgeProcessor                                   │
    │          ↓                                               │
    │   ToolFactory.build_tools(context)  ← Dynamic!           │
    │          ↓                                               │
    │   [PlaneCreateTaskTool(user_api_key)]  ← Per-User        │
    │          ↓                                               │
    │   AgentExecutor.run(tools=...)                           │
    └──────────────────────────────────────────────────────────┘

Usage:
    # Define factory (at startup - stateless)
    factory = PlaneToolFactory()

    # Build tools per-request (at runtime - with user context)
    context = ToolContext(
        user_id="user-123",
        secrets={"plane_api_key": "user-specific-key"},
    )
    tools = factory.build_tools(context)

    # Tools are now scoped to this specific user
    result = await executor.run(goal="Create task", tools=tools)
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import Tool

logger = logging.getLogger(__name__)


# =============================================================================
# Tool Context
# =============================================================================


@dataclass(frozen=True)
class ToolContext:
    """
    Context for tool instantiation.

    Carries user-specific information needed to build tools:
    - Credentials (API keys, tokens)
    - User identity
    - Request metadata

    This is passed to ToolFactory.build_tools() at runtime,
    allowing tools to be scoped to the specific user/request.

    Attributes:
        user_id: Unique identifier for the user/tenant
        secrets: Credential map (e.g., {"plane_api_key": "..."})
        metadata: Additional context (workspace, project, etc.)

    Note:
        This is frozen (immutable) to prevent accidental mutation
        of credentials during tool execution.
    """

    user_id: str
    secrets: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_secret(self, key: str, default: str | None = None) -> str | None:
        """Get a secret by key."""
        return self.secrets.get(key, default)

    def require_secret(self, key: str) -> str:
        """
        Get a required secret, raising if not present.

        Args:
            key: Secret key to retrieve

        Returns:
            The secret value

        Raises:
            KeyError: If secret is not present
        """
        if key not in self.secrets:
            raise KeyError(f"Required secret '{key}' not found in ToolContext")
        return self.secrets[key]


# =============================================================================
# Tool Factory Protocol
# =============================================================================


@runtime_checkable
class ToolFactory(Protocol):
    """
    Protocol for tool factories.

    A ToolFactory builds tools at runtime with user-specific context.
    This enables multi-tenancy: each user's request gets tools
    configured with their credentials.

    Implementations:
        - PlaneToolFactory: Builds Plane API tools
        - CompositeToolFactory: Combines multiple factories
        - OpenAPIToolFactory: Builds tools from OpenAPI spec

    Example:
        class MyToolFactory:
            def build_tools(self, context: ToolContext) -> Sequence[Tool]:
                api_key = context.require_secret("api_key")
                return [MyTool(api_key=api_key)]
    """

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """
        Build tools for the given context.

        Args:
            context: User/request-specific context with credentials

        Returns:
            Sequence of Tool instances configured for this user
        """
        ...


# =============================================================================
# Factory Implementations
# =============================================================================


class StaticToolFactory:
    """
    Factory that returns pre-built tools (for testing/single-tenant).

    This is the simplest factory - it ignores context and returns
    the same tools every time. Useful for:
    - Testing
    - Single-tenant deployments
    - Migration from old static tool pattern

    Warning:
        This defeats multi-tenancy. Use only when you're certain
        all requests should use the same credentials.

    Example:
        tools = [MockTool(), AnotherMockTool()]
        factory = StaticToolFactory(tools)

        # Same tools returned regardless of context
        result = factory.build_tools(ToolContext(user_id="anyone"))
    """

    def __init__(self, tools: Sequence[Tool]) -> None:
        """
        Initialize with static tools.

        Args:
            tools: Pre-built tools to return for all requests
        """
        self._tools = tuple(tools)

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """Return the static tools (ignores context)."""
        logger.debug(
            f"[tool_factory:static] Returning {len(self._tools)} static tools "
            f"(context.user_id={context.user_id})"
        )
        return self._tools


class CompositeToolFactory:
    """
    Factory that combines multiple factories.

    Aggregates tools from multiple factories into a single set.
    Useful for combining domain-specific factories:

    Example:
        composite = CompositeToolFactory([
            PlaneToolFactory(),
            ZulipToolFactory(),
            SlackToolFactory(),
        ])

        # Gets tools from all factories
        tools = composite.build_tools(context)
    """

    def __init__(self, factories: Sequence[ToolFactory]) -> None:
        """
        Initialize with multiple factories.

        Args:
            factories: Factories to combine
        """
        self._factories = list(factories)

    def add_factory(self, factory: ToolFactory) -> CompositeToolFactory:
        """Add a factory. Returns self for chaining."""
        self._factories.append(factory)
        return self

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """Build tools from all factories."""
        tools: list[Tool] = []
        for factory in self._factories:
            factory_tools = factory.build_tools(context)
            tools.extend(factory_tools)

        logger.debug(
            f"[tool_factory:composite] Built {len(tools)} tools from "
            f"{len(self._factories)} factories"
        )
        return tuple(tools)


class ConditionalToolFactory:
    """
    Factory that conditionally builds tools based on context.

    Only builds tools if a condition is met. Useful for:
    - Feature flags
    - Capability-based tool loading
    - Tenant-specific tool sets

    Example:
        # Only build admin tools if user has admin role
        factory = ConditionalToolFactory(
            inner=AdminToolFactory(),
            condition=lambda ctx: ctx.metadata.get("role") == "admin",
        )
    """

    def __init__(
        self,
        inner: ToolFactory,
        condition: Callable[[ToolContext], bool],
    ) -> None:
        """
        Initialize with inner factory and condition.

        Args:
            inner: Factory to delegate to if condition passes
            condition: Function that returns True if tools should be built
        """
        self._inner = inner
        self._condition = condition

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """Build tools only if condition is met."""
        if self._condition(context):
            return self._inner.build_tools(context)
        return ()


# =============================================================================
# Context Extraction Helpers
# =============================================================================


def extract_tool_context_from_frame(
    frame: Any,
    secret_provider: Callable[[str], dict[str, str]] | None = None,
) -> ToolContext:
    """
    Extract ToolContext from a Frame's metadata.

    This is the bridge between v1 (Frames) and v2.1 (ToolFactory).
    The Frame carries user context from earlier pipeline stages;
    we extract it here to build user-scoped tools.

    SECURITY (v3.2):
        Secrets are NEVER extracted from Frame metadata.
        Frames may be serialized/logged, so secrets would leak.

        Instead, use the `secret_provider` callback to fetch secrets
        at runtime based on user_id. This ensures secrets are:
        - Never serialized with frames
        - Never logged by observability
        - Always fetched fresh from secure storage

    Args:
        frame: A Frame with user context in metadata
        secret_provider: Callback (user_id) -> dict[str, str] for secrets.
                        If None, returns empty secrets (tools won't have auth).

    Returns:
        ToolContext for tool factory

    Example:
        # With secret provider (production)
        context = extract_tool_context_from_frame(
            frame,
            secret_provider=get_user_secrets,  # from dependencies
        )

        # Without secrets (testing or pre-authenticated tools)
        context = extract_tool_context_from_frame(frame)
    """
    metadata = getattr(frame, "metadata", {}) or {}

    user_id = metadata.get("user_id") or getattr(frame, "sender_phone", None) or "anonymous"

    # SECURITY: Never extract secrets from frame metadata
    # Use the secret_provider callback instead
    secrets: dict[str, str] = {}
    if secret_provider is not None:
        try:
            secrets = secret_provider(user_id)
        except Exception as e:
            logger.warning(f"[tool_factory] Secret provider failed: {e}")

    # Log warning if someone tried to put secrets in frame (legacy/mistake)
    if "secrets" in metadata:
        logger.warning(
            "[tool_factory] SECURITY: Ignoring 'secrets' in frame.metadata. "
            "Secrets should be provided via secret_provider callback, not frames."
        )

    return ToolContext(
        user_id=user_id,
        secrets=secrets,
        metadata={
            k: v
            for k, v in metadata.items()
            if k not in ("user_id", "secrets")  # Explicitly exclude secrets
        },
    )


async def extract_tool_context_with_vault(
    frame: Any,
    vault_client: Any,
) -> ToolContext:
    """
    Extract ToolContext with secrets from a vault.

    This is the production pattern: Frame carries user_id,
    secrets are fetched from a secure vault at runtime.

    Args:
        frame: A Frame with user_id in metadata
        vault_client: Client for secrets vault (HashiCorp, AWS SM, etc.)

    Returns:
        ToolContext with secrets from vault

    Example:
        async def process(self, frame, ctx):
            tool_context = await extract_tool_context_with_vault(
                frame,
                ctx.vault_client,
            )
            tools = self.factory.build_tools(tool_context)
    """
    metadata = getattr(frame, "metadata", {}) or {}

    user_id = metadata.get("user_id") or getattr(frame, "sender_phone", None) or "anonymous"

    # Fetch secrets from vault
    secrets = await vault_client.get_secrets_for_user(user_id)

    return ToolContext(
        user_id=user_id,
        secrets=secrets,
        metadata={k: v for k, v in metadata.items() if k not in ("user_id", "secrets")},
    )
