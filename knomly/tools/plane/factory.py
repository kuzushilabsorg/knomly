"""
Plane Tool Factory (v2.1 Multi-Tenancy).

Creates Plane tools per-request with user-specific credentials.

This factory solves the "Static Tool Trap":
- Previously: Tools created at startup with single API key
- Now: Tools built per-request from user's credentials in ToolContext

Usage:
    # Create factory (at startup - stateless)
    factory = PlaneToolFactory(
        base_url="https://api.plane.so/api/v1",
        workspace_slug="your-workspace",
    )

    # Build tools per-request (at runtime)
    context = ToolContext(
        user_id="user-123",
        secrets={"plane_api_key": "user-specific-key"},
    )
    tools = factory.build_tools(context)

    # Tools are now scoped to this user
    for tool in tools:
        result = await tool.execute({"name": "My Task", "project": "Mobile"})

Secret Keys:
    The factory looks for these keys in ToolContext.secrets:
    - plane_api_key: User's Plane API key (required)
    - plane_workspace: Override workspace slug (optional)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knomly.tools.base import Tool
    from knomly.tools.factory import ToolContext

logger = logging.getLogger(__name__)


# Secret key names
PLANE_API_KEY = "plane_api_key"
PLANE_WORKSPACE = "plane_workspace"


@dataclass
class PlaneToolFactory:
    """
    Factory for building Plane tools with user-specific credentials.

    Creates PlaneCreateTaskTool and PlaneQueryTasksTool for each request,
    using the API key from the user's ToolContext.

    Attributes:
        base_url: Plane API base URL
        workspace_slug: Default workspace (can be overridden per-user)
        include_query_tool: Whether to include the query tool
        cache_ttl: Cache TTL for entity resolution

    Example:
        factory = PlaneToolFactory(
            base_url="https://api.plane.so/api/v1",
            workspace_slug="my-workspace",
        )

        # User's request comes in with their API key
        context = ToolContext(
            user_id="user-123",
            secrets={"plane_api_key": "pk_xxx"},
        )

        tools = factory.build_tools(context)
        # Returns [PlaneCreateTaskTool, PlaneQueryTasksTool] configured
        # with user-123's API key
    """

    base_url: str = "https://api.plane.so/api/v1"
    workspace_slug: str = ""
    include_query_tool: bool = True
    cache_ttl: int = 300  # 5 minutes

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """
        Build Plane tools for the given context.

        Args:
            context: User context with API key in secrets

        Returns:
            Sequence of Plane tools configured for this user

        Raises:
            KeyError: If plane_api_key not in context.secrets
        """
        from knomly.integrations.base import IntegrationConfig
        from knomly.integrations.plane import PlaneClient
        from knomly.integrations.plane.cache import PlaneEntityCache
        from knomly.tools.plane import PlaneCreateTaskTool, PlaneQueryTasksTool

        # Get API key from context
        api_key = context.secrets.get(PLANE_API_KEY)
        if not api_key:
            logger.warning(
                f"[plane_factory] No API key for user {context.user_id}, " f"returning empty tools"
            )
            return ()

        # Get workspace (from context or default)
        workspace = context.secrets.get(PLANE_WORKSPACE) or self.workspace_slug
        if not workspace:
            logger.warning(f"[plane_factory] No workspace for user {context.user_id}")
            return ()

        # Build client config
        config = IntegrationConfig(
            api_key=api_key,
            base_url=f"{self.base_url}/workspaces/{workspace}",
        )

        # Create client and cache
        # Note: In production, you might want to pool/cache these
        client = PlaneClient(config)
        cache = PlaneEntityCache(client=client, ttl_seconds=self.cache_ttl)

        # Build tools
        tools: list[Tool] = [
            PlaneCreateTaskTool(client=client, cache=cache),
        ]

        if self.include_query_tool:
            tools.append(PlaneQueryTasksTool(client=client, cache=cache))

        logger.debug(f"[plane_factory] Built {len(tools)} tools for user {context.user_id}")

        return tuple(tools)


class CachedPlaneToolFactory:
    """
    Plane tool factory with client caching.

    For high-volume scenarios, this factory caches PlaneClient instances
    per user_id to avoid creating new connections for every request.

    The cache uses LRU eviction and has a configurable max size.

    Example:
        factory = CachedPlaneToolFactory(
            base_url="https://api.plane.so/api/v1",
            workspace_slug="my-workspace",
            max_clients=100,
        )

        # First request for user-123: creates new client
        tools1 = factory.build_tools(context)

        # Second request for same user: reuses client
        tools2 = factory.build_tools(context)
    """

    def __init__(
        self,
        base_url: str = "https://api.plane.so/api/v1",
        workspace_slug: str = "",
        include_query_tool: bool = True,
        cache_ttl: int = 300,
        max_clients: int = 100,
    ):
        self.base_url = base_url
        self.workspace_slug = workspace_slug
        self.include_query_tool = include_query_tool
        self.cache_ttl = cache_ttl
        self.max_clients = max_clients

        # LRU cache of clients by user_id
        self._clients: dict[str, tuple[Any, Any]] = {}  # user_id -> (client, cache)
        self._access_order: list[str] = []

    def _get_or_create_client(
        self,
        user_id: str,
        api_key: str,
        workspace: str,
    ) -> tuple[Any, Any]:
        """Get cached client or create new one."""
        from knomly.integrations.base import IntegrationConfig
        from knomly.integrations.plane import PlaneClient
        from knomly.integrations.plane.cache import PlaneEntityCache

        # Check cache
        if user_id in self._clients:
            # Move to end (most recently used)
            self._access_order.remove(user_id)
            self._access_order.append(user_id)
            return self._clients[user_id]

        # Create new client
        config = IntegrationConfig(
            api_key=api_key,
            base_url=f"{self.base_url}/workspaces/{workspace}",
        )
        client = PlaneClient(config)
        cache = PlaneEntityCache(client=client, ttl_seconds=self.cache_ttl)

        # Evict if over limit
        while len(self._clients) >= self.max_clients:
            oldest = self._access_order.pop(0)
            del self._clients[oldest]
            logger.debug(f"[plane_factory] Evicted client for {oldest}")

        # Store in cache
        self._clients[user_id] = (client, cache)
        self._access_order.append(user_id)

        return client, cache

    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        """Build Plane tools using cached client."""
        from knomly.tools.plane import PlaneCreateTaskTool, PlaneQueryTasksTool

        # Get credentials
        api_key = context.secrets.get(PLANE_API_KEY)
        if not api_key:
            return ()

        workspace = context.secrets.get(PLANE_WORKSPACE) or self.workspace_slug
        if not workspace:
            return ()

        # Get or create client
        client, cache = self._get_or_create_client(
            context.user_id,
            api_key,
            workspace,
        )

        # Build tools
        tools: list[Tool] = [
            PlaneCreateTaskTool(client=client, cache=cache),
        ]

        if self.include_query_tool:
            tools.append(PlaneQueryTasksTool(client=client, cache=cache))

        return tuple(tools)

    def clear_cache(self) -> None:
        """Clear all cached clients."""
        self._clients.clear()
        self._access_order.clear()

    def remove_user(self, user_id: str) -> None:
        """Remove a specific user's cached client."""
        if user_id in self._clients:
            del self._clients[user_id]
            self._access_order.remove(user_id)
