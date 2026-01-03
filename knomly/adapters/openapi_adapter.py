"""
OpenAPI Tool Adapter.

Converts ToolDefinition(source="openapi") to live OpenAPIOperationTool instances.

Design Principle:
    "One Definition → One Tool"

    Each ToolDefinition represents a SINGLE API operation. The adapter
    handles the complexity of:
    1. Fetching/caching the OpenAPI spec
    2. Extracting the specific operation
    3. Building the tool with user credentials

This solves the "OpenAPI Granularity Problem" identified in architectural review:
    - OpenAPIToolkit returns MULTIPLE tools from a spec
    - ToolDefinition represents ONE tool
    - This adapter bridges the gap

Performance Optimization (v3.3 - Review Feedback):
    Two caching layers:
    1. SpecCache - Caches raw OpenAPI spec (already existed)
    2. OperationCache - Caches PARSED OpenAPIOperation objects (new)

    Why: Parsing large specs (Salesforce, Jira) is CPU-intensive.
    Parsing once, reusing parsed operations saves O(N) work per request.

    Connection pooling:
    - Shared httpx.AsyncClient for spec fetching
    - Shared client passed to tools for API calls
    - Eliminates 200ms+ SSL handshake overhead per request

Architecture:
    ┌──────────────────────────────────────────────────────────────┐
    │                    OpenAPIToolAdapter                         │
    │                                                               │
    │   Input: ToolDefinition + ToolContext                        │
    │                                                               │
    │   source_config:                                              │
    │     spec_url: "https://api.plane.so/openapi.json"           │
    │     operation_id: "createWorkItem"                           │
    │     base_url: "https://api.plane.so"  (optional override)   │
    │     auth_type: "bearer" | "x_api_key" | "basic"             │
    │     auth_secret_key: "plane_api_key"                         │
    │                                                               │
    │   1. Fetch spec (cached via SpecCache)                       │
    │   2. Get parsed operation (cached via OperationCache)        │
    │   3. Get auth from context.secrets                           │
    │   4. Build OpenAPIOperationTool (O(1) - just wraps operation)│
    │                                                               │
    │   Output: OpenAPIOperationTool                               │
    └──────────────────────────────────────────────────────────────┘

Usage:
    adapter = OpenAPIToolAdapter()

    # From database
    definition = ToolDefinition(
        name="create_task",
        description="Create a task",
        source="openapi",
        source_config={
            "spec_url": "https://api.plane.so/openapi.json",
            "operation_id": "createWorkItem",
            "auth_type": "bearer",
            "auth_secret_key": "plane_api_key",
        },
    )

    # Build with user context
    context = ToolContext(
        user_id="user-123",
        secrets={"plane_api_key": "user-specific-key"},
    )

    tool = await adapter.build_tool(definition, context)
    # tool is now a live OpenAPIOperationTool ready for agent use
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import httpx

from .base import BaseToolAdapter

if TYPE_CHECKING:
    from knomly.tools.base import Tool
    from knomly.tools.factory import ToolContext

    from .schemas import ToolDefinition

logger = logging.getLogger(__name__)


class SpecCache:
    """
    Simple in-memory cache for OpenAPI specs.

    Specs are cached by URL with a configurable TTL.
    This prevents fetching the same spec multiple times
    when building tools for different operations.

    Thread-safe via asyncio locks.
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            ttl_seconds: Cache entry TTL (default 1 hour)
        """
        self._cache: dict[str, tuple[dict[str, Any], datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    async def get(self, url: str) -> dict[str, Any] | None:
        """Get spec from cache if not expired."""
        async with self._lock:
            if url not in self._cache:
                return None

            spec, cached_at = self._cache[url]
            if datetime.now(UTC) - cached_at > self._ttl:
                del self._cache[url]
                return None

            return spec

    async def set(self, url: str, spec: dict[str, Any]) -> None:
        """Store spec in cache."""
        async with self._lock:
            self._cache[url] = (spec, datetime.now(UTC))

    async def clear(self) -> None:
        """Clear all cached specs."""
        async with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get number of cached specs."""
        return len(self._cache)


class OperationCache:
    """
    Cache for PARSED OpenAPI operations (v3.3 Performance Optimization).

    Why this matters:
        Parsing a large OpenAPI spec (Salesforce, Jira, etc.) involves:
        - Iterating all paths
        - Resolving $ref references
        - Building OpenAPIOperation objects

        For a 1MB spec with 200 operations, this takes ~100-500ms.
        Caching parsed operations reduces this to ~1ms lookup.

    Key Structure:
        (spec_url_or_hash, operation_id) -> (OpenAPIOperation, base_url, cached_at)

    Thread-safe via asyncio locks.
    """

    def __init__(self, ttl_seconds: int = 3600):
        """
        Initialize cache.

        Args:
            ttl_seconds: Cache entry TTL (default 1 hour)
        """
        self._cache: dict[tuple[str, str], tuple[Any, str, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
        self._lock = asyncio.Lock()

    async def get(self, spec_key: str, operation_id: str) -> tuple[Any, str] | None:
        """
        Get cached operation and base_url.

        Args:
            spec_key: Spec URL or hash
            operation_id: Operation ID

        Returns:
            Tuple of (OpenAPIOperation, base_url) or None
        """
        async with self._lock:
            cache_key = (spec_key, operation_id)
            if cache_key not in self._cache:
                return None

            operation, base_url, cached_at = self._cache[cache_key]
            if datetime.now(UTC) - cached_at > self._ttl:
                del self._cache[cache_key]
                return None

            return (operation, base_url)

    async def set(self, spec_key: str, operation_id: str, operation: Any, base_url: str) -> None:
        """
        Store parsed operation in cache.

        Args:
            spec_key: Spec URL or hash
            operation_id: Operation ID
            operation: Parsed OpenAPIOperation
            base_url: Base URL for the API
        """
        async with self._lock:
            cache_key = (spec_key, operation_id)
            self._cache[cache_key] = (operation, base_url, datetime.now(UTC))

    async def get_all_for_spec(self, spec_key: str) -> dict[str, tuple[Any, str]]:
        """Get all cached operations for a spec."""
        async with self._lock:
            now = datetime.now(UTC)
            result = {}
            for (sk, op_id), (op, base_url, cached_at) in list(self._cache.items()):
                if sk == spec_key and now - cached_at <= self._ttl:
                    result[op_id] = (op, base_url)
            return result

    async def clear(self) -> None:
        """Clear all cached operations."""
        async with self._lock:
            self._cache.clear()

    def size(self) -> int:
        """Get number of cached operations."""
        return len(self._cache)


# Global caches (shared across adapter instances)
_spec_cache = SpecCache()
_operation_cache = OperationCache()

# Global shared HTTP client for connection pooling
_http_client: httpx.AsyncClient | None = None
_http_client_lock = asyncio.Lock()


async def get_shared_http_client(timeout: float = 30.0) -> httpx.AsyncClient:
    """
    Get or create the shared HTTP client.

    This client is shared across all adapter instances for connection pooling.
    Eliminates 200ms+ SSL handshake overhead per request.

    Args:
        timeout: Request timeout (used only on first creation)

    Returns:
        Shared httpx.AsyncClient
    """
    global _http_client
    async with _http_client_lock:
        if _http_client is None or _http_client.is_closed:
            _http_client = httpx.AsyncClient(timeout=timeout)
            logger.info("[openapi_adapter] Created shared HTTP client for connection pooling")
        return _http_client


async def close_shared_http_client() -> None:
    """Close the shared HTTP client (call on shutdown)."""
    global _http_client
    async with _http_client_lock:
        if _http_client is not None:
            await _http_client.aclose()
            _http_client = None
            logger.info("[openapi_adapter] Closed shared HTTP client")


class OpenAPIToolAdapter(BaseToolAdapter):
    """
    Adapter for building tools from OpenAPI specifications.

    This adapter handles:
    1. Fetching OpenAPI specs (with caching)
    2. Extracting specific operations by ID (with caching)
    3. Building tools with user-scoped credentials

    Performance (v3.3):
        - SpecCache: Caches raw specs (prevents redundant fetches)
        - OperationCache: Caches parsed operations (prevents redundant parsing)
        - Shared HTTP client: Connection pooling (eliminates SSL handshake overhead)

    The adapter is stateless and can be shared across requests.
    User-specific data comes from ToolContext at build time.

    source_config schema:
        {
            "spec_url": str,           # URL to OpenAPI spec (required if no spec)
            "spec": dict,              # Inline spec (alternative to spec_url)
            "operation_id": str,       # Operation to extract (required)
            "base_url": str,           # Override spec's server URL (optional)
            "auth_type": str,          # "bearer", "x_api_key", "basic" (default: bearer)
            "auth_secret_key": str,    # Key in secrets for auth (optional)
            "timeout": float,          # Request timeout in seconds (default: 30)
            "default_headers": dict,   # Headers to include in all requests (optional)
        }

    Example:
        adapter = OpenAPIToolAdapter()

        definition = ToolDefinition(
            name="create_task",
            source="openapi",
            source_config={
                "spec_url": "https://api.plane.so/openapi.json",
                "operation_id": "createWorkItem",
                "auth_type": "bearer",
                "auth_secret_key": "plane_api_key",
            },
        )

        tool = await adapter.build_tool(definition, context)
    """

    source_type = "openapi"

    def __init__(
        self,
        *,
        spec_cache: SpecCache | None = None,
        operation_cache: OperationCache | None = None,
        http_timeout: float = 30.0,
    ):
        """
        Initialize adapter.

        Args:
            spec_cache: Optional custom spec cache (uses global if None)
            operation_cache: Optional custom operation cache (uses global if None)
            http_timeout: Default HTTP timeout for spec fetching
        """
        self._spec_cache = spec_cache or _spec_cache
        self._operation_cache = operation_cache or _operation_cache
        self._http_timeout = http_timeout

    async def build_tool(
        self,
        definition: ToolDefinition,
        context: ToolContext,
    ) -> Tool:
        """
        Build a live Tool from an OpenAPI-based definition.

        Performance (v3.3):
            1. Check OperationCache for pre-parsed operation (O(1) lookup)
            2. If miss, fetch spec (SpecCache), parse operation, cache it
            3. Build tool with user's auth (O(1) wrapping)
            4. Pass shared HTTP client for connection pooling

        Args:
            definition: Tool definition with source="openapi"
            context: User context with credentials

        Returns:
            OpenAPIOperationTool ready for use

        Raises:
            ValueError: If definition is invalid
            RuntimeError: If tool cannot be built
        """
        # Import here to avoid circular imports
        from knomly.tools.generic.openapi import (
            OpenAPIOperationTool,
            OpenAPIToolkit,
        )

        config = definition.source_config
        operation_id = config.get("operation_id") or definition.name
        spec_key = config.get("spec_url") or "inline"

        # 1. Check operation cache first (fast path)
        cached = await self._operation_cache.get(spec_key, operation_id)
        if cached is not None:
            operation, base_url = cached
            logger.debug(f"[openapi_adapter] Operation cache hit: {operation_id}")

            # Build auth from context secrets
            auth = self._build_auth(config, context)

            # Get shared HTTP client for connection pooling
            http_client = await get_shared_http_client(self._http_timeout)

            # Create tool directly from cached operation (O(1))
            tool = OpenAPIOperationTool(
                operation=operation,
                base_url=base_url,
                auth=auth,
                default_headers=config.get("default_headers"),
                timeout=config.get("timeout", self._http_timeout),
                http_client=http_client,  # Connection pooling
            )

            logger.info(
                f"[openapi_adapter] Built tool from cache: {tool.name} "
                f"({(tool.annotations.read_only_hint and 'read-only') or 'read-write'})"
            )
            return tool

        # 2. Cache miss - need to parse spec
        logger.info(
            f"[openapi_adapter] Building tool: {definition.name} (operation_id={operation_id})"
        )

        # Get the OpenAPI spec (spec cache)
        spec = await self._get_spec(config)

        # Determine base URL
        base_url = config.get("base_url")
        if not base_url:
            servers = spec.get("servers", [])
            if servers:
                base_url = servers[0].get("url", "")

        if not base_url:
            raise ValueError(
                f"No base_url in source_config and none found in spec for tool '{definition.name}'"
            )

        # Build auth from context secrets
        auth = self._build_auth(config, context)

        # Get shared HTTP client for connection pooling
        http_client = await get_shared_http_client(self._http_timeout)

        # Parse spec and extract operation
        toolkit = OpenAPIToolkit.from_spec(
            spec,
            base_url=base_url,
            auth=auth,
            operations=[operation_id],  # Only parse this operation
            timeout=config.get("timeout", self._http_timeout),
            default_headers=config.get("default_headers"),
            http_client=http_client,  # Connection pooling
        )

        # Get the specific tool
        try:
            tool = toolkit.get_tool(operation_id)
        except KeyError:
            available = toolkit.list_operations()[:5]
            raise ValueError(
                f"Operation '{operation_id}' not found in spec. Available: {available}..."
            )

        # Cache the parsed operation for future requests
        # Access internal operation object for caching
        operation = toolkit._operations.get(operation_id)
        if operation:
            await self._operation_cache.set(spec_key, operation_id, operation, base_url)
            logger.debug(f"[openapi_adapter] Cached operation: {operation_id}")

        logger.info(
            f"[openapi_adapter] Built tool: {tool.name} "
            f"({(tool.annotations.read_only_hint and 'read-only') or 'read-write'})"
        )

        return tool

    async def _get_spec(self, config: dict[str, Any]) -> dict[str, Any]:
        """
        Get OpenAPI spec from inline config or URL.

        Performance (v3.3):
            Uses shared HTTP client for connection pooling.

        Args:
            config: source_config from ToolDefinition

        Returns:
            Parsed OpenAPI spec dict
        """
        # Check for inline spec first
        if "spec" in config:
            return config["spec"]

        spec_url = config.get("spec_url")
        if not spec_url:
            raise ValueError("Either 'spec' or 'spec_url' required in source_config")

        # Check cache
        cached = await self._spec_cache.get(spec_url)
        if cached is not None:
            logger.debug(f"[openapi_adapter] Spec cache hit: {spec_url}")
            return cached

        # Fetch spec using shared HTTP client (connection pooling)
        logger.info(f"[openapi_adapter] Fetching spec: {spec_url}")

        client = await get_shared_http_client(self._http_timeout)
        response = await client.get(spec_url)
        response.raise_for_status()

        # Parse JSON or YAML
        try:
            spec = response.json()
        except Exception:
            try:
                import yaml

                spec = yaml.safe_load(response.text)
            except ImportError:
                raise ValueError(
                    "YAML spec detected but PyYAML not installed. Install with: pip install pyyaml"
                )

        # Cache spec
        await self._spec_cache.set(spec_url, spec)

        return spec

    def _build_auth(
        self,
        config: dict[str, Any],
        context: ToolContext,
    ) -> dict[str, str] | None:
        """
        Build auth dict from config and context secrets.

        Args:
            config: source_config from ToolDefinition
            context: User context with secrets

        Returns:
            Auth dict for OpenAPIOperationTool or None
        """
        auth_secret_key = config.get("auth_secret_key")
        if not auth_secret_key:
            return None

        auth_value = context.secrets.get(auth_secret_key)
        if not auth_value:
            logger.warning(f"[openapi_adapter] Secret '{auth_secret_key}' not found in context")
            return None

        auth_type = config.get("auth_type", "bearer")

        if auth_type == "bearer":
            return {"api_key": auth_value}
        elif auth_type == "x_api_key":
            return {"x_api_key": auth_value}
        elif auth_type == "basic":
            # Expect "username:password" format or separate keys
            if ":" in auth_value:
                user, password = auth_value.split(":", 1)
                return {"basic_user": user, "basic_pass": password}
            else:
                # Look for separate username key
                user_key = config.get("auth_username_key")
                if user_key and user_key in context.secrets:
                    return {
                        "basic_user": context.secrets[user_key],
                        "basic_pass": auth_value,
                    }
                return {"basic_user": "", "basic_pass": auth_value}
        else:
            logger.warning(f"[openapi_adapter] Unknown auth_type: {auth_type}")
            return {"api_key": auth_value}


class OpenAPISpecImporter:
    """
    Utility for importing an OpenAPI spec into ToolDefinitions.

    Use this to bulk-import all operations from an OpenAPI spec
    into your database as ToolDefinition documents.

    Usage:
        importer = OpenAPISpecImporter()

        # Import all operations
        definitions = await importer.import_spec(
            spec_url="https://api.plane.so/openapi.json",
            auth_secret_key="plane_api_key",
        )

        # Save to database
        for defn in definitions:
            await db.tools.insert_one(defn.model_dump())
    """

    async def import_spec(
        self,
        spec_url: str | None = None,
        spec: dict[str, Any] | None = None,
        *,
        base_url: str | None = None,
        auth_type: str = "bearer",
        auth_secret_key: str | None = None,
        tags_filter: list[str] | None = None,
        operations_filter: list[str] | None = None,
    ) -> list[ToolDefinition]:
        """
        Import operations from an OpenAPI spec as ToolDefinitions.

        Args:
            spec_url: URL to fetch spec from
            spec: Inline spec dict (alternative to spec_url)
            base_url: Override spec's server URL
            auth_type: Authentication type for all tools
            auth_secret_key: Secret key for authentication
            tags_filter: Only import operations with these tags
            operations_filter: Only import these operation IDs

        Returns:
            List of ToolDefinition objects
        """
        from knomly.tools.generic.openapi import OpenAPIToolkit

        from .schemas import ToolDefinition

        # Fetch spec if URL provided
        if spec is None:
            if spec_url is None:
                raise ValueError("Either spec or spec_url required")

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(spec_url)
                response.raise_for_status()
                spec = response.json()

        # Parse with toolkit to get operations
        toolkit = OpenAPIToolkit.from_spec(
            spec,
            base_url=base_url or "",
            tags=tags_filter,
            operations=operations_filter,
        )

        definitions = []
        for op_id in toolkit.list_operations():
            op = toolkit._operations[op_id]

            defn = ToolDefinition.from_openapi_operation(
                operation_id=op.operation_id,
                method=op.method,
                path=op.path,
                summary=op.summary or op.description,
                parameters=list(op.parameters),
                request_body_schema=op.request_body_schema,
                tags=list(op.tags),
                spec_url=spec_url,
                base_url=base_url,
                auth_type=auth_type,
                auth_secret_key=auth_secret_key,
            )

            definitions.append(defn)

        logger.info(
            f"[openapi_importer] Imported {len(definitions)} operations "
            f"from {spec_url or 'inline spec'}"
        )

        return definitions


# =============================================================================
# Convenience Functions
# =============================================================================


def get_openapi_adapter() -> OpenAPIToolAdapter:
    """Get a shared OpenAPIToolAdapter instance."""
    return OpenAPIToolAdapter()


async def import_openapi_tools(
    spec_url: str,
    *,
    auth_secret_key: str | None = None,
    tags: list[str] | None = None,
) -> list[ToolDefinition]:
    """
    Convenience function to import tools from an OpenAPI spec.

    Args:
        spec_url: URL to the OpenAPI spec
        auth_secret_key: Secret key for authentication
        tags: Optional tag filter

    Returns:
        List of ToolDefinition objects
    """
    importer = OpenAPISpecImporter()
    return await importer.import_spec(
        spec_url=spec_url,
        auth_secret_key=auth_secret_key,
        tags_filter=tags,
    )
