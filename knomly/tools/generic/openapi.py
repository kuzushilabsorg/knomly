"""
OpenAPI Tool - Universal API Adapter.

This module provides tools that can interact with any API given an OpenAPI spec.
Instead of writing individual tools for each API, you write ONE universal adapter.

Design Principle (Phase 2.4):
    "Stop writing Tools manually."

The OpenAPIToolkit reads an OpenAPI 3.x specification and generates Tool instances
for each operation. These tools can be registered with an AgentExecutor just like
hand-written tools.

Benefits:
    - No custom code for each API integration
    - Automatic schema generation for LLM tool use
    - Automatic input validation
    - Consistent error handling
    - Easy to add new APIs (just point to their OpenAPI spec)

Usage:
    # From a spec dict
    toolkit = OpenAPIToolkit.from_spec(
        spec_dict,
        base_url="https://api.plane.so",
        auth={"api_key": "..."},
    )

    # From a URL
    toolkit = await OpenAPIToolkit.from_spec_url(
        "https://api.example.com/openapi.json",
        auth={"api_key": "..."},
    )

    # Get all tools
    tools = toolkit.get_tools()

    # Get specific operations
    tools = toolkit.get_tools(tags=["projects"])

    # Get single tool
    create_task = toolkit.get_tool("createWorkItem")

Architecture:
    OpenAPIToolkit (parses spec, manages tools)
        └── OpenAPIOperationTool (wraps single operation)
                └── OpenAPIOperation (parsed operation data)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlencode

import httpx

from knomly.tools.base import Tool, ToolAnnotations, ToolResult

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================


@dataclass(frozen=True, slots=True)
class OpenAPIOperation:
    """
    Parsed OpenAPI operation.

    Contains all metadata needed to:
    1. Generate a Tool interface for the LLM
    2. Construct and execute HTTP requests

    Attributes:
        operation_id: Unique identifier (used as tool name)
        method: HTTP method (get, post, put, patch, delete)
        path: URL path template (e.g., /projects/{project_id}/tasks)
        summary: Brief description for LLM
        description: Detailed description
        parameters: Path, query, and header parameters
        request_body_schema: JSON Schema for request body
        response_schema: Expected response schema
        tags: Operation tags for filtering
        security: Security requirements
    """

    operation_id: str
    method: str
    path: str
    summary: str = ""
    description: str = ""
    parameters: tuple[dict[str, Any], ...] = ()
    request_body_schema: dict[str, Any] | None = None
    request_body_required: bool = False
    response_schema: dict[str, Any] | None = None
    tags: tuple[str, ...] = ()
    security: tuple[dict[str, Any], ...] = ()

    @property
    def full_description(self) -> str:
        """Get full description for LLM."""
        parts = []
        if self.tags:
            parts.append(f"[{', '.join(self.tags)}]")
        if self.summary:
            parts.append(self.summary)
        elif self.description:
            parts.append(self.description[:200])
        else:
            parts.append(f"{self.method.upper()} {self.path}")
        return " ".join(parts)


# =============================================================================
# OpenAPI Operation Tool
# =============================================================================


class OpenAPIOperationTool(Tool):
    """
    A Tool generated from a single OpenAPI operation.

    This tool:
    1. Uses the operation's schema for input validation
    2. Constructs the HTTP request from LLM arguments
    3. Executes the request and returns structured results

    The tool is fully compatible with AgentExecutor and can be
    registered alongside hand-written tools.

    Lifecycle (v3.2):
        HTTP clients are managed automatically. By default, a fresh client
        is created for each execute() call and closed afterward. For better
        performance, pass a shared client via the constructor.

    Example:
        op = OpenAPIOperation(
            operation_id="listProjects",
            method="get",
            path="/api/v1/workspaces/{workspace}/projects/",
            summary="List all projects in a workspace",
            parameters=(
                {"name": "workspace", "in": "path", "required": True, ...},
            ),
        )

        tool = OpenAPIOperationTool(
            operation=op,
            base_url="https://api.plane.so",
            auth={"api_key": "..."},
        )

        result = await tool.execute({"workspace": "my-workspace"})

        # Or with shared client for connection pooling:
        async with httpx.AsyncClient() as client:
            tool = OpenAPIOperationTool(
                operation=op,
                base_url="https://api.plane.so",
                auth={"api_key": "..."},
                http_client=client,  # Caller manages lifecycle
            )
            result = await tool.execute(...)
    """

    def __init__(
        self,
        operation: OpenAPIOperation,
        base_url: str,
        *,
        auth: dict[str, str] | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ):
        """
        Initialize the tool.

        Args:
            operation: Parsed OpenAPI operation
            base_url: API base URL
            auth: Authentication config:
                - api_key: Bearer token
                - x_api_key: X-API-Key header
                - basic_user/basic_pass: Basic auth
            default_headers: Headers to include in all requests
            timeout: Request timeout in seconds
            http_client: Optional shared HTTP client (caller manages lifecycle).
                        If not provided, a fresh client is created per-execute
                        and closed automatically.
        """
        self._operation = operation
        self._base_url = base_url.rstrip("/")
        self._auth = auth or {}
        self._default_headers = default_headers or {}
        self._timeout = timeout
        self._shared_client = http_client  # Caller-managed (don't close)
        self._owned_client: httpx.AsyncClient | None = None  # Self-managed (do close)

    @property
    def name(self) -> str:
        """Tool name (operation ID)."""
        return self._operation.operation_id

    @property
    def description(self) -> str:
        """Tool description for LLM."""
        return self._operation.full_description

    @property
    def input_schema(self) -> dict[str, Any]:
        """
        Generate JSON Schema from OpenAPI parameters and requestBody.

        Combines:
        - Path parameters (required)
        - Query parameters (may be optional)
        - Header parameters (may be optional)
        - Request body properties (flattened to top level)

        Returns:
            JSON Schema compatible with LLM tool use
        """
        properties: dict[str, Any] = {}
        required: list[str] = []

        # Add parameters (path, query, header)
        for param in self._operation.parameters:
            param_name = param.get("name", "")
            if not param_name:
                continue

            param_schema = param.get("schema", {"type": "string"})
            param_desc = param.get("description", "")
            param_in = param.get("in", "query")

            # Build property schema
            prop_schema = {**param_schema}
            if param_desc:
                prop_schema["description"] = param_desc

            # Add context about where parameter goes
            if param_in == "path":
                prop_schema["description"] = (
                    f"(path) {prop_schema.get('description', '')}"
                ).strip()
            elif param_in == "header":
                prop_schema["description"] = (
                    f"(header) {prop_schema.get('description', '')}"
                ).strip()

            properties[param_name] = prop_schema

            if param.get("required", False):
                required.append(param_name)

        # Add request body properties (flattened)
        if self._operation.request_body_schema:
            body_schema = self._operation.request_body_schema

            if body_schema.get("type") == "object":
                body_props = body_schema.get("properties", {})
                body_required = body_schema.get("required", [])

                for prop_name, prop_schema in body_props.items():
                    # Don't overwrite parameter with same name
                    if prop_name not in properties:
                        properties[prop_name] = prop_schema
                        if prop_name in body_required:
                            required.append(prop_name)

        return {
            "type": "object",
            "properties": properties,
            "required": required,
        }

    @property
    def annotations(self) -> ToolAnnotations:
        """Tool annotations based on HTTP method."""
        method = self._operation.method.upper()
        return ToolAnnotations(
            title=self._operation.summary or self._operation.operation_id,
            read_only_hint=method in ("GET", "HEAD", "OPTIONS"),
            destructive_hint=method == "DELETE",
            idempotent_hint=method in ("GET", "PUT", "DELETE"),
            open_world_hint=True,  # Always calls external API
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute the API call.

        HTTP Client Lifecycle (v3.2):
            - If http_client was provided: use it (caller manages lifecycle)
            - Otherwise: create a fresh client, use it, close it

        Args:
            arguments: Dict matching input_schema

        Returns:
            ToolResult with response data or error
        """
        # Determine which client to use
        if self._shared_client is not None:
            # Use caller-provided client (they manage lifecycle)
            client = self._shared_client
            close_after = False
        else:
            # Create fresh client for this request (we manage lifecycle)
            client = httpx.AsyncClient(timeout=self._timeout)
            close_after = True

        try:
            # Build the request
            url, params, body, headers = self._build_request(arguments)
            method = self._operation.method.upper()

            logger.info(
                f"[openapi_tool:{self.name}] {method} {url} "
                f"params={list(params.keys()) if params else None}"
            )

            # Execute request
            response = await client.request(
                method=method,
                url=url,
                params=params if params else None,
                json=body if body else None,
                headers=headers,
            )

            # Log response
            logger.info(
                f"[openapi_tool:{self.name}] Response: {response.status_code}"
            )

            # Handle error responses
            if response.status_code >= 400:
                error_text = response.text[:500]
                logger.warning(
                    f"[openapi_tool:{self.name}] Error {response.status_code}: {error_text}"
                )
                return ToolResult.error(
                    f"API error {response.status_code}: {error_text}",
                    structured={
                        "status_code": response.status_code,
                        "error": error_text,
                    },
                )

            # Parse successful response
            try:
                data = response.json()
                return ToolResult.success(
                    text=self._format_success(response.status_code, data),
                    structured=data,
                )
            except Exception:
                # Non-JSON response
                return ToolResult.success(
                    text=f"Success ({response.status_code}): {response.text[:500]}"
                )

        except httpx.TimeoutException:
            logger.error(f"[openapi_tool:{self.name}] Timeout after {self._timeout}s")
            return ToolResult.error(f"Request timed out after {self._timeout}s")

        except httpx.ConnectError as e:
            logger.error(f"[openapi_tool:{self.name}] Connection error: {e}")
            return ToolResult.error(f"Connection failed: {e}")

        except Exception as e:
            logger.error(f"[openapi_tool:{self.name}] Unexpected error: {e}")
            return ToolResult.error(f"Request failed: {e}")

        finally:
            # Clean up owned client (never close shared client)
            if close_after:
                await client.aclose()

    def _build_request(
        self, arguments: dict[str, Any]
    ) -> tuple[str, dict[str, Any], dict[str, Any] | None, dict[str, str]]:
        """
        Build URL, params, body, and headers from arguments.

        Distributes arguments to:
        - Path parameters (substituted into URL)
        - Query parameters (added to URL params)
        - Header parameters (added to headers)
        - Body (remaining arguments)

        Returns:
            Tuple of (url, params, body, headers)
        """
        path = self._operation.path
        params: dict[str, Any] = {}
        body: dict[str, Any] = {}
        headers: dict[str, str] = {**self._default_headers}

        # Add authentication headers
        self._add_auth_headers(headers)

        # Categorize parameters by location
        path_params = {
            p["name"]: p
            for p in self._operation.parameters
            if p.get("in") == "path"
        }
        query_params = {
            p["name"]: p
            for p in self._operation.parameters
            if p.get("in") == "query"
        }
        header_params = {
            p["name"]: p
            for p in self._operation.parameters
            if p.get("in") == "header"
        }

        # Distribute arguments
        for key, value in arguments.items():
            if value is None:
                continue

            if key in path_params:
                # Substitute into path
                path = path.replace(f"{{{key}}}", str(value))
            elif key in query_params:
                # Add to query params
                params[key] = value
            elif key in header_params:
                # Add to headers
                headers[key] = str(value)
            else:
                # Assume it's a body parameter
                body[key] = value

        # Ensure all path parameters are substituted
        if "{" in path:
            missing = re.findall(r"\{(\w+)\}", path)
            logger.warning(
                f"[openapi_tool:{self.name}] Missing path params: {missing}"
            )

        url = f"{self._base_url}{path}"

        return url, params, body if body else None, headers

    def _add_auth_headers(self, headers: dict[str, str]) -> None:
        """Add authentication headers based on config."""
        if "api_key" in self._auth:
            headers["Authorization"] = f"Bearer {self._auth['api_key']}"

        if "x_api_key" in self._auth:
            headers["X-API-Key"] = self._auth["x_api_key"]

        if "basic_user" in self._auth and "basic_pass" in self._auth:
            import base64

            credentials = f"{self._auth['basic_user']}:{self._auth['basic_pass']}"
            encoded = base64.b64encode(credentials.encode()).decode()
            headers["Authorization"] = f"Basic {encoded}"

    def _format_success(self, status_code: int, data: Any) -> str:
        """Format successful response for text output."""
        prefix = f"Success ({status_code})"

        if isinstance(data, dict):
            # Try common patterns
            if "id" in data:
                if "name" in data:
                    return f"{prefix}: Created/updated '{data['name']}' (id={data['id']})"
                return f"{prefix}: id={data['id']}"

            if "results" in data:
                count = len(data["results"])
                return f"{prefix}: {count} result(s)"

            if "data" in data and isinstance(data["data"], list):
                count = len(data["data"])
                return f"{prefix}: {count} item(s)"

            # Just count keys
            return f"{prefix}: Response with {len(data)} field(s)"

        elif isinstance(data, list):
            return f"{prefix}: {len(data)} item(s)"

        return f"{prefix}: {str(data)[:100]}"

    async def close(self) -> None:
        """
        Close owned HTTP client (no-op if using shared client).

        Note (v3.2):
            With the new per-request lifecycle, this method is typically
            not needed. Clients are closed automatically after each execute().
            This method exists for backwards compatibility and for cleanup
            if using the deprecated persistent client pattern.
        """
        if self._owned_client:
            await self._owned_client.aclose()
            self._owned_client = None
        # Never close shared client - caller manages it


# =============================================================================
# OpenAPI Toolkit
# =============================================================================


class OpenAPIToolkit:
    """
    Collection of tools from an OpenAPI specification.

    This toolkit:
    1. Parses an OpenAPI 3.x specification
    2. Creates Tool instances for each operation
    3. Provides methods to get tools for agent registration

    The toolkit is the main entry point for the "Universal Adapter" pattern.
    Instead of writing custom tools for each API, you load the OpenAPI spec
    and get ready-to-use tools.

    Usage:
        # From dict
        toolkit = OpenAPIToolkit.from_spec(spec_dict, base_url="...")

        # From URL
        toolkit = await OpenAPIToolkit.from_spec_url("https://...")

        # Get all tools
        tools = toolkit.get_tools()

        # Get filtered tools
        tools = toolkit.get_tools(tags=["projects"])

        # Get single tool
        tool = toolkit.get_tool("createWorkItem")

        # List available operations
        ops = toolkit.list_operations()
    """

    def __init__(
        self,
        operations: list[OpenAPIOperation],
        base_url: str,
        *,
        auth: dict[str, str] | None = None,
        default_headers: dict[str, str] | None = None,
        timeout: float = 30.0,
        spec_info: dict[str, Any] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ):
        """
        Initialize toolkit.

        Args:
            operations: Parsed OpenAPI operations
            base_url: API base URL
            auth: Authentication config
            default_headers: Default headers for all requests
            timeout: Request timeout
            spec_info: OpenAPI info section (title, version, etc.)
            http_client: Optional shared HTTP client for connection pooling.
                        If provided, all tools will use this client.
                        Caller is responsible for closing the client.
        """
        self._operations = {op.operation_id: op for op in operations}
        self._base_url = base_url
        self._auth = auth
        self._default_headers = default_headers
        self._timeout = timeout
        self._spec_info = spec_info or {}
        self._http_client = http_client
        self._tools: dict[str, OpenAPIOperationTool] = {}

    @property
    def title(self) -> str:
        """API title from spec."""
        return self._spec_info.get("title", "OpenAPI")

    @property
    def version(self) -> str:
        """API version from spec."""
        return self._spec_info.get("version", "unknown")

    @classmethod
    def from_spec(
        cls,
        spec: dict[str, Any],
        *,
        base_url: str | None = None,
        auth: dict[str, str] | None = None,
        default_headers: dict[str, str] | None = None,
        operations: list[str] | None = None,
        tags: list[str] | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> "OpenAPIToolkit":
        """
        Create toolkit from OpenAPI spec dict.

        Args:
            spec: OpenAPI 3.x specification dict
            base_url: Base URL (overrides spec's servers)
            auth: Authentication config
            default_headers: Default headers for all requests
            operations: Filter to specific operation IDs
            tags: Filter to operations with these tags
            timeout: Request timeout
            http_client: Optional shared HTTP client for connection pooling

        Returns:
            Configured OpenAPIToolkit

        Raises:
            ValueError: If base_url not provided and not in spec
        """
        # Get base URL from spec if not provided
        if not base_url:
            servers = spec.get("servers", [])
            if servers:
                base_url = servers[0].get("url", "")

        if not base_url:
            raise ValueError("base_url required (not found in spec)")

        # Build reference resolver for $ref
        resolver = _RefResolver(spec)

        # Parse operations
        parsed_ops = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            # Handle path-level parameters
            path_params = path_item.get("parameters", [])

            for method in ("get", "post", "put", "patch", "delete"):
                if method not in path_item:
                    continue

                op = path_item[method]
                op_id = op.get("operationId")

                if not op_id:
                    # Generate operation ID from method + path
                    op_id = _generate_operation_id(method, path)

                op_tags = tuple(op.get("tags", []))

                # Filter by operation ID
                if operations and op_id not in operations:
                    continue

                # Filter by tags
                if tags and not any(t in op_tags for t in tags):
                    continue

                # Combine path-level and operation-level parameters
                all_params = [*path_params, *op.get("parameters", [])]

                # Resolve parameter refs
                resolved_params = tuple(
                    resolver.resolve(p) for p in all_params
                )

                # Parse request body
                request_body = op.get("requestBody")
                body_schema = None
                body_required = False

                if request_body:
                    request_body = resolver.resolve(request_body)
                    body_required = request_body.get("required", False)
                    content = request_body.get("content", {})

                    # Prefer JSON content
                    for content_type in ("application/json", "application/x-www-form-urlencoded"):
                        if content_type in content:
                            body_schema = resolver.resolve(
                                content[content_type].get("schema", {})
                            )
                            break

                # Parse success response schema
                response_schema = None
                responses = op.get("responses", {})
                for status in ("200", "201", "202", "default"):
                    if status in responses:
                        resp = resolver.resolve(responses[status])
                        resp_content = resp.get("content", {})
                        if "application/json" in resp_content:
                            response_schema = resolver.resolve(
                                resp_content["application/json"].get("schema", {})
                            )
                        break

                parsed_op = OpenAPIOperation(
                    operation_id=op_id,
                    method=method,
                    path=path,
                    summary=op.get("summary", ""),
                    description=op.get("description", ""),
                    parameters=resolved_params,
                    request_body_schema=body_schema,
                    request_body_required=body_required,
                    response_schema=response_schema,
                    tags=op_tags,
                    security=tuple(op.get("security", [])),
                )
                parsed_ops.append(parsed_op)

        logger.info(
            f"[openapi_toolkit] Parsed {len(parsed_ops)} operations from spec"
        )

        return cls(
            operations=parsed_ops,
            base_url=base_url,
            auth=auth,
            default_headers=default_headers,
            timeout=timeout,
            spec_info=spec.get("info", {}),
            http_client=http_client,
        )

    @classmethod
    async def from_spec_url(
        cls,
        spec_url: str,
        *,
        base_url: str | None = None,
        auth: dict[str, str] | None = None,
        default_headers: dict[str, str] | None = None,
        operations: list[str] | None = None,
        tags: list[str] | None = None,
        timeout: float = 30.0,
        http_client: httpx.AsyncClient | None = None,
    ) -> "OpenAPIToolkit":
        """
        Create toolkit from OpenAPI spec URL.

        Args:
            spec_url: URL to OpenAPI JSON/YAML spec
            base_url: Base URL (overrides spec's servers)
            auth: Authentication config
            default_headers: Default headers for all requests
            operations: Filter to specific operation IDs
            tags: Filter to operations with these tags
            timeout: Request timeout
            http_client: Optional shared HTTP client for connection pooling

        Returns:
            Configured OpenAPIToolkit

        Raises:
            httpx.HTTPError: If spec fetch fails
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(spec_url)
            response.raise_for_status()

            # Try JSON first
            try:
                spec = response.json()
            except Exception:
                # Fall back to YAML
                try:
                    import yaml

                    spec = yaml.safe_load(response.text)
                except ImportError:
                    raise ValueError(
                        "YAML spec detected but PyYAML not installed. "
                        "Install with: pip install pyyaml"
                    )

        return cls.from_spec(
            spec,
            base_url=base_url,
            auth=auth,
            default_headers=default_headers,
            operations=operations,
            tags=tags,
            timeout=timeout,
            http_client=http_client,
        )

    def get_tool(self, operation_id: str) -> OpenAPIOperationTool:
        """
        Get a tool for a specific operation.

        Args:
            operation_id: Operation ID to get

        Returns:
            OpenAPIOperationTool for the operation

        Raises:
            KeyError: If operation not found
        """
        if operation_id not in self._tools:
            if operation_id not in self._operations:
                available = list(self._operations.keys())[:5]
                raise KeyError(
                    f"Unknown operation: '{operation_id}'. "
                    f"Available: {available}..."
                )

            self._tools[operation_id] = OpenAPIOperationTool(
                operation=self._operations[operation_id],
                base_url=self._base_url,
                auth=self._auth,
                default_headers=self._default_headers,
                timeout=self._timeout,
                http_client=self._http_client,  # Pass shared client
            )

        return self._tools[operation_id]

    def get_tools(
        self,
        *,
        tags: list[str] | None = None,
        operations: list[str] | None = None,
    ) -> list[Tool]:
        """
        Get all tools (optionally filtered).

        Args:
            tags: Filter to operations with these tags
            operations: Filter to specific operation IDs

        Returns:
            List of Tool instances
        """
        tools = []

        for op_id, op in self._operations.items():
            # Filter by operations list
            if operations and op_id not in operations:
                continue

            # Filter by tags
            if tags and not any(t in op.tags for t in tags):
                continue

            tools.append(self.get_tool(op_id))

        return tools

    def list_operations(
        self,
        *,
        tags: list[str] | None = None,
    ) -> list[str]:
        """
        List all available operation IDs.

        Args:
            tags: Filter to operations with these tags

        Returns:
            List of operation IDs
        """
        if not tags:
            return list(self._operations.keys())

        return [
            op_id
            for op_id, op in self._operations.items()
            if any(t in op.tags for t in tags)
        ]

    def get_operations_by_tag(self) -> dict[str, list[str]]:
        """
        Group operations by tag.

        Returns:
            Dict mapping tag name to list of operation IDs
        """
        by_tag: dict[str, list[str]] = {}

        for op_id, op in self._operations.items():
            for tag in op.tags:
                if tag not in by_tag:
                    by_tag[tag] = []
                by_tag[tag].append(op_id)

        return by_tag

    async def close(self) -> None:
        """Close all HTTP clients."""
        for tool in self._tools.values():
            await tool.close()

    def __repr__(self) -> str:
        return (
            f"OpenAPIToolkit(title={self.title!r}, "
            f"version={self.version!r}, "
            f"operations={len(self._operations)})"
        )


# =============================================================================
# Helper Functions
# =============================================================================


class _RefResolver:
    """
    Simple $ref resolver for OpenAPI specs.

    Handles local references like:
    - #/components/schemas/Task
    - #/components/parameters/workspace_slug
    """

    def __init__(self, spec: dict[str, Any]):
        self._spec = spec
        self._cache: dict[str, Any] = {}

    def resolve(self, obj: Any) -> Any:
        """Resolve $ref in an object."""
        if not isinstance(obj, dict):
            return obj

        if "$ref" not in obj:
            # Recursively resolve nested objects
            return {
                k: self.resolve(v) if isinstance(v, (dict, list)) else v
                for k, v in obj.items()
            }

        ref = obj["$ref"]

        # Check cache
        if ref in self._cache:
            return self._cache[ref]

        # Parse reference
        if not ref.startswith("#/"):
            logger.warning(f"[ref_resolver] Unsupported $ref: {ref}")
            return obj

        # Navigate to referenced object
        path = ref[2:].split("/")
        current = self._spec

        for part in path:
            # Handle URL-encoded characters
            part = part.replace("~1", "/").replace("~0", "~")
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                logger.warning(f"[ref_resolver] Could not resolve: {ref}")
                return obj

        # Resolve any nested refs and cache
        resolved = self.resolve(current)
        self._cache[ref] = resolved

        return resolved


def _generate_operation_id(method: str, path: str) -> str:
    """
    Generate operation ID from method and path.

    Examples:
        GET /projects -> get_projects
        POST /projects/{id}/tasks -> post_projects_id_tasks
    """
    # Clean path
    clean_path = path.strip("/")
    clean_path = re.sub(r"\{[^}]+\}", "by_id", clean_path)
    clean_path = clean_path.replace("/", "_").replace("-", "_")

    return f"{method}_{clean_path}"
