"""
Tests for OpenAPI Universal Adapter (Phase 2.4).

Tests cover:
- OpenAPIOperation parsing
- OpenAPIOperationTool schema generation and execution
- OpenAPIToolkit spec parsing and tool creation
- Reference resolution ($ref)
- Authentication handling
- Error handling

The "Universal Adapter" pattern eliminates manual tool writing by
generating tools from OpenAPI specifications.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import httpx

from knomly.tools.generic.openapi import (
    OpenAPIOperation,
    OpenAPIOperationTool,
    OpenAPIToolkit,
    _RefResolver,
    _generate_operation_id,
)
from knomly.tools.base import ToolResult


# =============================================================================
# Test Fixtures: Sample OpenAPI Specs
# =============================================================================


@pytest.fixture
def simple_spec():
    """Simple OpenAPI spec with one endpoint."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Simple API",
            "version": "1.0.0",
        },
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/items": {
                "get": {
                    "operationId": "listItems",
                    "summary": "List all items",
                    "tags": ["items"],
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {"type": "integer", "default": 10},
                            "description": "Maximum items to return",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of items",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "array",
                                        "items": {"type": "object"},
                                    }
                                }
                            },
                        }
                    },
                },
                "post": {
                    "operationId": "createItem",
                    "summary": "Create an item",
                    "tags": ["items"],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "name": {
                                            "type": "string",
                                            "description": "Item name",
                                        },
                                        "price": {
                                            "type": "number",
                                            "description": "Item price",
                                        },
                                    },
                                    "required": ["name"],
                                }
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Created item",
                        }
                    },
                },
            },
            "/items/{item_id}": {
                "get": {
                    "operationId": "getItem",
                    "summary": "Get an item by ID",
                    "tags": ["items"],
                    "parameters": [
                        {
                            "name": "item_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Item ID",
                        }
                    ],
                    "responses": {
                        "200": {"description": "Item details"},
                    },
                },
                "delete": {
                    "operationId": "deleteItem",
                    "summary": "Delete an item",
                    "tags": ["items"],
                    "parameters": [
                        {
                            "name": "item_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        }
                    ],
                    "responses": {
                        "204": {"description": "Item deleted"},
                    },
                },
            },
        },
    }


@pytest.fixture
def plane_like_spec():
    """Plane-like OpenAPI spec for realistic testing."""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "Plane API",
            "version": "1.0.0",
        },
        "servers": [{"url": "https://api.plane.so"}],
        "paths": {
            "/api/v1/workspaces/{workspace_slug}/projects/": {
                "get": {
                    "operationId": "listProjects",
                    "summary": "List all projects in a workspace",
                    "tags": ["projects"],
                    "parameters": [
                        {
                            "name": "workspace_slug",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Workspace slug",
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "List of projects",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "results": {
                                                "type": "array",
                                                "items": {"$ref": "#/components/schemas/Project"},
                                            }
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/api/v1/workspaces/{workspace_slug}/projects/{project_id}/work-items/": {
                "post": {
                    "operationId": "createWorkItem",
                    "summary": "Create a new work item (task)",
                    "tags": ["work-items"],
                    "parameters": [
                        {
                            "name": "workspace_slug",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "project_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "Project UUID",
                        },
                    ],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/WorkItemCreate"}
                            }
                        },
                    },
                    "responses": {
                        "201": {
                            "description": "Created work item",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/WorkItem"}
                                }
                            },
                        }
                    },
                },
                "get": {
                    "operationId": "listWorkItems",
                    "summary": "List work items in a project",
                    "tags": ["work-items"],
                    "parameters": [
                        {
                            "name": "workspace_slug",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "project_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                        },
                        {
                            "name": "priority",
                            "in": "query",
                            "schema": {
                                "type": "string",
                                "enum": ["urgent", "high", "medium", "low", "none"],
                            },
                            "description": "Filter by priority",
                        },
                    ],
                    "responses": {
                        "200": {"description": "List of work items"},
                    },
                },
            },
        },
        "components": {
            "schemas": {
                "Project": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "format": "uuid"},
                        "name": {"type": "string"},
                        "identifier": {"type": "string"},
                    },
                },
                "WorkItemCreate": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Work item title",
                        },
                        "description_html": {
                            "type": "string",
                            "description": "HTML description",
                        },
                        "priority": {
                            "type": "string",
                            "enum": ["urgent", "high", "medium", "low", "none"],
                            "description": "Priority level",
                        },
                    },
                    "required": ["name"],
                },
                "WorkItem": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                        "sequence_id": {"type": "integer"},
                    },
                },
            }
        },
    }


@pytest.fixture
def spec_with_refs():
    """OpenAPI spec with $ref references."""
    return {
        "openapi": "3.0.0",
        "info": {"title": "Ref Test", "version": "1.0.0"},
        "servers": [{"url": "https://api.example.com"}],
        "paths": {
            "/users": {
                "parameters": [
                    {"$ref": "#/components/parameters/PageSize"}
                ],
                "get": {
                    "operationId": "listUsers",
                    "summary": "List users",
                    "responses": {
                        "200": {
                            "description": "Users",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/UserList"}
                                }
                            },
                        }
                    },
                },
            },
        },
        "components": {
            "parameters": {
                "PageSize": {
                    "name": "page_size",
                    "in": "query",
                    "schema": {"type": "integer", "default": 50},
                    "description": "Results per page",
                },
            },
            "schemas": {
                "UserList": {
                    "type": "object",
                    "properties": {
                        "results": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/User"},
                        }
                    },
                },
                "User": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "name": {"type": "string"},
                    },
                },
            },
        },
    }


# =============================================================================
# OpenAPIOperation Tests
# =============================================================================


class TestOpenAPIOperation:
    """Tests for OpenAPIOperation dataclass."""

    def test_operation_creation(self):
        """Test creating an operation."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
            summary="List all items",
            tags=("items", "read"),
        )

        assert op.operation_id == "listItems"
        assert op.method == "get"
        assert op.path == "/items"
        assert "items" in op.tags

    def test_full_description_with_tags(self):
        """Test full_description includes tags."""
        op = OpenAPIOperation(
            operation_id="createItem",
            method="post",
            path="/items",
            summary="Create an item",
            tags=("items",),
        )

        assert "[items]" in op.full_description
        assert "Create an item" in op.full_description

    def test_full_description_fallback(self):
        """Test full_description fallback to method+path."""
        op = OpenAPIOperation(
            operation_id="customOp",
            method="post",
            path="/custom",
        )

        assert "POST /custom" in op.full_description


# =============================================================================
# OpenAPIOperationTool Tests
# =============================================================================


class TestOpenAPIOperationTool:
    """Tests for OpenAPIOperationTool."""

    def test_tool_name_from_operation(self):
        """Test tool name matches operation ID."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        assert tool.name == "listItems"

    def test_tool_description(self):
        """Test tool description from operation."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
            summary="List all items",
            tags=("items",),
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        assert "List all items" in tool.description
        assert "[items]" in tool.description

    def test_input_schema_from_parameters(self):
        """Test input schema generated from parameters."""
        op = OpenAPIOperation(
            operation_id="getItem",
            method="get",
            path="/items/{item_id}",
            parameters=(
                {
                    "name": "item_id",
                    "in": "path",
                    "required": True,
                    "schema": {"type": "string"},
                    "description": "Item ID",
                },
                {
                    "name": "include_details",
                    "in": "query",
                    "schema": {"type": "boolean"},
                    "description": "Include details",
                },
            ),
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        schema = tool.input_schema
        assert schema["type"] == "object"
        assert "item_id" in schema["properties"]
        assert "include_details" in schema["properties"]
        assert "item_id" in schema["required"]
        assert "include_details" not in schema["required"]

    def test_input_schema_from_request_body(self):
        """Test input schema includes request body properties."""
        op = OpenAPIOperation(
            operation_id="createItem",
            method="post",
            path="/items",
            request_body_schema={
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "price": {"type": "number"},
                },
                "required": ["name"],
            },
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        schema = tool.input_schema
        assert "name" in schema["properties"]
        assert "price" in schema["properties"]
        assert "name" in schema["required"]

    def test_annotations_for_get(self):
        """Test annotations for GET requests."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        annotations = tool.annotations
        assert annotations.read_only_hint is True
        assert annotations.destructive_hint is False
        assert annotations.idempotent_hint is True

    def test_annotations_for_delete(self):
        """Test annotations for DELETE requests."""
        op = OpenAPIOperation(
            operation_id="deleteItem",
            method="delete",
            path="/items/{id}",
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        annotations = tool.annotations
        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is True
        assert annotations.idempotent_hint is True

    def test_annotations_for_post(self):
        """Test annotations for POST requests."""
        op = OpenAPIOperation(
            operation_id="createItem",
            method="post",
            path="/items",
        )
        tool = OpenAPIOperationTool(op, "https://api.example.com")

        annotations = tool.annotations
        assert annotations.read_only_hint is False
        assert annotations.idempotent_hint is False

    @pytest.mark.asyncio
    async def test_execute_success(self):
        """Test successful execution."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
        )

        # Mock httpx client - pass via constructor (v3.2 lifecycle pattern)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"results": [{"id": "1", "name": "Item 1"}]}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        tool = OpenAPIOperationTool(
            op, "https://api.example.com", http_client=mock_client
        )
        result = await tool.execute({})

        assert result.is_error is False
        assert "1 result" in result.text
        assert result.structured_content["results"][0]["id"] == "1"

    @pytest.mark.asyncio
    async def test_execute_with_path_params(self):
        """Test execution with path parameters."""
        op = OpenAPIOperation(
            operation_id="getItem",
            method="get",
            path="/items/{item_id}",
            parameters=(
                {"name": "item_id", "in": "path", "required": True},
            ),
        )

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"id": "123", "name": "Test"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        tool = OpenAPIOperationTool(
            op, "https://api.example.com", http_client=mock_client
        )
        result = await tool.execute({"item_id": "123"})

        # Verify URL was built correctly
        call_args = mock_client.request.call_args
        assert call_args.kwargs["url"] == "https://api.example.com/items/123"

        assert result.is_error is False

    @pytest.mark.asyncio
    async def test_execute_with_body(self):
        """Test execution with request body."""
        op = OpenAPIOperation(
            operation_id="createItem",
            method="post",
            path="/items",
            request_body_schema={
                "type": "object",
                "properties": {"name": {"type": "string"}},
            },
        )

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": "new-123", "name": "New Item"}

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        tool = OpenAPIOperationTool(
            op, "https://api.example.com", http_client=mock_client
        )
        result = await tool.execute({"name": "New Item"})

        # Verify body was sent
        call_args = mock_client.request.call_args
        assert call_args.kwargs["json"] == {"name": "New Item"}

        assert result.is_error is False
        assert "New Item" in result.text

    @pytest.mark.asyncio
    async def test_execute_error_response(self):
        """Test handling of error responses."""
        op = OpenAPIOperation(
            operation_id="getItem",
            method="get",
            path="/items/{id}",
        )

        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Item not found"

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        tool = OpenAPIOperationTool(
            op, "https://api.example.com", http_client=mock_client
        )
        result = await tool.execute({"id": "nonexistent"})

        assert result.is_error is True
        assert "404" in result.text
        assert "Item not found" in result.text

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        """Test handling of timeout."""
        op = OpenAPIOperation(
            operation_id="slowOp",
            method="get",
            path="/slow",
        )

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))

        tool = OpenAPIOperationTool(
            op, "https://api.example.com", timeout=1.0, http_client=mock_client
        )
        result = await tool.execute({})

        assert result.is_error is True
        assert "timed out" in result.text.lower()

    def test_auth_bearer(self):
        """Test Bearer token authentication."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
        )
        tool = OpenAPIOperationTool(
            op,
            "https://api.example.com",
            auth={"api_key": "test-token"},
        )

        url, params, body, headers = tool._build_request({})

        assert headers["Authorization"] == "Bearer test-token"

    def test_auth_x_api_key(self):
        """Test X-API-Key authentication."""
        op = OpenAPIOperation(
            operation_id="listItems",
            method="get",
            path="/items",
        )
        tool = OpenAPIOperationTool(
            op,
            "https://api.example.com",
            auth={"x_api_key": "secret-key"},
        )

        url, params, body, headers = tool._build_request({})

        assert headers["X-API-Key"] == "secret-key"


# =============================================================================
# OpenAPIToolkit Tests
# =============================================================================


class TestOpenAPIToolkit:
    """Tests for OpenAPIToolkit."""

    def test_from_spec_parses_operations(self, simple_spec):
        """Test toolkit parses all operations."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        ops = toolkit.list_operations()
        assert "listItems" in ops
        assert "createItem" in ops
        assert "getItem" in ops
        assert "deleteItem" in ops

    def test_from_spec_with_base_url_override(self, simple_spec):
        """Test base_url can be overridden."""
        toolkit = OpenAPIToolkit.from_spec(
            simple_spec,
            base_url="https://custom.api.com",
        )

        tool = toolkit.get_tool("listItems")
        url, _, _, _ = tool._build_request({})

        assert url.startswith("https://custom.api.com")

    def test_from_spec_filter_by_operations(self, simple_spec):
        """Test filtering by operation IDs."""
        toolkit = OpenAPIToolkit.from_spec(
            simple_spec,
            operations=["listItems", "createItem"],
        )

        ops = toolkit.list_operations()
        assert "listItems" in ops
        assert "createItem" in ops
        assert "getItem" not in ops
        assert "deleteItem" not in ops

    def test_from_spec_filter_by_tags(self, simple_spec):
        """Test filtering by tags."""
        # All operations have "items" tag, so this should get all
        toolkit = OpenAPIToolkit.from_spec(
            simple_spec,
            tags=["items"],
        )

        ops = toolkit.list_operations()
        assert len(ops) == 4

    def test_get_tool_returns_tool(self, simple_spec):
        """Test get_tool returns Tool instance."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        tool = toolkit.get_tool("listItems")

        assert tool.name == "listItems"
        assert "List all items" in tool.description

    def test_get_tool_caches_tools(self, simple_spec):
        """Test get_tool caches and reuses tools."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        tool1 = toolkit.get_tool("listItems")
        tool2 = toolkit.get_tool("listItems")

        assert tool1 is tool2

    def test_get_tool_unknown_raises(self, simple_spec):
        """Test get_tool raises for unknown operation."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        with pytest.raises(KeyError, match="Unknown operation"):
            toolkit.get_tool("nonexistent")

    def test_get_tools_returns_all(self, simple_spec):
        """Test get_tools returns all tools."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        tools = toolkit.get_tools()

        assert len(tools) == 4
        names = [t.name for t in tools]
        assert "listItems" in names
        assert "createItem" in names

    def test_get_tools_filter_by_tags(self, plane_like_spec):
        """Test get_tools filters by tags."""
        toolkit = OpenAPIToolkit.from_spec(plane_like_spec)

        # Get only project operations
        project_tools = toolkit.get_tools(tags=["projects"])
        assert len(project_tools) == 1
        assert project_tools[0].name == "listProjects"

        # Get only work-item operations
        work_item_tools = toolkit.get_tools(tags=["work-items"])
        assert len(work_item_tools) == 2

    def test_get_operations_by_tag(self, plane_like_spec):
        """Test grouping operations by tag."""
        toolkit = OpenAPIToolkit.from_spec(plane_like_spec)

        by_tag = toolkit.get_operations_by_tag()

        assert "projects" in by_tag
        assert "listProjects" in by_tag["projects"]
        assert "work-items" in by_tag
        assert "createWorkItem" in by_tag["work-items"]

    def test_spec_info(self, simple_spec):
        """Test toolkit exposes spec info."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        assert toolkit.title == "Simple API"
        assert toolkit.version == "1.0.0"

    def test_repr(self, simple_spec):
        """Test toolkit repr."""
        toolkit = OpenAPIToolkit.from_spec(simple_spec)

        repr_str = repr(toolkit)
        assert "Simple API" in repr_str
        assert "operations=4" in repr_str


# =============================================================================
# Reference Resolution Tests
# =============================================================================


class TestRefResolver:
    """Tests for $ref resolution."""

    def test_resolve_schema_ref(self, spec_with_refs):
        """Test resolving schema references."""
        resolver = _RefResolver(spec_with_refs)

        ref_obj = {"$ref": "#/components/schemas/User"}
        resolved = resolver.resolve(ref_obj)

        assert resolved["type"] == "object"
        assert "id" in resolved["properties"]
        assert "name" in resolved["properties"]

    def test_resolve_parameter_ref(self, spec_with_refs):
        """Test resolving parameter references."""
        resolver = _RefResolver(spec_with_refs)

        ref_obj = {"$ref": "#/components/parameters/PageSize"}
        resolved = resolver.resolve(ref_obj)

        assert resolved["name"] == "page_size"
        assert resolved["in"] == "query"

    def test_resolve_nested_refs(self, spec_with_refs):
        """Test resolving nested references."""
        resolver = _RefResolver(spec_with_refs)

        # UserList has results with $ref to User
        ref_obj = {"$ref": "#/components/schemas/UserList"}
        resolved = resolver.resolve(ref_obj)

        # The nested User ref should also be resolved
        assert resolved["properties"]["results"]["items"]["type"] == "object"
        assert "id" in resolved["properties"]["results"]["items"]["properties"]

    def test_resolve_caches_results(self, spec_with_refs):
        """Test that resolution is cached."""
        resolver = _RefResolver(spec_with_refs)

        ref = "#/components/schemas/User"
        result1 = resolver.resolve({"$ref": ref})
        result2 = resolver.resolve({"$ref": ref})

        # Should be same object (cached)
        assert result1 is result2


class TestToolkitWithRefs:
    """Tests for toolkit with $ref resolution."""

    def test_parses_spec_with_refs(self, spec_with_refs):
        """Test toolkit parses spec with $ref."""
        toolkit = OpenAPIToolkit.from_spec(spec_with_refs)

        tool = toolkit.get_tool("listUsers")

        # Path-level parameter should be resolved
        schema = tool.input_schema
        assert "page_size" in schema["properties"]

    def test_parses_plane_like_spec(self, plane_like_spec):
        """Test toolkit parses Plane-like spec with refs."""
        toolkit = OpenAPIToolkit.from_spec(plane_like_spec)

        # createWorkItem should have body properties from WorkItemCreate
        tool = toolkit.get_tool("createWorkItem")
        schema = tool.input_schema

        assert "name" in schema["properties"]
        assert "priority" in schema["properties"]
        assert "name" in schema["required"]


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestGenerateOperationId:
    """Tests for operation ID generation."""

    def test_simple_path(self):
        """Test simple path."""
        op_id = _generate_operation_id("get", "/items")
        assert op_id == "get_items"

    def test_path_with_params(self):
        """Test path with parameters."""
        op_id = _generate_operation_id("get", "/items/{item_id}")
        assert op_id == "get_items_by_id"

    def test_nested_path(self):
        """Test nested path."""
        op_id = _generate_operation_id("post", "/projects/{id}/tasks")
        assert op_id == "post_projects_by_id_tasks"

    def test_path_with_dashes(self):
        """Test path with dashes."""
        op_id = _generate_operation_id("get", "/work-items")
        assert op_id == "get_work_items"


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_plane_like_workflow(self, plane_like_spec):
        """Test a realistic Plane-like workflow."""
        toolkit = OpenAPIToolkit.from_spec(
            plane_like_spec,
            auth={"x_api_key": "test-key"},
        )

        # Get list projects tool
        list_projects = toolkit.get_tool("listProjects")
        assert list_projects.name == "listProjects"

        # Verify schema has workspace_slug
        schema = list_projects.input_schema
        assert "workspace_slug" in schema["properties"]
        assert "workspace_slug" in schema["required"]

        # Get create work item tool
        create_item = toolkit.get_tool("createWorkItem")
        schema = create_item.input_schema

        # Should have path params + body params
        assert "workspace_slug" in schema["properties"]
        assert "project_id" in schema["properties"]
        assert "name" in schema["properties"]
        assert "priority" in schema["properties"]

        # Required should include path params and body required
        assert "workspace_slug" in schema["required"]
        assert "project_id" in schema["required"]
        assert "name" in schema["required"]

    @pytest.mark.asyncio
    async def test_execute_create_work_item(self, plane_like_spec):
        """Test executing create work item."""
        # Mock response
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": "task-uuid-123",
            "name": "Fix login bug",
            "sequence_id": 42,
        }

        mock_client = AsyncMock()
        mock_client.request = AsyncMock(return_value=mock_response)

        # Pass mock client through toolkit (v3.2 lifecycle pattern)
        toolkit = OpenAPIToolkit.from_spec(
            plane_like_spec,
            auth={"x_api_key": "test-key"},
            http_client=mock_client,
        )

        tool = toolkit.get_tool("createWorkItem")

        result = await tool.execute({
            "workspace_slug": "my-workspace",
            "project_id": "project-uuid",
            "name": "Fix login bug",
            "priority": "high",
        })

        # Verify request
        call_args = mock_client.request.call_args
        assert call_args.kwargs["method"] == "POST"
        assert "/my-workspace/" in call_args.kwargs["url"]
        assert "/project-uuid/" in call_args.kwargs["url"]
        assert call_args.kwargs["json"]["name"] == "Fix login bug"
        assert call_args.kwargs["json"]["priority"] == "high"
        assert call_args.kwargs["headers"]["X-API-Key"] == "test-key"

        assert result.is_error is False
        assert "Fix login bug" in result.text
        assert result.structured_content["id"] == "task-uuid-123"
