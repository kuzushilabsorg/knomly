"""
Tool Definition Schema.

JSON-serializable schema for defining tools that can be stored in databases
and instantiated at runtime.

Design Principle:
    Tools are defined declaratively, not imperatively.
    A ToolDefinition contains all metadata needed to:
    1. Present the tool to an LLM (name, description, parameters)
    2. Build a live Tool instance at runtime (source, source_config)

Comparison to Pipecat FunctionSchema:
    Pipecat's FunctionSchema provides name, description, properties, required.
    We extend this with:
    - source: Where the tool implementation comes from (openapi, native, mcp)
    - source_config: Configuration for building the tool
    - tags: For filtering/grouping
    - enabled: For feature flags

Usage:
    # From JSON (database)
    tool_def = ToolDefinition.model_validate({
        "name": "create_task",
        "description": "Create a new task",
        "parameters": {
            "type": "object",
            "properties": {
                "project": {"type": "string", "description": "Project name"},
                "title": {"type": "string", "description": "Task title"},
            },
            "required": ["project", "title"],
        },
        "source": "openapi",
        "source_config": {
            "spec_url": "https://api.plane.so/openapi.json",
            "operation_id": "createWorkItem",
            "auth_secret_key": "plane_api_key",
        },
    })

    # To Tool instance (via adapter)
    adapter = get_adapter(tool_def.source)
    tool = await adapter.build_tool(tool_def, tool_context)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolParameter(BaseModel):
    """
    Single parameter definition for a tool.

    This is an alternative to inline JSON Schema for simpler definitions.
    Useful for UI-driven tool creation.

    Attributes:
        name: Parameter name
        type: JSON Schema type (string, number, boolean, array, object)
        description: Human-readable description
        required: Whether parameter is required
        default: Default value
        enum: Allowed values (for dropdowns)
    """

    name: str = Field(..., description="Parameter name")
    type: str = Field(default="string", description="JSON Schema type")
    description: str = Field(default="", description="Parameter description")
    required: bool = Field(default=False, description="Is required")
    default: Any = Field(default=None, description="Default value")
    enum: list[str] | None = Field(default=None, description="Allowed values")

    def to_json_schema(self) -> dict[str, Any]:
        """Convert to JSON Schema property."""
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.default is not None:
            schema["default"] = self.default
        if self.enum:
            schema["enum"] = self.enum
        return schema


class ToolDefinition(BaseModel):
    """
    Complete tool definition that can be stored in a database.

    This schema captures everything needed to:
    1. Register the tool with an LLM (name, description, parameters)
    2. Build a live Tool instance (source, source_config)
    3. Filter/organize tools (tags, enabled, category)

    Sources:
        - "openapi": Build from OpenAPI operation
        - "native": Built-in Python implementation
        - "mcp": Model Context Protocol tool
        - "function": Direct callable function

    Example (OpenAPI source):
        {
            "name": "create_task",
            "description": "Create a task in Plane",
            "parameters": {...},
            "source": "openapi",
            "source_config": {
                "spec_url": "https://api.plane.so/openapi.json",
                "operation_id": "createWorkItem",
                "base_url": "https://api.plane.so",
                "auth_type": "bearer",
                "auth_secret_key": "plane_api_key"
            }
        }

    Example (Native source):
        {
            "name": "send_email",
            "description": "Send an email",
            "parameters": {...},
            "source": "native",
            "source_config": {
                "class": "knomly.tools.email.SendEmailTool",
                "init_params": {"smtp_host": "..."}
            }
        }
    """

    # Core identity
    name: str = Field(..., description="Unique tool name (used as tool_id)")
    description: str = Field(..., description="Human-readable description for LLM")

    # LLM interface
    parameters: dict[str, Any] = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {},
            "required": [],
        },
        description="JSON Schema for tool parameters",
    )

    # Alternative: structured parameters (converted to JSON Schema)
    structured_params: list[ToolParameter] | None = Field(
        default=None,
        description="Structured parameters (alternative to raw JSON Schema)",
    )

    # Implementation source
    source: Literal["openapi", "native", "mcp", "function"] = Field(
        default="native",
        description="Where the tool implementation comes from",
    )
    source_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for building the tool from source",
    )

    # Metadata for organization
    tags: list[str] = Field(default_factory=list, description="Tags for filtering")
    category: str = Field(default="general", description="Tool category")
    enabled: bool = Field(default=True, description="Whether tool is active")

    # Annotations (hints for agent)
    read_only: bool = Field(default=False, description="Tool only reads, no writes")
    destructive: bool = Field(default=False, description="Tool may delete data")
    idempotent: bool = Field(default=False, description="Safe to retry")
    requires_confirmation: bool = Field(
        default=False,
        description="Should agent confirm before calling",
    )

    # Audit
    created_at: datetime | None = Field(default=None, description="Creation time")
    updated_at: datetime | None = Field(default=None, description="Last update time")
    version: int = Field(default=1, description="Schema version")

    class Config:
        populate_by_name = True

    def get_parameters_schema(self) -> dict[str, Any]:
        """
        Get JSON Schema for parameters.

        If structured_params is provided, converts to JSON Schema.
        Otherwise returns the parameters field directly.
        """
        if self.structured_params:
            properties = {}
            required = []

            for param in self.structured_params:
                properties[param.name] = param.to_json_schema()
                if param.required:
                    required.append(param.name)

            return {
                "type": "object",
                "properties": properties,
                "required": required,
            }

        return self.parameters

    def to_function_schema(self) -> dict[str, Any]:
        """
        Convert to Pipecat-compatible function schema.

        Returns dict matching the format expected by LLM tool use.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_parameters_schema(),
        }

    def to_openai_tool(self) -> dict[str, Any]:
        """
        Convert to OpenAI tool format.

        Returns dict for OpenAI's tools parameter.
        """
        return {
            "type": "function",
            "function": self.to_function_schema(),
        }

    @classmethod
    def from_function_schema(
        cls,
        name: str,
        description: str,
        properties: dict[str, Any],
        required: list[str] | None = None,
        **kwargs: Any,
    ) -> "ToolDefinition":
        """
        Create from Pipecat-style function schema.

        Args:
            name: Tool name
            description: Tool description
            properties: Parameter properties
            required: Required parameter names
            **kwargs: Additional ToolDefinition fields

        Returns:
            ToolDefinition instance
        """
        return cls(
            name=name,
            description=description,
            parameters={
                "type": "object",
                "properties": properties,
                "required": required or [],
            },
            **kwargs,
        )

    @classmethod
    def from_openapi_operation(
        cls,
        operation_id: str,
        method: str,
        path: str,
        summary: str,
        parameters: list[dict[str, Any]],
        request_body_schema: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        spec_url: str | None = None,
        base_url: str | None = None,
        auth_type: str = "bearer",
        auth_secret_key: str = "api_key",
    ) -> "ToolDefinition":
        """
        Create from parsed OpenAPI operation.

        This is used by the OpenAPI adapter to convert operations
        to ToolDefinitions that can be stored in a database.

        Args:
            operation_id: Unique operation ID
            method: HTTP method
            path: URL path template
            summary: Operation summary
            parameters: Path/query/header parameters
            request_body_schema: Request body JSON Schema
            tags: Operation tags
            spec_url: URL to OpenAPI spec
            base_url: API base URL
            auth_type: Authentication type (bearer, x_api_key, basic)
            auth_secret_key: Key in ToolContext.secrets for auth

        Returns:
            ToolDefinition ready for storage
        """
        # Build parameters schema from OpenAPI params
        properties: dict[str, Any] = {}
        required: list[str] = []

        for param in parameters:
            param_name = param.get("name", "")
            if not param_name:
                continue

            param_schema = param.get("schema", {"type": "string"})
            param_desc = param.get("description", "")
            param_in = param.get("in", "query")

            prop_schema = {**param_schema}
            if param_desc:
                loc_prefix = f"({param_in}) " if param_in != "query" else ""
                prop_schema["description"] = f"{loc_prefix}{param_desc}"

            properties[param_name] = prop_schema

            if param.get("required", False):
                required.append(param_name)

        # Add request body properties
        if request_body_schema and request_body_schema.get("type") == "object":
            body_props = request_body_schema.get("properties", {})
            body_required = request_body_schema.get("required", [])

            for prop_name, prop_schema in body_props.items():
                if prop_name not in properties:
                    properties[prop_name] = prop_schema
                    if prop_name in body_required:
                        required.append(prop_name)

        # Determine annotations from method
        is_read_only = method.upper() in ("GET", "HEAD", "OPTIONS")
        is_destructive = method.upper() == "DELETE"
        is_idempotent = method.upper() in ("GET", "PUT", "DELETE")

        return cls(
            name=operation_id,
            description=summary or f"{method.upper()} {path}",
            parameters={
                "type": "object",
                "properties": properties,
                "required": required,
            },
            source="openapi",
            source_config={
                "spec_url": spec_url,
                "base_url": base_url,
                "operation_id": operation_id,
                "method": method,
                "path": path,
                "auth_type": auth_type,
                "auth_secret_key": auth_secret_key,
            },
            tags=list(tags) if tags else [],
            read_only=is_read_only,
            destructive=is_destructive,
            idempotent=is_idempotent,
        )
