"""
Tool Base Classes (MCP-Aligned).

This module defines the core abstractions for v2 tools:
- Tool: Base class for all tools
- ToolResult: Result from tool execution
- ToolAnnotations: Behavioral hints for tools
- ContentBlock: Content blocks in tool results

Design Principle (ADR-005):
    Tools are the "hands" of the agent. They wrap existing v1 logic
    into callable units. Tools do NOT know they are called by an agent.

MCP Alignment:
    This interface follows Model Context Protocol standards:
    - Tool has name, description, input_schema
    - ToolResult has content blocks and is_error flag
    - Annotations are advisory hints only

Usage:
    class MyTool(Tool):
        @property
        def name(self) -> str:
            return "my_tool"

        @property
        def description(self) -> str:
            return "Does something useful"

        @property
        def input_schema(self) -> dict:
            return {
                "type": "object",
                "properties": {
                    "input": {"type": "string"}
                },
                "required": ["input"]
            }

        async def execute(self, arguments: dict) -> ToolResult:
            return ToolResult.success(f"Processed: {arguments['input']}")
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ContentType(Enum):
    """Type of content in a tool result (MCP-aligned)."""

    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    RESOURCE = "resource"
    RESOURCE_LINK = "resource_link"


@dataclass(frozen=True, slots=True)
class ContentBlock:
    """
    Content block in tool result (MCP-aligned).

    Supports multiple content types:
    - TEXT: Plain text result
    - IMAGE: Base64-encoded image data
    - AUDIO: Base64-encoded audio data
    - RESOURCE: Embedded resource content
    - RESOURCE_LINK: URI reference to a resource

    Example:
        # Text content
        ContentBlock.from_text("Task created successfully")

        # Structured content with text
        ContentBlock.from_text(
            "Created task: Mobile App Login",
            annotations={"priority": 0.9}
        )
    """

    type: ContentType
    text_content: str | None = None  # Renamed to avoid classmethod conflict
    data: bytes | None = None  # For binary content (image, audio)
    mime_type: str | None = None
    uri: str | None = None  # For resource links
    name: str | None = None  # Resource name
    annotations: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_text(
        cls,
        content: str,
        *,
        annotations: dict[str, Any] | None = None,
    ) -> ContentBlock:
        """Create a text content block."""
        return cls(
            type=ContentType.TEXT,
            text_content=content,
            annotations=annotations or {},
        )

    # Alias for backwards compatibility
    text = from_text  # This doesn't work with frozen dataclass, use from_text

    @classmethod
    def from_image(
        cls,
        data: bytes,
        mime_type: str = "image/png",
    ) -> ContentBlock:
        """Create an image content block."""
        return cls(
            type=ContentType.IMAGE,
            data=data,
            mime_type=mime_type,
        )

    @classmethod
    def from_resource_link(
        cls,
        uri: str,
        name: str | None = None,
        mime_type: str | None = None,
    ) -> ContentBlock:
        """Create a resource link content block."""
        return cls(
            type=ContentType.RESOURCE_LINK,
            uri=uri,
            name=name,
            mime_type=mime_type,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {"type": self.type.value}

        if self.text_content is not None:
            result["text"] = self.text_content
        if self.data is not None:
            import base64

            result["data"] = base64.b64encode(self.data).decode()
        if self.mime_type is not None:
            result["mimeType"] = self.mime_type
        if self.uri is not None:
            result["uri"] = self.uri
        if self.name is not None:
            result["name"] = self.name
        if self.annotations:
            result["annotations"] = self.annotations

        return result


@dataclass(frozen=True, slots=True)
class ToolAnnotations:
    """
    Behavioral hints for tools (MCP-aligned).

    These are ADVISORY only - they do not enforce behavior and should
    not be relied upon for security decisions.

    Attributes:
        title: Human-readable title for display
        read_only_hint: If True, tool does not modify environment
        destructive_hint: For non-read-only tools, may destroy data
        idempotent_hint: Repeated calls with same args have no additional effect
        open_world_hint: Tool interacts with external entities

    Example:
        # Read-only query tool
        ToolAnnotations(
            title="Query Tasks",
            read_only_hint=True,
            open_world_hint=True,  # Calls external API
        )

        # Destructive delete tool
        ToolAnnotations(
            title="Delete Task",
            read_only_hint=False,
            destructive_hint=True,
            idempotent_hint=True,  # Delete same ID = no-op
        )
    """

    title: str | None = None
    read_only_hint: bool = False
    destructive_hint: bool = True  # Default True for write operations
    idempotent_hint: bool = False
    open_world_hint: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {}

        if self.title is not None:
            result["title"] = self.title
        if self.read_only_hint:
            result["readOnlyHint"] = True
        if not self.destructive_hint:
            result["destructiveHint"] = False
        if self.idempotent_hint:
            result["idempotentHint"] = True
        if self.open_world_hint:
            result["openWorldHint"] = True

        return result


@dataclass(frozen=True, slots=True)
class ToolResult:
    """
    Result from tool execution (MCP-aligned).

    Every tool execution returns a ToolResult containing:
    - content: Array of content blocks (text, images, etc.)
    - is_error: Whether the execution failed
    - structured_content: Optional structured data matching output_schema

    Error Handling:
        Tool execution errors should be reported IN the result,
        not as exceptions. This allows the agent to reason about
        errors and potentially retry or adjust.

    Example:
        # Success with text
        ToolResult.success("Task created: ID-123")

        # Success with structured data
        ToolResult.success(
            text="Task created",
            structured={"task_id": "123", "name": "My Task"}
        )

        # Error
        ToolResult.error("Project not found: Mobile App")
    """

    content: tuple[ContentBlock, ...]
    is_error: bool = False
    structured_content: dict[str, Any] | None = None

    @classmethod
    def success(
        cls,
        text: str,
        *,
        structured: dict[str, Any] | None = None,
        additional_content: tuple[ContentBlock, ...] | None = None,
    ) -> ToolResult:
        """
        Create a successful result.

        Args:
            text: Human-readable result text
            structured: Optional structured data for programmatic use
            additional_content: Additional content blocks (images, etc.)

        Returns:
            ToolResult with is_error=False
        """
        content = [ContentBlock.from_text(text)]
        if additional_content:
            content.extend(additional_content)

        return cls(
            content=tuple(content),
            is_error=False,
            structured_content=structured,
        )

    @classmethod
    def error(
        cls,
        message: str,
        *,
        structured: dict[str, Any] | None = None,
    ) -> ToolResult:
        """
        Create an error result.

        Args:
            message: Error description
            structured: Optional structured error data

        Returns:
            ToolResult with is_error=True
        """
        return cls(
            content=(ContentBlock.from_text(f"Error: {message}"),),
            is_error=True,
            structured_content=structured,
        )

    @property
    def text(self) -> str:
        """Get the primary text content (convenience accessor)."""
        for block in self.content:
            if block.type == ContentType.TEXT and block.text_content:
                return block.text_content
        return ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result: dict[str, Any] = {
            "content": [block.to_dict() for block in self.content],
        }

        if self.is_error:
            result["isError"] = True
        if self.structured_content is not None:
            result["structuredContent"] = self.structured_content

        return result


class Tool(ABC):
    """
    Base class for all tools (MCP-aligned).

    Tools are the "hands" of the agent. They wrap existing logic
    into callable units that agents can select and invoke.

    Contract:
        - name: Unique identifier (snake_case recommended)
        - description: Clear description for LLM understanding
        - input_schema: JSON Schema for arguments
        - execute: Async method that performs the action

    Design Principle (ADR-005):
        Tools do NOT know they are called by an agent.
        They are independent, testable units.

    Example:
        class GreetTool(Tool):
            @property
            def name(self) -> str:
                return "greet_user"

            @property
            def description(self) -> str:
                return "Greet a user by name"

            @property
            def input_schema(self) -> dict:
                return {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "User's name"
                        }
                    },
                    "required": ["name"]
                }

            async def execute(self, arguments: dict) -> ToolResult:
                name = arguments.get("name", "World")
                return ToolResult.success(f"Hello, {name}!")
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for the tool.

        Convention: snake_case (e.g., "plane_create_task")
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Human-readable description of what the tool does.

        This is used by the LLM to understand when to use the tool.
        Be specific about:
        - What the tool does
        - When to use it
        - What inputs it expects
        """
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """
        JSON Schema defining expected input arguments.

        Must be a valid JSON Schema object with:
        - type: "object"
        - properties: dict of parameter definitions
        - required: list of required parameter names

        Example:
            {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Task name"
                    },
                    "priority": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "default": "medium"
                    }
                },
                "required": ["name"]
            }
        """
        ...

    @property
    def output_schema(self) -> dict[str, Any] | None:
        """
        Optional JSON Schema for structured output.

        If provided, structured_content in ToolResult should
        conform to this schema.
        """
        return None

    @property
    def annotations(self) -> ToolAnnotations:
        """
        Behavioral hints for the tool.

        Override to provide hints about:
        - read_only_hint: Does not modify environment
        - destructive_hint: May destroy data
        - idempotent_hint: Safe to retry
        - open_world_hint: Calls external services
        """
        return ToolAnnotations()

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Execute the tool with the given arguments.

        Args:
            arguments: Dict matching input_schema

        Returns:
            ToolResult with execution outcome

        Important:
            - Report errors in ToolResult.error(), don't raise exceptions
            - Exceptions should only be raised for unexpected failures
            - The agent can reason about errors returned in ToolResult
        """
        ...

    def to_llm_schema(self) -> dict[str, Any]:
        """
        Convert to schema format for LLM tool use.

        This format is compatible with Claude/OpenAI tool calling.
        """
        schema = {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

        if self.output_schema:
            schema["output_schema"] = self.output_schema

        return schema

    def to_mcp_schema(self) -> dict[str, Any]:
        """
        Convert to full MCP tool schema.

        Includes annotations and optional output schema.
        """
        schema = self.to_llm_schema()

        annotations = self.annotations.to_dict()
        if annotations:
            schema["annotations"] = annotations

        return schema

    def __repr__(self) -> str:
        return f"<Tool {self.name}>"
