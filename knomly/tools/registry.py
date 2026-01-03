"""
Tool Registry for v2 Agentic Layer.

The registry manages available tools for agents:
- Registration with validation
- Lookup by name
- Schema export for LLM

Design Principle (ADR-005):
    Tools are registered once at startup and immutable during execution.
    The registry is passed to AgentProcessor for tool discovery.

Usage:
    registry = ToolRegistry()
    registry.register(PlaneCreateTaskTool(client, cache))
    registry.register(PlaneQueryTasksTool(client))

    # Get tool by name
    tool = registry.get("plane_create_task")

    # Get all schemas for LLM
    schemas = registry.to_llm_schemas()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .base import Tool

logger = logging.getLogger(__name__)


class ToolRegistryError(Exception):
    """Error in tool registry operations."""

    pass


class ToolRegistry:
    """
    Registry of available tools for agents.

    Thread-safe registry for tool management. Tools are registered
    by name and can be looked up for execution.

    Example:
        registry = ToolRegistry()

        # Register tools
        registry.register(PlaneCreateTaskTool(client, cache))
        registry.register(ZulipSendTool(zulip_client))

        # Use in agent
        agent = AgentProcessor(tools=registry)

        # Get specific tool
        tool = registry.get("plane_create_task")
        result = await tool.execute({"name": "My Task", "project": "Mobile"})
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Register a tool.

        Args:
            tool: Tool instance to register

        Raises:
            ToolRegistryError: If tool name already registered
        """
        if tool.name in self._tools:
            raise ToolRegistryError(
                f"Tool '{tool.name}' already registered. Use a unique name or unregister first."
            )

        # Validate tool has required properties
        self._validate_tool(tool)

        self._tools[tool.name] = tool
        logger.info(f"[tool_registry] Registered tool: {tool.name}")

    def unregister(self, name: str) -> bool:
        """
        Unregister a tool by name.

        Args:
            name: Tool name to unregister

        Returns:
            True if tool was unregistered, False if not found
        """
        if name in self._tools:
            del self._tools[name]
            logger.info(f"[tool_registry] Unregistered tool: {name}")
            return True
        return False

    def get(self, name: str) -> Tool | None:
        """
        Get a tool by name.

        Args:
            name: Tool name

        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(name)

    def get_required(self, name: str) -> Tool:
        """
        Get a tool by name, raising if not found.

        Args:
            name: Tool name

        Returns:
            Tool instance

        Raises:
            ToolRegistryError: If tool not found
        """
        tool = self._tools.get(name)
        if tool is None:
            available = list(self._tools.keys())
            raise ToolRegistryError(f"Tool '{name}' not found. Available tools: {available}")
        return tool

    def list_tools(self) -> list[Tool]:
        """
        List all registered tools.

        Returns:
            List of Tool instances
        """
        return list(self._tools.values())

    def list_names(self) -> list[str]:
        """
        List all registered tool names.

        Returns:
            List of tool names
        """
        return list(self._tools.keys())

    def to_llm_schemas(self) -> list[dict[str, Any]]:
        """
        Get all tool schemas for LLM tool use.

        Returns:
            List of tool schemas compatible with Claude/OpenAI
        """
        return [tool.to_llm_schema() for tool in self._tools.values()]

    def to_mcp_schemas(self) -> list[dict[str, Any]]:
        """
        Get all tool schemas in full MCP format.

        Returns:
            List of MCP-compliant tool schemas
        """
        return [tool.to_mcp_schema() for tool in self._tools.values()]

    def _validate_tool(self, tool: Tool) -> None:
        """
        Validate tool has required properties.

        Raises:
            ToolRegistryError: If tool is invalid
        """
        # Check name
        if not tool.name or not isinstance(tool.name, str):
            raise ToolRegistryError(f"Tool must have a valid name: {tool}")

        # Check description
        if not tool.description or not isinstance(tool.description, str):
            raise ToolRegistryError(f"Tool '{tool.name}' must have a description")

        # Check input_schema is a valid JSON Schema object
        schema = tool.input_schema
        if not isinstance(schema, dict):
            raise ToolRegistryError(f"Tool '{tool.name}' input_schema must be a dict")

        if schema.get("type") != "object":
            raise ToolRegistryError(f"Tool '{tool.name}' input_schema must have type: 'object'")

        if "properties" not in schema:
            raise ToolRegistryError(f"Tool '{tool.name}' input_schema must have 'properties'")

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"<ToolRegistry tools={list(self._tools.keys())}>"


# =============================================================================
# Factory Functions
# =============================================================================


def create_default_registry() -> ToolRegistry:
    """
    Create a registry with default tools.

    This is a convenience function for common setups.
    Specific tools are registered based on available clients.

    Returns:
        ToolRegistry with common tools registered
    """
    registry = ToolRegistry()

    # Tools are registered by the application based on available clients
    # This function is a placeholder for future default tool registration

    return registry
