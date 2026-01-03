"""
Generic Tools for Knomly.

This module provides universal adapters that can interact with any API
without writing custom tool implementations.

Available Tools:
    - OpenAPIOperationTool: Tool for a single OpenAPI operation
    - OpenAPIToolkit: Bundle of tools from an OpenAPI spec

Usage:
    from knomly.tools.generic import OpenAPIToolkit

    # Create toolkit from spec
    toolkit = OpenAPIToolkit.from_spec(spec_dict, base_url="https://api.example.com")

    # Get tools for agent
    tools = toolkit.get_tools()

    # Or get a specific operation
    create_task = toolkit.get_tool("createTask")
"""

from .openapi import (
    OpenAPIOperation,
    OpenAPIOperationTool,
    OpenAPIToolkit,
)

__all__ = [
    "OpenAPIOperation",
    "OpenAPIOperationTool",
    "OpenAPIToolkit",
]
