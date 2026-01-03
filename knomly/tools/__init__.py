"""
Knomly Tools (v2 Agentic Layer).

Tools are the "hands" of the agent. They wrap existing v1 logic
into callable units that agents can select and invoke.

Design Principle (ADR-005):
    - Tools are independent, testable units
    - Tool results flow through ToolResultFrame
    - Tools do NOT know they are called by an agent

MCP Alignment:
    Tool interface follows Model Context Protocol standards.
    See: https://modelcontextprotocol.io/specification/

Usage:
    # Define a tool
    class MyTool(Tool):
        @property
        def name(self) -> str:
            return "my_tool"

        async def execute(self, arguments: dict) -> ToolResult:
            return ToolResult.success("Done!")

    # Register and use
    registry = ToolRegistry()
    registry.register(MyTool())

    result = await registry.get("my_tool").execute({"arg": "value"})
"""

from .base import (
    Tool,
    ToolResult,
    ToolAnnotations,
    ContentBlock,
    ContentType,
)
from .registry import ToolRegistry
from .generic import (
    OpenAPIOperation,
    OpenAPIOperationTool,
    OpenAPIToolkit,
)
from .skill import (
    Skill,
    SkillProtocol,
    SkillRegistry,
    create_skill_from_openapi,
)
from .factory import (
    ToolContext,
    ToolFactory,
    StaticToolFactory,
    CompositeToolFactory,
    ConditionalToolFactory,
    extract_tool_context_from_frame,
)

__all__ = [
    # Core Tool Protocol
    "Tool",
    "ToolResult",
    "ToolAnnotations",
    "ContentBlock",
    "ContentType",
    "ToolRegistry",
    # Generic Tools (Phase 2.4)
    "OpenAPIOperation",
    "OpenAPIOperationTool",
    "OpenAPIToolkit",
    # Skill Abstraction (Phase 2.5)
    "Skill",
    "SkillProtocol",
    "SkillRegistry",
    "create_skill_from_openapi",
    # Tool Factory (v2.1 Multi-Tenancy)
    "ToolContext",
    "ToolFactory",
    "StaticToolFactory",
    "CompositeToolFactory",
    "ConditionalToolFactory",
    "extract_tool_context_from_frame",
]
