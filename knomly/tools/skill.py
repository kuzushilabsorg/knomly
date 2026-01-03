"""
Skill Abstraction (Phase 2.5).

A Skill is a bundle of related tools with shared configuration.
Think of it as a "capability set" the agent can use.

Design Rationale:
    - Tools are independent units; Skills are cohesive bundles
    - Skills manage shared resources (auth, connections, caches)
    - Skills provide lifecycle management (init/cleanup)
    - Skills enable domain-specific tool composition

Examples:
    - PlaneSkill: create_task, query_tasks, update_task (shared API key)
    - ZulipSkill: send_message, list_streams (shared bot token)
    - OpenAPISkill: Generated tools from any OpenAPI spec

Usage:
    # Create a skill from existing tools
    plane_skill = Skill(
        name="plane",
        description="Plane project management tools",
        tools=[PlaneCreateTaskTool(...), PlaneQueryTasksTool(...)],
    )

    # Or from OpenAPI spec
    plane_skill = Skill.from_openapi(
        name="plane",
        spec=plane_openapi_spec,
        auth={"Authorization": "Bearer xyz"},
    )

    # Get all tools for agent
    tools = plane_skill.get_tools()

    # Lifecycle
    await plane_skill.initialize()
    # ... use tools ...
    await plane_skill.cleanup()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from .generic import OpenAPIToolkit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .base import Tool

logger = logging.getLogger(__name__)


# =============================================================================
# Skill Protocol
# =============================================================================


@runtime_checkable
class SkillProtocol(Protocol):
    """
    Protocol for Skill implementations.

    A Skill is a bundle of related tools with:
    - Shared configuration (auth, base URLs, etc.)
    - Lifecycle management (initialize/cleanup)
    - Metadata for documentation and discovery

    This is a structural protocol - any class with these
    attributes/methods is compatible, no inheritance needed.
    """

    name: str
    """Unique identifier for this skill."""

    description: str
    """Human-readable description of what this skill does."""

    def get_tools(self) -> Sequence[Tool]:
        """Return all tools provided by this skill."""
        ...

    async def initialize(self) -> None:
        """
        Initialize the skill.

        Called before first tool use. Override to set up
        connections, caches, or other resources.
        """
        ...

    async def cleanup(self) -> None:
        """
        Clean up skill resources.

        Called when skill is no longer needed. Override to
        close connections, flush caches, etc.
        """
        ...


# =============================================================================
# Skill Implementation
# =============================================================================


@dataclass
class Skill:
    """
    A bundle of related tools with shared configuration.

    Attributes:
        name: Unique identifier for this skill
        description: Human-readable description
        tools: List of Tool instances
        auth: Shared authentication headers (applied to OpenAPI tools)
        metadata: Additional metadata for documentation

    Example:
        skill = Skill(
            name="plane",
            description="Manage tasks in Plane",
            tools=[create_task_tool, query_tasks_tool],
            auth={"Authorization": "Bearer xyz"},
        )

        # Get tools for agent
        for tool in skill.get_tools():
            print(f"  - {tool.name}: {tool.description}")
    """

    name: str
    description: str
    tools: list[Tool] = field(default_factory=list)
    auth: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    _initialized: bool = field(default=False, repr=False)

    def get_tools(self) -> Sequence[Tool]:
        """Return all tools in this skill."""
        return tuple(self.tools)

    def get_tool(self, name: str) -> Tool | None:
        """Get a specific tool by name."""
        for tool in self.tools:
            if tool.name == name:
                return tool
        return None

    def add_tool(self, tool: Tool) -> Skill:
        """Add a tool to this skill. Returns self for chaining."""
        self.tools.append(tool)
        return self

    async def initialize(self) -> None:
        """Initialize all tools that support initialization."""
        if self._initialized:
            return

        logger.debug(f"[skill:{self.name}] Initializing {len(self.tools)} tools")

        for tool in self.tools:
            if hasattr(tool, "initialize") and callable(tool.initialize):
                await tool.initialize()

        self._initialized = True
        logger.info(f"[skill:{self.name}] Initialized successfully")

    async def cleanup(self) -> None:
        """Clean up all tools that support cleanup."""
        if not self._initialized:
            return

        logger.debug(f"[skill:{self.name}] Cleaning up {len(self.tools)} tools")

        for tool in self.tools:
            if hasattr(tool, "cleanup") and callable(tool.cleanup):
                try:
                    await tool.cleanup()
                except Exception as e:
                    logger.warning(f"[skill:{self.name}] Cleanup error for {tool.name}: {e}")

        self._initialized = False
        logger.info(f"[skill:{self.name}] Cleaned up successfully")

    # -------------------------------------------------------------------------
    # Factory Methods
    # -------------------------------------------------------------------------

    @classmethod
    def from_openapi(
        cls,
        name: str,
        spec: dict[str, Any],
        *,
        description: str | None = None,
        base_url: str | None = None,
        auth: dict[str, str] | None = None,
        operations: list[str] | None = None,
        tags: list[str] | None = None,
        timeout: float = 30.0,
        metadata: dict[str, Any] | None = None,
    ) -> Skill:
        """
        Create a Skill from an OpenAPI specification.

        This is the recommended way to create skills from external APIs.
        The OpenAPI spec is parsed and tools are generated automatically.

        Args:
            name: Unique identifier for this skill
            spec: OpenAPI 3.x specification dict
            description: Human-readable description (defaults to spec info)
            base_url: Override base URL from spec
            auth: Authentication headers to include in requests
            operations: Filter to specific operation IDs
            tags: Filter to operations with these tags
            timeout: Request timeout in seconds
            metadata: Additional metadata

        Returns:
            Skill instance with generated tools

        Example:
            import yaml

            with open("plane_openapi.yaml") as f:
                spec = yaml.safe_load(f)

            plane_skill = Skill.from_openapi(
                name="plane",
                spec=spec,
                auth={"X-API-Key": "your-api-key"},
                tags=["work-items"],  # Only include work item operations
            )

            tools = plane_skill.get_tools()
            # [PlaneCreateWorkItem, PlaneListWorkItems, ...]
        """
        # Parse OpenAPI spec into toolkit
        toolkit = OpenAPIToolkit.from_spec(
            spec,
            base_url=base_url,
            auth=auth,
            operations=operations,
            tags=tags,
            timeout=timeout,
        )

        # Extract description from spec if not provided
        if description is None:
            info = spec.get("info", {})
            title = info.get("title", name)
            desc = info.get("description", "")
            description = f"{title}: {desc}" if desc else title

        # Build skill
        return cls(
            name=name,
            description=description,
            tools=list(toolkit.get_tools()),
            auth=auth or {},
            metadata=metadata or {},
        )

    @classmethod
    def from_tools(
        cls,
        name: str,
        tools: Sequence[Tool],
        *,
        description: str = "",
        auth: dict[str, str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Skill:
        """
        Create a Skill from existing Tool instances.

        Use this when you have manually created tools that should
        be bundled together as a skill.

        Args:
            name: Unique identifier for this skill
            tools: Sequence of Tool instances
            description: Human-readable description
            auth: Shared authentication headers
            metadata: Additional metadata

        Returns:
            Skill instance containing the tools

        Example:
            plane_skill = Skill.from_tools(
                name="plane",
                tools=[
                    PlaneCreateTaskTool(client),
                    PlaneQueryTasksTool(client),
                ],
                description="Manage Plane project tasks",
            )
        """
        return cls(
            name=name,
            description=description,
            tools=list(tools),
            auth=auth or {},
            metadata=metadata or {},
        )


# =============================================================================
# Skill Registry
# =============================================================================


class SkillRegistry:
    """
    Registry for managing multiple skills.

    Provides:
    - Skill registration and lookup
    - Aggregated tool access across skills
    - Lifecycle management for all skills

    Example:
        registry = SkillRegistry()
        registry.register(plane_skill)
        registry.register(zulip_skill)

        # Get all tools from all skills
        all_tools = registry.get_all_tools()

        # Or get tools from specific skill
        plane_tools = registry.get_skill("plane").get_tools()

        # Lifecycle
        await registry.initialize_all()
        # ... agent execution ...
        await registry.cleanup_all()
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> SkillRegistry:
        """
        Register a skill.

        Args:
            skill: Skill instance to register

        Returns:
            Self for chaining

        Raises:
            ValueError: If skill with same name already registered
        """
        if skill.name in self._skills:
            raise ValueError(f"Skill '{skill.name}' already registered")

        self._skills[skill.name] = skill
        logger.info(
            f"[skill_registry] Registered skill '{skill.name}' with {len(skill.tools)} tools"
        )
        return self

    def get_skill(self, name: str) -> Skill | None:
        """Get a skill by name."""
        return self._skills.get(name)

    def get_skills(self) -> Sequence[Skill]:
        """Get all registered skills."""
        return tuple(self._skills.values())

    def get_all_tools(self) -> Sequence[Tool]:
        """Get all tools from all registered skills."""
        tools: list[Tool] = []
        for skill in self._skills.values():
            tools.extend(skill.get_tools())
        return tuple(tools)

    def get_tool(self, name: str) -> Tool | None:
        """
        Find a tool by name across all skills.

        Args:
            name: Tool name to find

        Returns:
            Tool if found, None otherwise
        """
        for skill in self._skills.values():
            tool = skill.get_tool(name)
            if tool is not None:
                return tool
        return None

    async def initialize_all(self) -> None:
        """Initialize all registered skills."""
        for skill in self._skills.values():
            await skill.initialize()

    async def cleanup_all(self) -> None:
        """Clean up all registered skills."""
        for skill in self._skills.values():
            await skill.cleanup()

    def __len__(self) -> int:
        return len(self._skills)

    def __contains__(self, name: str) -> bool:
        return name in self._skills


# =============================================================================
# Convenience Functions
# =============================================================================


def create_skill_from_openapi(
    name: str,
    spec: dict[str, Any],
    *,
    auth: dict[str, str] | None = None,
    **kwargs: Any,
) -> Skill:
    """
    Convenience function to create a skill from OpenAPI spec.

    See Skill.from_openapi() for full documentation.
    """
    return Skill.from_openapi(name, spec, auth=auth, **kwargs)
