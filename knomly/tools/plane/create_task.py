"""
Plane Create Task Tool.

This tool enables agents to create tasks in Plane project management.
It wraps the PlaneClient.create_work_item() method with:
- Project name resolution (via PlaneEntityCache)
- User name resolution for assignees
- Priority mapping

Design Principle (ADR-005):
    This tool wraps existing v1 logic. It does NOT know it's called by an agent.
    Errors are returned in ToolResult, not raised as exceptions.

Usage:
    tool = PlaneCreateTaskTool(client=plane_client, cache=entity_cache)

    result = await tool.execute({
        "name": "Fix login bug",
        "project": "Mobile App",
        "priority": "high",
        "description": "Users can't login on iOS",
    })

    if result.is_error:
        print(f"Failed: {result.text}")
    else:
        print(f"Created: {result.structured_content['task_id']}")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from knomly.tools.base import Tool, ToolAnnotations, ToolResult

if TYPE_CHECKING:
    from knomly.integrations.plane import PlaneClient
    from knomly.integrations.plane.cache import PlaneEntityCache

logger = logging.getLogger(__name__)


class PlaneCreateTaskTool(Tool):
    """
    Tool for creating tasks in Plane.

    This tool enables agents to create work items in Plane projects.
    It handles:
    - Project name → ID resolution via cache
    - User name → ID resolution for assignees
    - Priority string → enum mapping

    Example:
        tool = PlaneCreateTaskTool(client, cache)
        result = await tool.execute({
            "name": "Implement dark mode",
            "project": "Mobile App",  # Resolved to UUID
            "priority": "medium",
            "assignee": "Steve Jobs",  # Resolved to UUID
        })
    """

    def __init__(
        self,
        *,
        client: PlaneClient,
        cache: PlaneEntityCache,
    ):
        """
        Initialize the tool.

        Args:
            client: Plane API client
            cache: Entity cache for name resolution
        """
        self._client = client
        self._cache = cache

    @property
    def name(self) -> str:
        return "plane_create_task"

    @property
    def description(self) -> str:
        return (
            "Create a new task in Plane project management. "
            "Use this when the user wants to create a task, issue, "
            "work item, or todo in a specific project. "
            "Requires a task name and project name."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Task title/name. Be concise but descriptive.",
                },
                "project": {
                    "type": "string",
                    "description": (
                        "Project name or identifier where the task should be created. "
                        "Examples: 'Mobile App', 'Backend API', 'MOB', 'API'."
                    ),
                },
                "description": {
                    "type": "string",
                    "description": "Detailed description of the task (optional).",
                    "default": "",
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low", "none"],
                    "description": "Task priority level.",
                    "default": "medium",
                },
                "assignee": {
                    "type": "string",
                    "description": (
                        "User to assign the task to (name or email). " "Leave empty for unassigned."
                    ),
                    "default": "",
                },
                "due_date": {
                    "type": "string",
                    "description": "Due date in YYYY-MM-DD format (optional).",
                    "default": "",
                },
                "labels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of label names to apply (optional).",
                    "default": [],
                },
            },
            "required": ["name", "project"],
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "task_id": {
                    "type": "string",
                    "description": "UUID of the created task",
                },
                "task_name": {
                    "type": "string",
                    "description": "Name of the created task",
                },
                "project_id": {
                    "type": "string",
                    "description": "UUID of the project",
                },
                "sequence_id": {
                    "type": "integer",
                    "description": "Human-readable sequence number (e.g., 123 in MOB-123)",
                },
                "identifier": {
                    "type": "string",
                    "description": "Human-readable identifier (e.g., MOB-123)",
                },
            },
            "required": ["task_id", "task_name", "project_id"],
        }

    @property
    def annotations(self) -> ToolAnnotations:
        return ToolAnnotations(
            title="Create Plane Task",
            read_only_hint=False,
            destructive_hint=False,  # Creates, doesn't destroy
            idempotent_hint=False,  # Creates new task each time
            open_world_hint=True,  # Calls external Plane API
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Create a task in Plane.

        Args:
            arguments: Dict with name, project, and optional fields

        Returns:
            ToolResult with task details or error
        """
        try:
            # Extract arguments
            task_name = arguments.get("name", "").strip()
            project_name = arguments.get("project", "").strip()
            description = arguments.get("description", "").strip()
            priority = arguments.get("priority", "medium")
            assignee = arguments.get("assignee", "").strip()
            due_date = arguments.get("due_date", "").strip()

            # Validate required fields
            if not task_name:
                return ToolResult.error("Task name is required")

            if not project_name:
                return ToolResult.error("Project name is required")

            # Resolve project name to ID
            project_id = self._cache.resolve_project(project_name)
            if not project_id:
                available = list(self._cache.get_project_mapping().keys())[:5]
                return ToolResult.error(
                    f"Unknown project: '{project_name}'. " f"Available projects: {available}"
                )

            # Resolve assignee to ID (optional)
            assignees: list[str] | None = None
            if assignee:
                user_id = self._cache.resolve_user(assignee)
                if user_id:
                    assignees = [user_id]
                else:
                    # Log warning but don't fail - assignee is optional
                    logger.warning(
                        f"[plane_create_task] Unknown assignee: {assignee}. "
                        f"Creating task without assignee."
                    )

            # Create the task
            logger.info(f"[plane_create_task] Creating '{task_name}' in project {project_id}")

            work_item = await self._client.create_work_item(
                project_id=project_id,
                name=task_name,
                description=description if description else None,
                priority=priority,
                assignees=assignees,
                target_date=due_date if due_date else None,
            )

            # Build success response
            identifier = f"{work_item.project_identifier}-{work_item.sequence_id}"

            logger.info(f"[plane_create_task] Created task {identifier} (ID: {work_item.id})")

            return ToolResult.success(
                text=(
                    f"Created task '{work_item.name}' ({identifier}) "
                    f"in project. Task ID: {work_item.id}"
                ),
                structured={
                    "task_id": work_item.id,
                    "task_name": work_item.name,
                    "project_id": project_id,
                    "sequence_id": work_item.sequence_id,
                    "identifier": identifier,
                },
            )

        except Exception as e:
            logger.error(f"[plane_create_task] Failed to create task: {e}")
            return ToolResult.error(f"Failed to create task: {e}")
