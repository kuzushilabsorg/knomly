"""
Plane Query Tasks Tool.

This tool enables agents to list and search tasks in Plane.
It wraps the PlaneClient.list_work_items() method with:
- Project name resolution
- Filtering by status, priority, assignee
- Pagination handling

Design Principle (ADR-005):
    This is a read-only tool. It does NOT modify any data.
    Errors are returned in ToolResult, not raised as exceptions.

Usage:
    tool = PlaneQueryTasksTool(client=plane_client, cache=entity_cache)

    result = await tool.execute({
        "project": "Mobile App",
        "status": "in_progress",
        "limit": 10,
    })

    if not result.is_error:
        for task in result.structured_content["tasks"]:
            print(f"- {task['name']} ({task['identifier']})")
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from knomly.tools.base import Tool, ToolAnnotations, ToolResult

if TYPE_CHECKING:
    from knomly.integrations.plane import PlaneClient
    from knomly.integrations.plane.cache import PlaneEntityCache

logger = logging.getLogger(__name__)


class PlaneQueryTasksTool(Tool):
    """
    Tool for querying tasks in Plane.

    This is a read-only tool that lists work items from Plane projects.
    It handles:
    - Project name → ID resolution
    - Filtering by status, priority, assignee
    - Pagination (returns up to `limit` results)

    Example:
        tool = PlaneQueryTasksTool(client, cache)
        result = await tool.execute({
            "project": "Mobile App",
            "status": "in_progress",
            "limit": 5,
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
        return "plane_query_tasks"

    @property
    def description(self) -> str:
        return (
            "Query and list tasks from a Plane project. "
            "Use this to see what tasks exist, check their status, "
            "or find tasks matching specific criteria. "
            "This is a read-only operation."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "project": {
                    "type": "string",
                    "description": (
                        "Project name or identifier to query. "
                        "Examples: 'Mobile App', 'Backend API', 'MOB'."
                    ),
                },
                "status": {
                    "type": "string",
                    "enum": [
                        "backlog",
                        "todo",
                        "in_progress",
                        "in_review",
                        "done",
                        "cancelled",
                        "",
                    ],
                    "description": "Filter by task status (optional).",
                    "default": "",
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low", "none", ""],
                    "description": "Filter by priority (optional).",
                    "default": "",
                },
                "assignee": {
                    "type": "string",
                    "description": (
                        "Filter by assignee name or email (optional). " "Leave empty to show all."
                    ),
                    "default": "",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of tasks to return.",
                    "default": 10,
                    "minimum": 1,
                    "maximum": 50,
                },
            },
            "required": ["project"],
        }

    @property
    def output_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "tasks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "name": {"type": "string"},
                            "identifier": {"type": "string"},
                            "status": {"type": "string"},
                            "priority": {"type": "string"},
                        },
                    },
                },
                "total_count": {"type": "integer"},
                "has_more": {"type": "boolean"},
            },
            "required": ["tasks", "total_count", "has_more"],
        }

    @property
    def annotations(self) -> ToolAnnotations:
        return ToolAnnotations(
            title="Query Plane Tasks",
            read_only_hint=True,  # This tool only reads data
            destructive_hint=False,
            idempotent_hint=True,
            open_world_hint=True,  # Calls external Plane API
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Query tasks from Plane.

        Args:
            arguments: Dict with project and optional filters

        Returns:
            ToolResult with task list or error
        """
        try:
            # Extract arguments
            project_name = arguments.get("project", "").strip()
            arguments.get("status", "").strip()
            priority = arguments.get("priority", "").strip()
            assignee = arguments.get("assignee", "").strip()
            limit = min(arguments.get("limit", 10), 50)

            # Validate required fields
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
            assignee_id: str | None = None
            if assignee:
                assignee_id = self._cache.resolve_user(assignee)
                if not assignee_id:
                    logger.warning(
                        f"[plane_query_tasks] Unknown assignee: {assignee}. " f"Ignoring filter."
                    )

            # Query tasks
            logger.info(f"[plane_query_tasks] Querying tasks from project {project_id}")

            result = await self._client.list_work_items(
                project_id=project_id,
                priority=priority if priority else None,
                assignee_id=assignee_id,
                per_page=limit,
            )

            # Format tasks for response
            tasks = []
            for item in result.results:
                identifier = f"{item.project_identifier}-{item.sequence_id}"
                tasks.append(
                    {
                        "id": item.id,
                        "name": item.name,
                        "identifier": identifier,
                        "status": item.state_name or "unknown",
                        "priority": item.priority_label or "none",
                    }
                )

            # Build response text
            if not tasks:
                text = f"No tasks found in project '{project_name}'."
            else:
                lines = [f"Found {len(tasks)} task(s) in '{project_name}':"]
                for task in tasks:
                    lines.append(
                        f"- {task['identifier']}: {task['name']} "
                        f"[{task['status']}] ({task['priority']})"
                    )
                if result.has_next:
                    lines.append(f"(More tasks available, showing first {limit})")
                text = "\n".join(lines)

            logger.info(f"[plane_query_tasks] Found {len(tasks)} tasks in project {project_id}")

            return ToolResult.success(
                text=text,
                structured={
                    "tasks": tasks,
                    "total_count": len(tasks),
                    "has_more": result.has_next,
                },
            )

        except Exception as e:
            logger.error(f"[plane_query_tasks] Failed to query tasks: {e}")
            return ToolResult.error(f"Failed to query tasks: {e}")
