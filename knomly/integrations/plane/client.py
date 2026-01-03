"""
Plane API Client for Knomly.

This client provides async access to Plane's REST API for work item
(issue/task) management. It handles authentication, pagination, and
error mapping.

Usage:
    async with PlaneClient(config) as client:
        # Create work item
        item = await client.create_work_item(
            project_id="...",
            name="Fix bug",
            priority="high",
        )

        # List work items
        items = await client.list_work_items(project_id="...")

        # Update work item
        updated = await client.update_work_item(
            project_id="...",
            work_item_id="...",
            state_id="done-state-id",
        )

API Reference:
    https://developers.plane.so/api-reference/introduction
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from knomly.integrations.base import IntegrationClient, IntegrationConfig
from knomly.integrations.plane.schemas import (
    Project,
    ProjectList,
    WorkItem,
    WorkItemCreate,
    WorkItemList,
    WorkItemPriority,
    WorkItemQuery,
    WorkItemUpdate,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class PlaneConfig(IntegrationConfig):
    """Configuration for Plane client."""

    # Required
    api_key: str = ""
    workspace_slug: str = ""

    # Optional - defaults to Plane Cloud
    base_url: str = "https://api.plane.so"

    # Rate limiting (Plane allows 60 req/min)
    max_retries: int = 3
    retry_delay: float = 1.0

    def __post_init__(self):
        """Validate configuration."""
        if not self.api_key:
            raise ValueError("Plane API key is required")
        if not self.workspace_slug:
            raise ValueError("Plane workspace slug is required")


# =============================================================================
# Client
# =============================================================================


class PlaneClient(IntegrationClient):
    """
    Async client for Plane API.

    Provides methods for:
    - Project management (list, get)
    - Work item CRUD operations
    - State and label queries

    The client handles:
    - Authentication via X-API-Key header
    - Pagination with cursors
    - Error mapping to IntegrationError subtypes
    - Rate limit awareness
    """

    def __init__(self, config: PlaneConfig):
        """
        Initialize Plane client.

        Args:
            config: Plane configuration with API key and workspace
        """
        super().__init__(config)
        self._config: PlaneConfig = config

    @property
    def name(self) -> str:
        """Integration name."""
        return "plane"

    @property
    def workspace_slug(self) -> str:
        """Get workspace slug."""
        return self._config.workspace_slug

    def _get_auth_headers(self) -> dict[str, str]:
        """Return Plane authentication headers."""
        return {"X-API-Key": self._config.api_key}

    # =========================================================================
    # Projects
    # =========================================================================

    async def list_projects(self) -> list[Project]:
        """
        List all projects in the workspace.

        Returns:
            List of projects
        """
        response = await self._request(
            "GET",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/",
        )

        data = response.json()
        project_list = ProjectList(**data)
        return project_list.results

    async def get_project(self, project_id: str) -> Project:
        """
        Get a specific project.

        Args:
            project_id: Project UUID

        Returns:
            Project details
        """
        response = await self._request(
            "GET",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/",
        )

        return Project(**response.json())

    # =========================================================================
    # Work Items
    # =========================================================================

    async def create_work_item(
        self,
        project_id: str,
        name: str,
        *,
        description: str | None = None,
        description_html: str | None = None,
        priority: str | WorkItemPriority | None = None,
        state_id: str | None = None,
        assignees: list[str] | None = None,
        labels: list[str] | None = None,
        parent_id: str | None = None,
        start_date: str | None = None,
        target_date: str | None = None,
        estimate_point: int | None = None,
    ) -> WorkItem:
        """
        Create a new work item.

        Args:
            project_id: Project UUID
            name: Work item title
            description: Plain text description (converted to HTML)
            description_html: HTML description (takes precedence)
            priority: Priority level (urgent, high, medium, low, none)
            state_id: State UUID
            assignees: List of user UUIDs
            labels: List of label UUIDs
            parent_id: Parent work item UUID (for sub-items)
            start_date: Start date (YYYY-MM-DD)
            target_date: Target/due date (YYYY-MM-DD)
            estimate_point: Story points estimate

        Returns:
            Created work item
        """
        # Convert plain description to HTML if needed
        html_description = description_html
        if not html_description and description:
            html_description = f"<p>{description}</p>"

        # Normalize priority
        if isinstance(priority, str):
            priority = WorkItemPriority.from_string(priority)

        # Build create request
        create_data = WorkItemCreate(
            name=name,
            description_html=html_description,
            priority=priority,
            state_id=state_id,
            assignees=assignees,
            labels=labels,
            parent_id=parent_id,
            start_date=start_date,
            target_date=target_date,
            estimate_point=estimate_point,
        )

        logger.info(f"[plane] Creating work item: {name} in project {project_id}")

        response = await self._request(
            "POST",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/",
            json=create_data.to_api_dict(),
        )

        work_item = WorkItem(**response.json())
        logger.info(f"[plane] Created work item: {work_item.id}")

        return work_item

    async def get_work_item(
        self,
        project_id: str,
        work_item_id: str,
        *,
        expand: list[str] | None = None,
    ) -> WorkItem:
        """
        Get a specific work item.

        Args:
            project_id: Project UUID
            work_item_id: Work item UUID
            expand: Fields to expand (assignees, labels, state)

        Returns:
            Work item details
        """
        params = {}
        if expand:
            params["expand"] = ",".join(expand)

        response = await self._request(
            "GET",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/{work_item_id}/",
            params=params if params else None,
        )

        return WorkItem(**response.json())

    async def get_work_item_by_identifier(
        self,
        identifier: str,
        *,
        expand: list[str] | None = None,
    ) -> WorkItem:
        """
        Get a work item by its human-readable identifier (e.g., PROJ-123).

        Args:
            identifier: Human-readable identifier like PROJ-123
            expand: Fields to expand

        Returns:
            Work item details
        """
        params = {}
        if expand:
            params["expand"] = ",".join(expand)

        response = await self._request(
            "GET",
            f"/api/v1/workspaces/{self.workspace_slug}/work-items/{identifier}/",
            params=params if params else None,
        )

        return WorkItem(**response.json())

    async def list_work_items(
        self,
        project_id: str,
        *,
        state_id: str | None = None,
        priority: str | WorkItemPriority | None = None,
        assignee_id: str | None = None,
        label_id: str | None = None,
        per_page: int = 50,
        cursor: str | None = None,
        order_by: str = "-created_at",
    ) -> WorkItemList:
        """
        List work items in a project.

        Args:
            project_id: Project UUID
            state_id: Filter by state UUID
            priority: Filter by priority
            assignee_id: Filter by assignee UUID
            label_id: Filter by label UUID
            per_page: Results per page (max 100)
            cursor: Pagination cursor
            order_by: Sort field (prefix with - for descending)

        Returns:
            Paginated list of work items
        """
        # Normalize priority
        if isinstance(priority, str):
            priority = WorkItemPriority.from_string(priority)

        query = WorkItemQuery(
            state_id=state_id,
            priority=priority,
            assignee_id=assignee_id,
            label_id=label_id,
            per_page=per_page,
            cursor=cursor,
            order_by=order_by,
        )

        response = await self._request(
            "GET",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/",
            params=query.to_params(),
        )

        return WorkItemList(**response.json())

    async def list_all_work_items(
        self,
        project_id: str,
        *,
        state_id: str | None = None,
        priority: str | WorkItemPriority | None = None,
        max_items: int = 500,
    ) -> list[WorkItem]:
        """
        List all work items, handling pagination automatically.

        Args:
            project_id: Project UUID
            state_id: Filter by state UUID
            priority: Filter by priority
            max_items: Maximum items to fetch (safety limit)

        Returns:
            List of all matching work items
        """
        items: list[WorkItem] = []
        cursor: str | None = None

        while len(items) < max_items:
            result = await self.list_work_items(
                project_id=project_id,
                state_id=state_id,
                priority=priority,
                cursor=cursor,
                per_page=100,
            )

            items.extend(result.results)

            if not result.has_next:
                break

            cursor = result.next_cursor

        return items[:max_items]

    async def update_work_item(
        self,
        project_id: str,
        work_item_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        description_html: str | None = None,
        priority: str | WorkItemPriority | None = None,
        state_id: str | None = None,
        assignees: list[str] | None = None,
        labels: list[str] | None = None,
        parent_id: str | None = None,
        start_date: str | None = None,
        target_date: str | None = None,
        estimate_point: int | None = None,
    ) -> WorkItem:
        """
        Update an existing work item.

        Only provided fields will be updated.

        Args:
            project_id: Project UUID
            work_item_id: Work item UUID
            name: New title
            description: New plain text description
            description_html: New HTML description
            priority: New priority
            state_id: New state UUID
            assignees: New assignee list (replaces existing)
            labels: New label list (replaces existing)
            parent_id: New parent UUID
            start_date: New start date
            target_date: New target date
            estimate_point: New estimate

        Returns:
            Updated work item
        """
        # Convert plain description to HTML if needed
        html_description = description_html
        if not html_description and description:
            html_description = f"<p>{description}</p>"

        # Normalize priority
        if isinstance(priority, str):
            priority = WorkItemPriority.from_string(priority)

        update_data = WorkItemUpdate(
            name=name,
            description_html=html_description,
            priority=priority,
            state_id=state_id,
            assignees=assignees,
            labels=labels,
            parent_id=parent_id,
            start_date=start_date,
            target_date=target_date,
            estimate_point=estimate_point,
        )

        logger.info(f"[plane] Updating work item: {work_item_id}")

        response = await self._request(
            "PATCH",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/{work_item_id}/",
            json=update_data.to_api_dict(),
        )

        work_item = WorkItem(**response.json())
        logger.info(f"[plane] Updated work item: {work_item.id}")

        return work_item

    async def delete_work_item(
        self,
        project_id: str,
        work_item_id: str,
    ) -> bool:
        """
        Delete a work item.

        Args:
            project_id: Project UUID
            work_item_id: Work item UUID

        Returns:
            True if deleted successfully
        """
        logger.info(f"[plane] Deleting work item: {work_item_id}")

        await self._request(
            "DELETE",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/{work_item_id}/",
        )

        logger.info(f"[plane] Deleted work item: {work_item_id}")
        return True

    # =========================================================================
    # Health Check
    # =========================================================================

    async def health_check(self) -> bool:
        """
        Check if Plane API is reachable.

        Returns:
            True if healthy
        """
        try:
            await self.list_projects()
            return True
        except Exception as e:
            logger.warning(f"[plane] Health check failed: {e}")
            return False
