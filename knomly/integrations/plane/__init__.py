"""
Plane Integration for Knomly.

Plane is an open-source project management tool. This integration provides:
- Work item (issue/task) management
- Project listing
- State and priority management

Usage:
    from knomly.integrations.plane import PlaneClient, PlaneConfig

    client = PlaneClient(PlaneConfig(
        api_key="plane_api_xxx",
        workspace_slug="my-workspace",
        base_url="https://api.plane.so",  # or self-hosted URL
    ))

    # Create a work item
    item = await client.create_work_item(
        project_id="project-123",
        name="Fix login bug",
        description="Users cannot log in with SSO",
        priority="high",
    )

    # List work items
    items = await client.list_work_items(
        project_id="project-123",
        state="in_progress",
    )

API Reference:
    https://developers.plane.so/api-reference/introduction

Note:
    Plane is deprecating /issues/ endpoints in favor of /work-items/.
    This client uses the new /work-items/ endpoints.
"""

from knomly.integrations.plane.cache import PlaneEntityCache
from knomly.integrations.plane.client import PlaneClient, PlaneConfig
from knomly.integrations.plane.schemas import (
    Project,
    WorkItem,
    WorkItemCreate,
    WorkItemPriority,
    WorkItemState,
    WorkItemUpdate,
)

__all__ = [
    "PlaneClient",
    "PlaneConfig",
    "PlaneEntityCache",
    "Project",
    "WorkItem",
    "WorkItemCreate",
    "WorkItemPriority",
    "WorkItemState",
    "WorkItemUpdate",
]
