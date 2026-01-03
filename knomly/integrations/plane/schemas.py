"""
Pydantic schemas for Plane API.

These schemas provide type-safe representations of Plane resources
with validation and serialization support.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# =============================================================================
# Enums
# =============================================================================


class WorkItemPriority(str, Enum):
    """Work item priority levels."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> WorkItemPriority:
        """Convert string to priority, with fallback."""
        try:
            return cls(value.lower())
        except ValueError:
            return cls.NONE


class WorkItemState(str, Enum):
    """Common work item states (actual states are project-specific)."""

    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    CANCELLED = "cancelled"


# =============================================================================
# Request Schemas
# =============================================================================


class WorkItemCreate(BaseModel):
    """Schema for creating a work item."""

    model_config = ConfigDict(use_enum_values=True)

    name: str = Field(..., min_length=1, max_length=255, description="Work item title")
    description_html: str | None = Field(None, description="HTML description")
    priority: WorkItemPriority | None = Field(None, description="Priority level")
    state_id: str | None = Field(None, description="State UUID")
    assignees: list[str] | None = Field(None, description="List of assignee UUIDs")
    labels: list[str] | None = Field(None, description="List of label UUIDs")
    parent_id: str | None = Field(None, description="Parent work item UUID")
    start_date: str | None = Field(None, description="Start date (YYYY-MM-DD)")
    target_date: str | None = Field(None, description="Target date (YYYY-MM-DD)")
    estimate_point: int | None = Field(None, ge=0, description="Story points")

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to API request format, excluding None values."""
        data = {}
        if self.name:
            data["name"] = self.name
        if self.description_html:
            data["description_html"] = self.description_html
        if self.priority:
            data["priority"] = (
                self.priority.value
                if isinstance(self.priority, WorkItemPriority)
                else self.priority
            )
        if self.state_id:
            data["state_id"] = self.state_id
        if self.assignees:
            data["assignees"] = self.assignees
        if self.labels:
            data["labels"] = self.labels
        if self.parent_id:
            data["parent_id"] = self.parent_id
        if self.start_date:
            data["start_date"] = self.start_date
        if self.target_date:
            data["target_date"] = self.target_date
        if self.estimate_point is not None:
            data["estimate_point"] = self.estimate_point
        return data


class WorkItemUpdate(BaseModel):
    """Schema for updating a work item."""

    model_config = ConfigDict(use_enum_values=True)

    name: str | None = Field(None, min_length=1, max_length=255)
    description_html: str | None = None
    priority: WorkItemPriority | None = None
    state_id: str | None = None
    assignees: list[str] | None = None
    labels: list[str] | None = None
    parent_id: str | None = None
    start_date: str | None = None
    target_date: str | None = None
    estimate_point: int | None = None

    def to_api_dict(self) -> dict[str, Any]:
        """Convert to API request format, excluding None values."""
        data = {}
        for field_name, value in self.model_dump(exclude_none=True).items():
            if isinstance(value, Enum):
                data[field_name] = value.value
            else:
                data[field_name] = value
        return data


class WorkItemQuery(BaseModel):
    """Query parameters for listing work items."""

    state_id: str | None = Field(None, description="Filter by state UUID")
    priority: WorkItemPriority | None = Field(None, description="Filter by priority")
    assignee_id: str | None = Field(None, description="Filter by assignee UUID")
    label_id: str | None = Field(None, description="Filter by label UUID")
    created_by: str | None = Field(None, description="Filter by creator UUID")
    per_page: int = Field(50, ge=1, le=100, description="Results per page")
    cursor: str | None = Field(None, description="Pagination cursor")
    order_by: str = Field("-created_at", description="Sort order")

    def to_params(self) -> dict[str, Any]:
        """Convert to query parameters."""
        params = {"per_page": self.per_page, "order_by": self.order_by}
        if self.state_id:
            params["state_id"] = self.state_id
        if self.priority:
            params["priority"] = (
                self.priority.value if isinstance(self.priority, Enum) else self.priority
            )
        if self.assignee_id:
            params["assignee_id"] = self.assignee_id
        if self.label_id:
            params["label_id"] = self.label_id
        if self.created_by:
            params["created_by"] = self.created_by
        if self.cursor:
            params["cursor"] = self.cursor
        return params


# =============================================================================
# Response Schemas
# =============================================================================


class User(BaseModel):
    """Plane user representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    email: str | None = None
    first_name: str | None = None
    last_name: str | None = None
    display_name: str | None = None
    avatar: str | None = None

    @property
    def full_name(self) -> str:
        """Get user's full name."""
        if self.first_name and self.last_name:
            return f"{self.first_name} {self.last_name}"
        return self.display_name or self.email or self.id


class State(BaseModel):
    """Work item state representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    color: str | None = None
    group: str | None = None  # backlog, unstarted, started, completed, cancelled
    sequence: float | None = None


class Label(BaseModel):
    """Work item label representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    color: str | None = None


class Project(BaseModel):
    """Plane project representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    identifier: str  # e.g., "PROJ" for PROJ-123
    description: str | None = None
    cover_image: str | None = None
    emoji: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None


class WorkItem(BaseModel):
    """Work item (issue/task) representation."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    sequence_id: int | None = None  # The number in PROJ-123
    project_id: str | None = None
    description_html: str | None = None
    priority: str | None = None
    state_id: str | None = None
    parent_id: str | None = None
    start_date: str | None = None
    target_date: str | None = None
    estimate_point: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    created_by: str | None = None

    # Expanded fields (when requested)
    state: State | None = None
    assignees: list[User] | None = None
    labels: list[Label] | None = None

    @property
    def identifier(self) -> str | None:
        """Get human-readable identifier like PROJ-123."""
        if self.sequence_id is not None:
            return f"{self.sequence_id}"
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return self.model_dump(exclude_none=True)


class WorkItemList(BaseModel):
    """Paginated list of work items."""

    model_config = ConfigDict(extra="ignore")

    count: int = 0
    total_pages: int = 0
    total_results: int = 0
    next_cursor: str | None = None
    prev_cursor: str | None = None
    results: list[WorkItem] = Field(default_factory=list)

    @property
    def has_next(self) -> bool:
        """Check if there are more pages."""
        return self.next_cursor is not None


class ProjectList(BaseModel):
    """Paginated list of projects."""

    model_config = ConfigDict(extra="ignore")

    count: int = 0
    total_pages: int = 0
    total_results: int = 0
    next_cursor: str | None = None
    prev_cursor: str | None = None
    results: list[Project] = Field(default_factory=list)
