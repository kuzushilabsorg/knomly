"""
Task Management Frame types.

These frames represent generic task/issue/work-item operations that can be
processed by any task management integration (Plane, Linear, Jira, Asana, etc.).

Design Principle:
    Frames describe WHAT (the data/intent), not WHERE (the platform).
    Processors handle the platform-specific mapping.

Frame Flow:
    TaskFrame → PlaneProcessor/LinearProcessor/JiraProcessor → TaskResultFrame

This allows:
    - Swapping task platforms without changing upstream code
    - Consistent interface across integrations
    - Domain-driven design (task management as a concept)

Usage:
    # Create task in any platform
    pipeline = (
        PipelineBuilder()
        .add(ExtractionProcessor())     # Produces ExtractionFrame
        .add(TaskCreatorProcessor())    # Converts to TaskFrame
        .add(PlaneProcessor())          # Or LinearProcessor, JiraProcessor
        .build()
    )

    # The TaskFrame is platform-agnostic
    task = TaskFrame(
        name="Fix login bug",
        priority="high",
        project="backend",
    )

    # PlaneProcessor maps it to Plane's API
    # LinearProcessor maps it to Linear's API
    # Same frame, different destination
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

from knomly.pipeline.frames.base import Frame

if TYPE_CHECKING:
    from uuid import UUID

# =============================================================================
# Enums
# =============================================================================


class TaskPriority(str, Enum):
    """Generic task priority levels."""

    URGENT = "urgent"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    NONE = "none"

    @classmethod
    def from_string(cls, value: str) -> TaskPriority:
        """Convert string to priority, with fallback."""
        if not value:
            return cls.NONE
        try:
            return cls(value.lower())
        except ValueError:
            return cls.NONE


class TaskStatus(str, Enum):
    """Generic task status levels."""

    BACKLOG = "backlog"
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    IN_REVIEW = "in_review"
    DONE = "done"
    CANCELLED = "cancelled"

    @classmethod
    def from_string(cls, value: str) -> TaskStatus:
        """Convert string to status, with fallback."""
        if not value:
            return cls.TODO
        try:
            return cls(value.lower().replace(" ", "_").replace("-", "_"))
        except ValueError:
            return cls.TODO


class TaskOperation(str, Enum):
    """Task operation types."""

    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    LIST = "list"
    GET = "get"


# =============================================================================
# Request Frames
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskFrame(Frame):
    """
    Generic task/issue/work-item representation.

    This frame can be processed by any task management integration:
    - Plane (work items)
    - Linear (issues)
    - Jira (issues)
    - Asana (tasks)
    - GitHub Issues
    - etc.

    The processor handles platform-specific mapping.
    """

    # Operation
    operation: TaskOperation = TaskOperation.CREATE

    # Core fields
    name: str = ""
    description: str = ""

    # Classification
    priority: TaskPriority | str = TaskPriority.NONE
    status: TaskStatus | str = TaskStatus.TODO

    # Assignment
    assignees: tuple[str, ...] = ()  # User identifiers (email, username, ID)
    labels: tuple[str, ...] = ()

    # Hierarchy
    project: str = ""  # Project name/key/ID
    parent_task_id: str = ""  # For sub-tasks

    # Dates
    start_date: str = ""  # YYYY-MM-DD
    due_date: str = ""  # YYYY-MM-DD

    # Estimation
    estimate_points: int | None = None
    estimate_hours: float | None = None

    # For updates
    task_id: str = ""  # Existing task ID (for update/delete operations)

    # Platform hints (optional - processor may ignore)
    platform_config: dict[str, Any] = field(default_factory=dict)

    @property
    def frame_type(self) -> str:
        return "TaskFrame"

    @property
    def normalized_priority(self) -> TaskPriority:
        """Get priority as enum."""
        if isinstance(self.priority, TaskPriority):
            return self.priority
        return TaskPriority.from_string(str(self.priority))

    @property
    def normalized_status(self) -> TaskStatus:
        """Get status as enum."""
        if isinstance(self.status, TaskStatus):
            return self.status
        return TaskStatus.from_string(str(self.status))

    def for_update(
        self,
        task_id: str,
        **updates,
    ) -> TaskFrame:
        """Create an update frame for an existing task."""
        return TaskFrame(
            operation=TaskOperation.UPDATE,
            task_id=task_id,
            name=updates.get("name", self.name),
            description=updates.get("description", self.description),
            priority=updates.get("priority", self.priority),
            status=updates.get("status", self.status),
            assignees=updates.get("assignees", self.assignees),
            labels=updates.get("labels", self.labels),
            project=updates.get("project", self.project),
            due_date=updates.get("due_date", self.due_date),
            source_frame_id=self.id,
            metadata=self.metadata,
        )


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskQueryFrame(Frame):
    """
    Query for listing/searching tasks.

    This frame requests a list of tasks matching certain criteria.
    Platform-specific filtering is handled by the processor.
    """

    # Project scope
    project: str = ""  # Empty means all projects

    # Filters
    status: TaskStatus | str = ""
    priority: TaskPriority | str = ""
    assignee: str = ""  # User identifier
    label: str = ""

    # Search
    search_text: str = ""

    # Date filters
    created_after: str = ""  # YYYY-MM-DD
    created_before: str = ""
    due_before: str = ""
    due_after: str = ""

    # Pagination
    limit: int = 50
    offset: int = 0
    cursor: str = ""  # For cursor-based pagination

    # Sorting
    order_by: str = "-created_at"  # Field name, prefix with - for descending

    @property
    def frame_type(self) -> str:
        return "TaskQueryFrame"


# =============================================================================
# Response Frames
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskResultFrame(Frame):
    """
    Result of a task operation.

    Returned by any task management processor after create/update/delete.
    """

    # Operation that was performed
    operation: TaskOperation = TaskOperation.CREATE

    # Success indicator
    success: bool = True

    # Created/updated task info
    task_id: str = ""
    task_name: str = ""
    task_url: str = ""  # Link to task in platform UI

    # Platform-specific identifiers
    sequence_number: int | None = None  # e.g., PROJ-123 → 123
    external_id: str = ""  # Platform's native ID

    # What platform handled it
    platform: str = ""  # "plane", "linear", "jira", etc.
    project_id: str = ""

    # For list operations
    tasks: tuple[TaskData, ...] = ()
    total_count: int = 0
    has_more: bool = False
    next_cursor: str = ""

    # Original request reference
    source_frame_id: UUID | None = None

    # Error info
    error_message: str = ""
    error_code: str = ""

    @property
    def frame_type(self) -> str:
        return "TaskResultFrame"

    @property
    def identifier(self) -> str:
        """Human-readable identifier."""
        if self.sequence_number:
            return f"#{self.sequence_number}"
        if self.external_id:
            return self.external_id[:8]
        return self.task_id[:8] if self.task_id else ""


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskData(Frame):
    """
    Immutable representation of a task item.

    Used in TaskResultFrame for list operations.
    """

    task_id: str
    name: str
    description: str = ""
    priority: str = ""
    status: str = ""
    assignees: tuple[str, ...] = ()
    labels: tuple[str, ...] = ()
    project: str = ""
    due_date: str = ""
    created_at: str = ""
    updated_at: str = ""
    url: str = ""
    sequence_number: int | None = None
    platform: str = ""

    @property
    def frame_type(self) -> str:
        return "TaskData"


# =============================================================================
# Action Frames (Pipeline output)
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class TaskActionFrame(Frame):
    """
    High-level action result for task operations.

    Summarizes what was done across potentially multiple tasks.
    Useful for confirmations and audit logging.
    """

    # Action performed
    action: str  # "created", "updated", "deleted", "listed"

    # Success indicator
    success: bool = True

    # Summary
    platform: str = ""  # Which platform was used
    project: str = ""
    task_count: int = 0
    task_ids: tuple[str, ...] = ()
    task_names: tuple[str, ...] = ()

    # Human-readable summary
    summary: str = ""

    # Original request reference
    source_frame_id: UUID | None = None

    # Error info
    error_message: str = ""

    @property
    def frame_type(self) -> str:
        return "TaskActionFrame"

    @classmethod
    def from_result(cls, result: TaskResultFrame) -> TaskActionFrame:
        """Create action frame from task result."""
        if result.operation == TaskOperation.LIST:
            return cls(
                action="listed",
                success=result.success,
                platform=result.platform,
                project=result.project_id,
                task_count=len(result.tasks),
                task_ids=tuple(t.task_id for t in result.tasks),
                task_names=tuple(t.name for t in result.tasks),
                summary=f"Listed {len(result.tasks)} tasks from {result.platform}",
                source_frame_id=result.source_frame_id,
                error_message=result.error_message,
            )
        else:
            action = (
                result.operation.value
                if isinstance(result.operation, TaskOperation)
                else str(result.operation)
            )
            return cls(
                action=action + "d",  # create → created
                success=result.success,
                platform=result.platform,
                project=result.project_id,
                task_count=1 if result.success else 0,
                task_ids=(result.task_id,) if result.task_id else (),
                task_names=(result.task_name,) if result.task_name else (),
                summary=f"{action.capitalize()}d task: {result.task_name}"
                if result.success
                else f"Failed: {result.error_message}",
                source_frame_id=result.source_frame_id,
                error_message=result.error_message,
            )
