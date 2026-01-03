"""
Plane Integration Processors for Knomly.

These processors implement task management operations using Plane as the backend.
They accept GENERIC TaskFrame and return GENERIC TaskResultFrame.

Design Principle:
    Frames describe WHAT (the data/intent)
    Processors handle WHERE (the platform) and HOW (the mapping)

This allows:
    - Swap PlaneProcessor for LinearProcessor without changing upstream code
    - Consistent interface across all task management integrations
    - Domain-driven design (tasks as a concept, not platform-specific)

Usage:
    # Same TaskFrame works with any processor
    task = TaskFrame(
        name="Fix login bug",
        priority="high",
        project="backend",
    )

    # PlaneProcessor maps to Plane's API
    plane_result = await PlaneProcessor(plane_client).process(task, ctx)

    # LinearProcessor would map to Linear's API
    linear_result = await LinearProcessor(linear_client).process(task, ctx)

    # Both return TaskResultFrame with consistent structure

Priority Mapping (Generic → Plane):
    urgent → urgent
    high → high
    medium → medium
    low → low
    none → none

Status Mapping (Generic → Plane):
    backlog → backlog state
    todo → unstarted state
    in_progress → started state
    done → completed state
    cancelled → cancelled state
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from knomly.pipeline.frames.task import (
    TaskData,
    TaskFrame,
    TaskOperation,
    TaskPriority,
    TaskQueryFrame,
    TaskResultFrame,
    TaskStatus,
)
from knomly.pipeline.processor import Processor

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knomly.integrations.plane import PlaneClient
    from knomly.pipeline.context import PipelineContext
    from knomly.pipeline.frames.base import Frame

logger = logging.getLogger(__name__)


# =============================================================================
# Mapping Configuration
# =============================================================================


class PlaneMappings:
    """
    Mappings between generic task concepts and Plane-specific values.

    These can be customized per-workspace or made configurable.
    """

    # Priority mapping: Generic → Plane
    PRIORITY_TO_PLANE = {
        TaskPriority.URGENT: "urgent",
        TaskPriority.HIGH: "high",
        TaskPriority.MEDIUM: "medium",
        TaskPriority.LOW: "low",
        TaskPriority.NONE: "none",
    }

    # Priority mapping: Plane → Generic
    PRIORITY_FROM_PLANE = {
        "urgent": TaskPriority.URGENT,
        "high": TaskPriority.HIGH,
        "medium": TaskPriority.MEDIUM,
        "low": TaskPriority.LOW,
        "none": TaskPriority.NONE,
    }

    @classmethod
    def priority_to_plane(cls, priority: TaskPriority | str) -> str:
        """Convert generic priority to Plane priority."""
        if isinstance(priority, str):
            priority = TaskPriority.from_string(priority)
        return cls.PRIORITY_TO_PLANE.get(priority, "none")

    @classmethod
    def priority_from_plane(cls, priority: str | None) -> TaskPriority:
        """Convert Plane priority to generic priority."""
        if not priority:
            return TaskPriority.NONE
        return cls.PRIORITY_FROM_PLANE.get(priority.lower(), TaskPriority.NONE)


# =============================================================================
# Plane Processor
# =============================================================================


class PlaneProcessor(Processor):
    """
    Processes generic task frames using Plane as the backend.

    Accepts:
    - TaskFrame: Create, update, delete, or get a task
    - TaskQueryFrame: List/search tasks

    Returns:
    - TaskResultFrame: Generic result with task data

    The processor handles:
    - Mapping generic fields to Plane's API
    - Resolving project names to IDs
    - Converting Plane responses to generic format

    Configuration:
        - default_project_id: Default Plane project for tasks without project
        - project_mapping: Map project names to Plane project IDs
        - state_mapping: Map generic status to Plane state IDs
    """

    def __init__(
        self,
        client: PlaneClient,
        *,
        default_project_id: str = "",
        project_mapping: dict[str, str] | None = None,
        state_mapping: dict[TaskStatus, str] | None = None,
    ):
        """
        Initialize Plane processor.

        Args:
            client: Configured PlaneClient instance
            default_project_id: Default project for tasks without project
            project_mapping: Map project names/keys to Plane project IDs
            state_mapping: Map generic TaskStatus to Plane state UUIDs
        """
        self._client = client
        self._default_project_id = default_project_id
        self._project_mapping = project_mapping or {}
        self._state_mapping = state_mapping or {}

    @property
    def name(self) -> str:
        return "plane"

    def _resolve_project_id(self, project: str) -> str:
        """
        Resolve a project name/key to Plane project ID.

        Args:
            project: Project name, key, or ID

        Returns:
            Plane project ID
        """
        # Check mapping first
        if project in self._project_mapping:
            return self._project_mapping[project]

        # If it looks like a UUID, use directly
        if len(project) == 36 and "-" in project:
            return project

        # Fall back to default
        if self._default_project_id:
            return self._default_project_id

        # Use project as-is (might be an ID)
        return project

    def _resolve_state_id(self, status: TaskStatus | str) -> str | None:
        """
        Resolve a generic status to Plane state ID.

        Args:
            status: Generic task status

        Returns:
            Plane state ID or None if not mapped
        """
        if isinstance(status, str):
            status = TaskStatus.from_string(status)

        return self._state_mapping.get(status)

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | Sequence[Frame] | None:
        """
        Process a task frame using Plane.

        Args:
            frame: TaskFrame or TaskQueryFrame
            ctx: Pipeline context

        Returns:
            TaskResultFrame with operation result
        """
        if isinstance(frame, TaskQueryFrame):
            return await self._handle_query(frame, ctx)
        elif isinstance(frame, TaskFrame):
            return await self._handle_task(frame, ctx)
        else:
            # Pass through non-matching frames
            return frame

    async def _handle_task(
        self,
        frame: TaskFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Handle task create/update/delete operations."""
        operation = frame.operation

        if operation == TaskOperation.CREATE:
            return await self._create_task(frame, ctx)
        elif operation == TaskOperation.UPDATE:
            return await self._update_task(frame, ctx)
        elif operation == TaskOperation.DELETE:
            return await self._delete_task(frame, ctx)
        elif operation == TaskOperation.GET:
            return await self._get_task(frame, ctx)
        else:
            return TaskResultFrame(
                success=False,
                operation=operation,
                platform="plane",
                error_message=f"Unsupported operation: {operation}",
                source_frame_id=frame.id,
            )

    async def _create_task(
        self,
        frame: TaskFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Create a new task in Plane."""
        project_id = self._resolve_project_id(frame.project)

        if not project_id:
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.CREATE,
                platform="plane",
                error_message="No project specified and no default project configured",
                source_frame_id=frame.id,
            )

        logger.info(f"[plane] Creating task: {frame.name} in project {project_id}")

        try:
            work_item = await self._client.create_work_item(
                project_id=project_id,
                name=frame.name,
                description=frame.description or None,
                priority=PlaneMappings.priority_to_plane(frame.priority),
                state_id=self._resolve_state_id(frame.status),
                assignees=list(frame.assignees) if frame.assignees else None,
                labels=list(frame.labels) if frame.labels else None,
                parent_id=frame.parent_task_id or None,
                start_date=frame.start_date or None,
                target_date=frame.due_date or None,
                estimate_point=frame.estimate_points,
            )

            logger.info(f"[plane] Created task: {work_item.id} - {work_item.name}")

            return TaskResultFrame(
                success=True,
                operation=TaskOperation.CREATE,
                task_id=work_item.id,
                task_name=work_item.name,
                sequence_number=work_item.sequence_id,
                external_id=work_item.id,
                platform="plane",
                project_id=project_id,
                source_frame_id=frame.id,
            )

        except Exception as e:
            logger.error(f"[plane] Failed to create task: {e}")
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.CREATE,
                platform="plane",
                project_id=project_id,
                error_message=str(e),
                source_frame_id=frame.id,
            )

    async def _update_task(
        self,
        frame: TaskFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Update an existing task in Plane."""
        project_id = self._resolve_project_id(frame.project)

        if not frame.task_id:
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.UPDATE,
                platform="plane",
                error_message="task_id is required for update",
                source_frame_id=frame.id,
            )

        logger.info(f"[plane] Updating task: {frame.task_id}")

        try:
            work_item = await self._client.update_work_item(
                project_id=project_id,
                work_item_id=frame.task_id,
                name=frame.name or None,
                description=frame.description or None,
                priority=PlaneMappings.priority_to_plane(frame.priority)
                if frame.priority
                else None,
                state_id=self._resolve_state_id(frame.status) if frame.status else None,
                assignees=list(frame.assignees) if frame.assignees else None,
                labels=list(frame.labels) if frame.labels else None,
                start_date=frame.start_date or None,
                target_date=frame.due_date or None,
                estimate_point=frame.estimate_points,
            )

            logger.info(f"[plane] Updated task: {work_item.id}")

            return TaskResultFrame(
                success=True,
                operation=TaskOperation.UPDATE,
                task_id=work_item.id,
                task_name=work_item.name,
                sequence_number=work_item.sequence_id,
                external_id=work_item.id,
                platform="plane",
                project_id=project_id,
                source_frame_id=frame.id,
            )

        except Exception as e:
            logger.error(f"[plane] Failed to update task: {e}")
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.UPDATE,
                task_id=frame.task_id,
                platform="plane",
                error_message=str(e),
                source_frame_id=frame.id,
            )

    async def _delete_task(
        self,
        frame: TaskFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Delete a task from Plane."""
        project_id = self._resolve_project_id(frame.project)

        if not frame.task_id:
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.DELETE,
                platform="plane",
                error_message="task_id is required for delete",
                source_frame_id=frame.id,
            )

        logger.info(f"[plane] Deleting task: {frame.task_id}")

        try:
            await self._client.delete_work_item(
                project_id=project_id,
                work_item_id=frame.task_id,
            )

            logger.info(f"[plane] Deleted task: {frame.task_id}")

            return TaskResultFrame(
                success=True,
                operation=TaskOperation.DELETE,
                task_id=frame.task_id,
                platform="plane",
                project_id=project_id,
                source_frame_id=frame.id,
            )

        except Exception as e:
            logger.error(f"[plane] Failed to delete task: {e}")
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.DELETE,
                task_id=frame.task_id,
                platform="plane",
                error_message=str(e),
                source_frame_id=frame.id,
            )

    async def _get_task(
        self,
        frame: TaskFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Get a single task from Plane."""
        project_id = self._resolve_project_id(frame.project)

        if not frame.task_id:
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.GET,
                platform="plane",
                error_message="task_id is required for get",
                source_frame_id=frame.id,
            )

        try:
            work_item = await self._client.get_work_item(
                project_id=project_id,
                work_item_id=frame.task_id,
                expand=["assignees", "labels", "state"],
            )

            task_data = TaskData(
                task_id=work_item.id,
                name=work_item.name,
                description=work_item.description_html or "",
                priority=work_item.priority or "",
                status=work_item.state.name if work_item.state else "",
                assignees=tuple(a.email or a.id for a in (work_item.assignees or [])),
                labels=tuple(label.name for label in (work_item.labels or [])),
                project=project_id,
                due_date=work_item.target_date or "",
                created_at=work_item.created_at.isoformat() if work_item.created_at else "",
                updated_at=work_item.updated_at.isoformat() if work_item.updated_at else "",
                sequence_number=work_item.sequence_id,
                platform="plane",
            )

            return TaskResultFrame(
                success=True,
                operation=TaskOperation.GET,
                task_id=work_item.id,
                task_name=work_item.name,
                sequence_number=work_item.sequence_id,
                platform="plane",
                project_id=project_id,
                tasks=(task_data,),
                source_frame_id=frame.id,
            )

        except Exception as e:
            logger.error(f"[plane] Failed to get task: {e}")
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.GET,
                task_id=frame.task_id,
                platform="plane",
                error_message=str(e),
                source_frame_id=frame.id,
            )

    async def _handle_query(
        self,
        frame: TaskQueryFrame,
        ctx: PipelineContext,
    ) -> TaskResultFrame:
        """Handle task list/search operations."""
        project_id = self._resolve_project_id(frame.project)

        if not project_id:
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.LIST,
                platform="plane",
                error_message="Project is required for listing tasks",
                source_frame_id=frame.id,
            )

        logger.info(f"[plane] Listing tasks in project: {project_id}")

        try:
            # Map generic filters to Plane parameters
            result = await self._client.list_work_items(
                project_id=project_id,
                state_id=self._resolve_state_id(frame.status) if frame.status else None,
                priority=PlaneMappings.priority_to_plane(frame.priority)
                if frame.priority
                else None,
                assignee_id=frame.assignee or None,
                label_id=frame.label or None,
                per_page=frame.limit,
                cursor=frame.cursor or None,
                order_by=frame.order_by,
            )

            # Convert to generic TaskData
            tasks = tuple(
                TaskData(
                    task_id=item.id,
                    name=item.name,
                    description=item.description_html or "",
                    priority=item.priority or "",
                    status=item.state.name if item.state else "",
                    created_at=item.created_at.isoformat() if item.created_at else "",
                    updated_at=item.updated_at.isoformat() if item.updated_at else "",
                    sequence_number=item.sequence_id,
                    platform="plane",
                    project=project_id,
                )
                for item in result.results
            )

            logger.info(f"[plane] Listed {len(tasks)} tasks")

            return TaskResultFrame(
                success=True,
                operation=TaskOperation.LIST,
                platform="plane",
                project_id=project_id,
                tasks=tasks,
                total_count=result.total_results,
                has_more=result.has_next,
                next_cursor=result.next_cursor or "",
                source_frame_id=frame.id,
            )

        except Exception as e:
            logger.error(f"[plane] Failed to list tasks: {e}")
            return TaskResultFrame(
                success=False,
                operation=TaskOperation.LIST,
                platform="plane",
                project_id=project_id,
                error_message=str(e),
                source_frame_id=frame.id,
            )


# =============================================================================
# Transformation Processors
# =============================================================================


class TaskCreatorProcessor(Processor):
    """
    Transforms ExtractionFrame into TaskFrames.

    This is a GENERIC processor that creates platform-agnostic TaskFrames.
    The downstream processor (PlaneProcessor, LinearProcessor, etc.)
    handles the platform-specific execution.

    Input: ExtractionFrame (from voice/text extraction)
    Output: Sequence[TaskFrame] (one per task)

    Usage:
        pipeline = (
            PipelineBuilder()
            .add(TranscriptionProcessor())
            .add(StandupExtractionProcessor())
            .add(TaskCreatorProcessor(default_project="backend"))
            .add(PlaneProcessor(client))  # Or LinearProcessor, etc.
            .build()
        )
    """

    def __init__(
        self,
        default_project: str = "",
        *,
        default_priority: TaskPriority = TaskPriority.MEDIUM,
    ):
        """
        Initialize task creator.

        Args:
            default_project: Default project for tasks
            default_priority: Default priority for tasks
        """
        self._default_project = default_project
        self._default_priority = default_priority

    @property
    def name(self) -> str:
        return "task_creator"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | Sequence[Frame] | None:
        """
        Convert extraction results to generic task frames.

        Args:
            frame: ExtractionFrame with extracted tasks
            ctx: Pipeline context

        Returns:
            Sequence of TaskFrame, or original frame if not applicable
        """
        # Import here to avoid circular imports
        from knomly.pipeline.frames.processing import ExtractionFrame

        if not isinstance(frame, ExtractionFrame):
            return frame

        tasks = frame.today_items

        if not tasks:
            logger.info("[task_creator] No tasks found in extraction")
            return frame

        logger.info(f"[task_creator] Creating {len(tasks)} task frames")

        # Create a TaskFrame for each task
        task_frames = []
        for task in tasks:
            # Detect priority from task text
            priority = self._detect_priority(task)

            task_frame = TaskFrame(
                operation=TaskOperation.CREATE,
                name=task,
                priority=priority,
                project=self._default_project,
                source_frame_id=frame.id,
                metadata={
                    **frame.metadata,
                    "source": "voice_extraction",
                    "original_extraction_id": str(frame.id),
                },
            )
            task_frames.append(task_frame)

        return task_frames

    def _detect_priority(self, task: str) -> TaskPriority:
        """Detect priority from task text."""
        task_lower = task.lower()

        if "urgent" in task_lower or "asap" in task_lower or "critical" in task_lower:
            return TaskPriority.URGENT
        elif "high priority" in task_lower or "important" in task_lower:
            return TaskPriority.HIGH
        elif (
            "low priority" in task_lower
            or "when possible" in task_lower
            or "nice to have" in task_lower
        ):
            return TaskPriority.LOW

        return self._default_priority
