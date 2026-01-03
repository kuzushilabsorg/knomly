"""
Tests for Plane integration.

Tests cover:
- PlaneClient API calls
- Generic Task Frame types
- PlaneProcessor with generic frames
- Error handling
- Integration with pipeline
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime
from uuid import uuid4

from knomly.integrations.plane import (
    PlaneClient,
    PlaneConfig,
    WorkItem,
    WorkItemCreate,
    WorkItemPriority,
)
from knomly.integrations.base import (
    IntegrationError,
    AuthenticationError,
    RateLimitError,
    NotFoundError,
)
from knomly.pipeline.frames.task import (
    TaskFrame,
    TaskQueryFrame,
    TaskResultFrame,
    TaskData,
    TaskPriority,
    TaskStatus,
    TaskOperation,
    TaskActionFrame,
)
from knomly.pipeline.processors.integrations.plane import (
    PlaneProcessor,
    TaskCreatorProcessor,
    PlaneMappings,
)
from knomly.pipeline.context import PipelineContext
from knomly.pipeline.frames.processing import ExtractionFrame


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def plane_config():
    """Create test Plane configuration."""
    return PlaneConfig(
        api_key="plane_api_test_key",
        workspace_slug="test-workspace",
        base_url="https://api.plane.so",
    )


@pytest.fixture
def mock_plane_client(plane_config):
    """Create mock Plane client."""
    client = PlaneClient(plane_config)
    return client


@pytest.fixture
def mock_work_item():
    """Create mock work item response."""
    return WorkItem(
        id="work-item-123",
        name="Test work item",
        sequence_id=42,
        project_id="project-456",
        priority="high",
        state_id="state-789",
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )


@pytest.fixture
def pipeline_context():
    """Create test pipeline context."""
    return PipelineContext()


# =============================================================================
# PlaneConfig Tests
# =============================================================================


class TestPlaneConfig:
    """Tests for PlaneConfig."""

    def test_config_creation(self):
        """Test valid config creation."""
        config = PlaneConfig(
            api_key="plane_api_xxx",
            workspace_slug="my-workspace",
        )
        assert config.api_key == "plane_api_xxx"
        assert config.workspace_slug == "my-workspace"
        assert config.base_url == "https://api.plane.so"

    def test_config_requires_api_key(self):
        """Test that API key is required."""
        with pytest.raises(ValueError, match="API key is required"):
            PlaneConfig(
                api_key="",
                workspace_slug="my-workspace",
            )

    def test_config_requires_workspace(self):
        """Test that workspace slug is required."""
        with pytest.raises(ValueError, match="workspace slug is required"):
            PlaneConfig(
                api_key="plane_api_xxx",
                workspace_slug="",
            )

    def test_config_custom_base_url(self):
        """Test custom base URL for self-hosted."""
        config = PlaneConfig(
            api_key="plane_api_xxx",
            workspace_slug="my-workspace",
            base_url="https://plane.mycompany.com",
        )
        assert config.base_url == "https://plane.mycompany.com"


# =============================================================================
# Generic Task Frame Tests
# =============================================================================


class TestTaskFrames:
    """Tests for generic task frame types."""

    def test_task_frame_creation(self):
        """Test TaskFrame creation."""
        frame = TaskFrame(
            name="Fix bug",
            description="Fix the login bug",
            priority=TaskPriority.HIGH,
            project="backend",
        )
        assert frame.name == "Fix bug"
        assert frame.priority == TaskPriority.HIGH
        assert frame.frame_type == "TaskFrame"
        assert frame.operation == TaskOperation.CREATE

    def test_task_frame_immutable(self):
        """Test frame immutability."""
        frame = TaskFrame(name="Test")
        with pytest.raises(AttributeError):
            frame.name = "Changed"

    def test_task_frame_string_priority(self):
        """Test TaskFrame with string priority."""
        frame = TaskFrame(
            name="Test",
            priority="high",
        )
        assert frame.normalized_priority == TaskPriority.HIGH

    def test_task_frame_for_update(self):
        """Test creating update frame."""
        original = TaskFrame(
            name="Original",
            project="backend",
        )
        update = original.for_update("task-123", name="Updated")
        assert update.operation == TaskOperation.UPDATE
        assert update.task_id == "task-123"
        assert update.name == "Updated"

    def test_task_query_frame(self):
        """Test TaskQueryFrame."""
        frame = TaskQueryFrame(
            project="backend",
            status=TaskStatus.IN_PROGRESS,
            limit=25,
        )
        assert frame.project == "backend"
        assert frame.limit == 25

    def test_task_result_frame_success(self):
        """Test TaskResultFrame for success."""
        frame = TaskResultFrame(
            success=True,
            operation=TaskOperation.CREATE,
            task_id="task-123",
            task_name="Fix bug",
            sequence_number=42,
            platform="plane",
        )
        assert frame.success
        assert frame.identifier == "#42"

    def test_task_result_frame_failure(self):
        """Test TaskResultFrame for failure."""
        frame = TaskResultFrame(
            success=False,
            operation=TaskOperation.CREATE,
            platform="plane",
            error_message="API error",
        )
        assert not frame.success
        assert frame.error_message == "API error"

    def test_task_action_frame_from_result(self):
        """Test TaskActionFrame creation from result."""
        result = TaskResultFrame(
            success=True,
            operation=TaskOperation.CREATE,
            task_id="task-123",
            task_name="Fix bug",
            platform="plane",
        )
        action = TaskActionFrame.from_result(result)
        assert action.action == "created"
        assert action.success
        assert "Fix bug" in action.summary


# =============================================================================
# Priority/Status Mapping Tests
# =============================================================================


class TestPlaneMappings:
    """Tests for Plane-specific mappings."""

    def test_priority_to_plane(self):
        """Test generic to Plane priority mapping."""
        assert PlaneMappings.priority_to_plane(TaskPriority.URGENT) == "urgent"
        assert PlaneMappings.priority_to_plane(TaskPriority.HIGH) == "high"
        assert PlaneMappings.priority_to_plane("medium") == "medium"

    def test_priority_from_plane(self):
        """Test Plane to generic priority mapping."""
        assert PlaneMappings.priority_from_plane("urgent") == TaskPriority.URGENT
        assert PlaneMappings.priority_from_plane("HIGH") == TaskPriority.HIGH
        assert PlaneMappings.priority_from_plane(None) == TaskPriority.NONE


# =============================================================================
# PlaneClient Tests
# =============================================================================


class TestPlaneClient:
    """Tests for PlaneClient."""

    def test_client_name(self, mock_plane_client):
        """Test client name property."""
        assert mock_plane_client.name == "plane"

    def test_client_workspace_slug(self, mock_plane_client):
        """Test workspace slug property."""
        assert mock_plane_client.workspace_slug == "test-workspace"

    def test_auth_headers(self, mock_plane_client):
        """Test authentication headers."""
        headers = mock_plane_client._get_auth_headers()
        assert headers["X-API-Key"] == "plane_api_test_key"

    @pytest.mark.asyncio
    async def test_create_work_item(self, mock_plane_client, mock_work_item):
        """Test work item creation."""
        with patch.object(
            mock_plane_client,
            "_request",
            new_callable=AsyncMock,
        ) as mock_request:
            mock_request.return_value = MagicMock(
                json=lambda: mock_work_item.model_dump()
            )

            result = await mock_plane_client.create_work_item(
                project_id="project-456",
                name="Test work item",
                priority="high",
            )

            assert result.id == "work-item-123"
            assert result.name == "Test work item"
            mock_request.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_work_items(self, mock_plane_client, mock_work_item):
        """Test work item listing."""
        with patch.object(
            mock_plane_client,
            "_request",
            new_callable=AsyncMock,
        ) as mock_request:
            mock_request.return_value = MagicMock(
                json=lambda: {
                    "count": 1,
                    "total_pages": 1,
                    "total_results": 1,
                    "results": [mock_work_item.model_dump()],
                }
            )

            result = await mock_plane_client.list_work_items(
                project_id="project-456",
            )

            assert len(result.results) == 1
            assert result.total_results == 1


# =============================================================================
# PlaneProcessor Tests (with Generic Frames)
# =============================================================================


class TestPlaneProcessor:
    """Tests for PlaneProcessor with generic TaskFrame."""

    @pytest.mark.asyncio
    async def test_creates_task_from_generic_frame(self, mock_plane_client, mock_work_item, pipeline_context):
        """Test creating task from generic TaskFrame."""
        with patch.object(
            mock_plane_client,
            "create_work_item",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_work_item

            processor = PlaneProcessor(
                mock_plane_client,
                default_project_id="project-456",
            )

            # Use GENERIC TaskFrame, not Plane-specific
            frame = TaskFrame(
                name="Fix login bug",
                priority=TaskPriority.HIGH,
                description="Users cannot log in",
            )

            result = await processor.process(frame, pipeline_context)

            # Get GENERIC TaskResultFrame back
            assert isinstance(result, TaskResultFrame)
            assert result.success
            assert result.task_id == "work-item-123"
            assert result.platform == "plane"
            assert result.source_frame_id == frame.id

    @pytest.mark.asyncio
    async def test_lists_tasks_from_generic_query(self, mock_plane_client, mock_work_item, pipeline_context):
        """Test listing tasks from generic TaskQueryFrame."""
        from knomly.integrations.plane.schemas import WorkItemList

        with patch.object(
            mock_plane_client,
            "list_work_items",
            new_callable=AsyncMock,
        ) as mock_list:
            mock_list.return_value = WorkItemList(
                count=1,
                total_results=1,
                results=[mock_work_item],
            )

            processor = PlaneProcessor(
                mock_plane_client,
                default_project_id="project-456",
            )

            # Use GENERIC TaskQueryFrame
            frame = TaskQueryFrame(
                project="project-456",
                status=TaskStatus.IN_PROGRESS,
            )

            result = await processor.process(frame, pipeline_context)

            assert isinstance(result, TaskResultFrame)
            assert result.success
            assert len(result.tasks) == 1
            assert result.platform == "plane"

    @pytest.mark.asyncio
    async def test_handles_api_error(self, mock_plane_client, pipeline_context):
        """Test error handling."""
        with patch.object(
            mock_plane_client,
            "create_work_item",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.side_effect = IntegrationError("API error", "plane")

            processor = PlaneProcessor(
                mock_plane_client,
                default_project_id="project-456",
            )

            frame = TaskFrame(name="Test task")

            result = await processor.process(frame, pipeline_context)

            assert isinstance(result, TaskResultFrame)
            assert not result.success
            assert "API error" in result.error_message

    @pytest.mark.asyncio
    async def test_passes_through_non_matching_frames(self, mock_plane_client, pipeline_context):
        """Test that non-matching frames pass through."""
        processor = PlaneProcessor(mock_plane_client)

        # Pass a non-task frame
        from knomly.pipeline.frames.base import Frame
        frame = Frame()

        result = await processor.process(frame, pipeline_context)
        assert result is frame

    @pytest.mark.asyncio
    async def test_project_mapping(self, mock_plane_client, mock_work_item, pipeline_context):
        """Test project name to ID mapping."""
        with patch.object(
            mock_plane_client,
            "create_work_item",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_work_item

            processor = PlaneProcessor(
                mock_plane_client,
                project_mapping={
                    "backend": "project-456",
                    "frontend": "project-789",
                },
            )

            # Use project NAME, not ID
            frame = TaskFrame(
                name="Fix bug",
                project="backend",  # Name, not UUID
            )

            await processor.process(frame, pipeline_context)

            # Verify it was mapped to the correct project ID
            mock_create.assert_called_once()
            call_kwargs = mock_create.call_args.kwargs
            assert call_kwargs["project_id"] == "project-456"


# =============================================================================
# TaskCreatorProcessor Tests
# =============================================================================


class TestTaskCreatorProcessor:
    """Tests for TaskCreatorProcessor."""

    @pytest.mark.asyncio
    async def test_creates_generic_task_frames_from_extraction(self, pipeline_context):
        """Test conversion of extraction to GENERIC task frames."""
        processor = TaskCreatorProcessor(default_project="backend")

        extraction = ExtractionFrame(
            today_items=("Fix login bug", "Review PR"),
            blockers=(),
            summary="Working on auth",
        )

        result = await processor.process(extraction, pipeline_context)

        # Should produce GENERIC TaskFrames
        assert isinstance(result, list)
        assert len(result) == 2
        assert all(isinstance(f, TaskFrame) for f in result)
        assert result[0].name == "Fix login bug"
        assert result[0].project == "backend"
        assert result[0].operation == TaskOperation.CREATE

    @pytest.mark.asyncio
    async def test_detects_priority_hints(self, pipeline_context):
        """Test priority detection from task text."""
        processor = TaskCreatorProcessor(default_project="test")

        extraction = ExtractionFrame(
            today_items=(
                "URGENT: Fix production bug",
                "Low priority: Update docs",
                "Regular task",
            ),
            blockers=(),
            summary="",
        )

        result = await processor.process(extraction, pipeline_context)

        assert result[0].priority == TaskPriority.URGENT
        assert result[1].priority == TaskPriority.LOW
        assert result[2].priority == TaskPriority.MEDIUM  # default


# =============================================================================
# Integration Tests
# =============================================================================


class TestPlaneIntegration:
    """Integration tests for Plane in pipeline context."""

    @pytest.mark.asyncio
    async def test_full_pipeline_flow_with_generic_frames(self, mock_plane_client, mock_work_item, pipeline_context):
        """Test full pipeline from extraction to task creation using generic frames."""
        from knomly.pipeline import PipelineBuilder

        with patch.object(
            mock_plane_client,
            "create_work_item",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_work_item

            # Pipeline uses GENERIC processors and frames
            pipeline = (
                PipelineBuilder()
                .add(TaskCreatorProcessor(default_project="backend"))
                .add(PlaneProcessor(mock_plane_client, default_project_id="project-456"))
                .build()
            )

            extraction = ExtractionFrame(
                today_items=("Test task",),
                blockers=(),
                summary="Test",
            )

            result = await pipeline.execute(extraction, pipeline_context)

            assert result.success
            assert len(result.output_frames) == 1
            # Output is GENERIC TaskResultFrame
            assert isinstance(result.output_frames[0], TaskResultFrame)
            assert result.output_frames[0].platform == "plane"

    @pytest.mark.asyncio
    async def test_processor_swappability(self, mock_plane_client, mock_work_item, pipeline_context):
        """Test that generic frames allow processor swapping."""
        # This test demonstrates that the same TaskFrame can work with different processors
        with patch.object(
            mock_plane_client,
            "create_work_item",
            new_callable=AsyncMock,
        ) as mock_create:
            mock_create.return_value = mock_work_item

            # Generic TaskFrame - platform agnostic
            task = TaskFrame(
                name="Fix bug",
                priority=TaskPriority.HIGH,
                project="backend",
            )

            # Process with PlaneProcessor
            plane_processor = PlaneProcessor(
                mock_plane_client,
                default_project_id="project-456",
            )
            plane_result = await plane_processor.process(task, pipeline_context)

            # Result is generic but knows which platform handled it
            assert isinstance(plane_result, TaskResultFrame)
            assert plane_result.platform == "plane"

            # The same TaskFrame could be processed by LinearProcessor, JiraProcessor, etc.
            # They would all return TaskResultFrame with their respective platform name
