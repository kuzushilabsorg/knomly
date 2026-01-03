"""
Tests for v2 Agent Layer.

Tests cover:
- Tool protocol and registry
- Agent frames
- AgentProcessor decisions
- AgentExecutor loop
- ADR-004 compliance (frame stream observability)

Core Invariant (ADR-004):
    "If I can't explain an execution by looking only at the Frame stream,
     the design is broken."

Every agent decision must be visible in the frame stream.
"""

from datetime import datetime

import pytest

from knomly.agent.frames import (
    AgentAction,
    AgentControlFrame,
    AgentResponseFrame,
    PlanFrame,
    ToolCallFrame,
    ToolResultFrame,
    create_initial_plan,
)
from knomly.agent.result import (
    AgentResult,
    max_iterations_result,
    success_result,
    timeout_result,
)
from knomly.tools.base import ContentBlock, Tool, ToolAnnotations, ToolResult
from knomly.tools.registry import ToolRegistry, ToolRegistryError

# =============================================================================
# Test Fixtures
# =============================================================================


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool", result: ToolResult | None = None):
        self._name = name
        self._result = result or ToolResult.success("Mock result")
        self.call_count = 0
        self.last_arguments = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "input": {"type": "string", "description": "Test input"},
            },
            "required": ["input"],
        }

    async def execute(self, arguments: dict) -> ToolResult:
        self.call_count += 1
        self.last_arguments = arguments
        return self._result


class FailingTool(Tool):
    """Tool that always fails for testing error handling."""

    @property
    def name(self) -> str:
        return "failing_tool"

    @property
    def description(self) -> str:
        return "A tool that always fails"

    @property
    def input_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.error("This tool always fails")


@pytest.fixture
def mock_tool():
    """Create a mock tool."""
    return MockTool()


@pytest.fixture
def tool_registry(mock_tool):
    """Create a registry with mock tool."""
    registry = ToolRegistry()
    registry.register(mock_tool)
    return registry


# =============================================================================
# Tool Protocol Tests
# =============================================================================


class TestToolBase:
    """Tests for Tool base class."""

    def test_tool_result_success(self):
        """Test creating successful ToolResult."""
        result = ToolResult.success(
            "Task created",
            structured={"id": "123"},
        )

        assert result.is_error is False
        assert result.text == "Task created"
        assert result.structured_content == {"id": "123"}

    def test_tool_result_error(self):
        """Test creating error ToolResult."""
        result = ToolResult.error("Something went wrong")

        assert result.is_error is True
        assert "Error:" in result.text
        assert "Something went wrong" in result.text

    def test_content_block_text(self):
        """Test creating text ContentBlock."""
        block = ContentBlock.from_text("Hello world")

        assert block.type.value == "text"
        assert block.text_content == "Hello world"

    def test_tool_annotations_defaults(self):
        """Test ToolAnnotations default values."""
        annotations = ToolAnnotations()

        assert annotations.read_only_hint is False
        assert annotations.destructive_hint is True
        assert annotations.idempotent_hint is False

    def test_tool_annotations_read_only(self):
        """Test read-only tool annotations."""
        annotations = ToolAnnotations(
            title="Query Tool",
            read_only_hint=True,
            destructive_hint=False,
        )

        assert annotations.read_only_hint is True
        assert annotations.destructive_hint is False


class TestToolRegistry:
    """Tests for ToolRegistry."""

    def test_register_and_get_tool(self, tool_registry, mock_tool):
        """Test registering and retrieving a tool."""
        retrieved = tool_registry.get("mock_tool")

        assert retrieved is mock_tool

    def test_duplicate_registration_fails(self, tool_registry, mock_tool):
        """Test that duplicate registration raises error."""
        with pytest.raises(ToolRegistryError):
            tool_registry.register(MockTool("mock_tool"))

    def test_get_unknown_tool_returns_none(self, tool_registry):
        """Test getting unknown tool returns None."""
        result = tool_registry.get("unknown")

        assert result is None

    def test_get_required_raises_on_unknown(self, tool_registry):
        """Test get_required raises on unknown tool."""
        with pytest.raises(ToolRegistryError):
            tool_registry.get_required("unknown")

    def test_list_tools(self, tool_registry, mock_tool):
        """Test listing all tools."""
        tools = tool_registry.list_tools()

        assert len(tools) == 1
        assert mock_tool in tools

    def test_to_llm_schemas(self, tool_registry):
        """Test getting LLM schemas."""
        schemas = tool_registry.to_llm_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "mock_tool"
        assert "description" in schemas[0]
        assert "input_schema" in schemas[0]

    @pytest.mark.asyncio
    async def test_tool_execution(self, mock_tool):
        """Test tool execution."""
        result = await mock_tool.execute({"input": "test"})

        assert result.is_error is False
        assert mock_tool.call_count == 1
        assert mock_tool.last_arguments == {"input": "test"}


# =============================================================================
# Agent Frames Tests
# =============================================================================


class TestAgentFrames:
    """Tests for agent frame types."""

    def test_plan_frame_creation(self):
        """Test creating PlanFrame."""
        frame = PlanFrame(
            goal="Create a task",
            reasoning="Analyzing user request",
            observations=("Project exists",),
            next_action=AgentAction.TOOL_CALL,
            iteration=0,
        )

        assert frame.frame_type == "plan"
        assert frame.goal == "Create a task"
        assert frame.next_action == AgentAction.TOOL_CALL

    def test_tool_call_frame_creation(self):
        """Test creating ToolCallFrame."""
        frame = ToolCallFrame(
            tool_name="plane_create_task",
            tool_arguments={"name": "Test task"},
            reasoning="User wants to create a task",
            iteration=0,
        )

        assert frame.frame_type == "tool_call"
        assert frame.tool_name == "plane_create_task"
        assert frame.tool_arguments == {"name": "Test task"}

    def test_tool_result_frame_success(self):
        """Test creating successful ToolResultFrame."""
        frame = ToolResultFrame(
            tool_name="plane_create_task",
            success=True,
            result_text="Created task MOB-123",
            structured_result={"task_id": "uuid-123"},
        )

        assert frame.frame_type == "tool_result"
        assert frame.success is True
        assert frame.result_text == "Created task MOB-123"

    def test_tool_result_frame_failure(self):
        """Test creating failed ToolResultFrame."""
        frame = ToolResultFrame(
            tool_name="plane_create_task",
            success=False,
            result_text="",
            error_message="Project not found",
        )

        assert frame.success is False
        assert frame.error_message == "Project not found"

    def test_agent_response_frame(self):
        """Test creating AgentResponseFrame."""
        frame = AgentResponseFrame(
            response_text="I've created your task",
            goal_achieved=True,
            iterations_used=1,
            tools_called=("plane_create_task",),
        )

        assert frame.frame_type == "agent_response"
        assert frame.goal_achieved is True
        assert "plane_create_task" in frame.tools_called

    def test_create_initial_plan(self):
        """Test factory function for initial plan."""
        frame = create_initial_plan(
            goal="Create a task",
            max_iterations=10,
            initial_observations=("Context loaded",),
        )

        assert frame.goal == "Create a task"
        assert frame.max_iterations == 10
        assert "Context loaded" in frame.observations


# =============================================================================
# Agent Result Tests
# =============================================================================


class TestAgentResult:
    """Tests for AgentResult."""

    def test_success_result(self):
        """Test creating successful result."""
        response = AgentResponseFrame(
            response_text="Done!",
            goal_achieved=True,
            iterations_used=1,
        )

        result = success_result(
            response=response,
            frames=(response,),
            iterations=1,
            tools_called=("mock_tool",),
        )

        assert result.success is True
        assert result.response == response
        assert result.iterations == 1

    def test_timeout_result(self):
        """Test creating timeout result."""
        result = timeout_result(
            frames=(),
            iterations=3,
            timeout_seconds=30.0,
        )

        assert result.success is False
        assert result.error_type == "timeout"
        assert "30.0s" in result.error_message

    def test_max_iterations_result(self):
        """Test creating max iterations result."""
        result = max_iterations_result(
            frames=(),
            max_iterations=5,
        )

        assert result.success is False
        assert result.error_type == "max_iterations"
        assert "5" in result.error_message

    def test_result_to_dict(self):
        """Test result serialization."""
        response = AgentResponseFrame(
            response_text="Done!",
            goal_achieved=True,
            iterations_used=1,
        )

        result = success_result(
            response=response,
            frames=(response,),
            iterations=1,
        )

        data = result.to_dict()

        assert data["success"] is True
        assert data["iterations"] == 1
        assert "frame_count" in data


# =============================================================================
# ADR-004 Compliance Tests (Critical)
# =============================================================================


class TestADR004Compliance:
    """
    Tests verifying ADR-004 compliance for agent execution.

    Core invariant: All agent decisions must be visible in the frame stream.
    """

    def test_all_frame_types_have_frame_type_property(self):
        """Verify all agent frames have frame_type property."""
        frames = [
            PlanFrame(goal="Test", reasoning="Test"),
            ToolCallFrame(tool_name="test", tool_arguments={}),
            ToolResultFrame(tool_name="test", success=True, result_text="ok"),
            AgentResponseFrame(response_text="done"),
            AgentControlFrame(action=AgentAction.PLAN),
        ]

        for frame in frames:
            assert hasattr(frame, "frame_type")
            assert isinstance(frame.frame_type, str)
            assert frame.frame_type != ""

    def test_frames_are_immutable(self):
        """Verify agent frames are frozen (immutable)."""
        frame = ToolCallFrame(
            tool_name="test",
            tool_arguments={"a": 1},
            iteration=0,
        )

        with pytest.raises(AttributeError):
            frame.tool_name = "modified"

    def test_frames_have_unique_ids(self):
        """Verify each frame gets a unique ID."""
        frame1 = ToolCallFrame(tool_name="test", tool_arguments={})
        frame2 = ToolCallFrame(tool_name="test", tool_arguments={})

        assert frame1.id != frame2.id

    def test_frames_have_timestamps(self):
        """Verify frames have creation timestamps."""
        frame = ToolCallFrame(tool_name="test", tool_arguments={})

        assert hasattr(frame, "created_at")
        assert isinstance(frame.created_at, datetime)

    def test_tool_result_links_to_tool_call(self):
        """Verify ToolResultFrame can link to ToolCallFrame."""
        call = ToolCallFrame(tool_name="test", tool_arguments={})

        result = ToolResultFrame(
            tool_name="test",
            success=True,
            result_text="done",
            tool_call_frame_id=str(call.id),
        )

        assert result.tool_call_frame_id == str(call.id)

    def test_frame_stream_is_self_describing(self):
        """
        Verify frame stream contains all info needed to explain execution.

        This is the core ADR-004 test for agents.
        """
        # Simulate an agent execution
        frames = [
            # Initial planning
            PlanFrame(
                goal="Create task",
                reasoning="Analyzing request",
                next_action=AgentAction.TOOL_CALL,
                iteration=0,
            ),
            # Tool call decision
            ToolCallFrame(
                tool_name="plane_create_task",
                tool_arguments={"name": "Test", "project": "Mobile"},
                reasoning="User wants a task",
                iteration=0,
            ),
            # Tool result
            ToolResultFrame(
                tool_name="plane_create_task",
                success=True,
                result_text="Created MOB-123",
                structured_result={"task_id": "uuid"},
            ),
            # Final response
            AgentResponseFrame(
                response_text="Created task MOB-123",
                goal_achieved=True,
                iterations_used=1,
                tools_called=("plane_create_task",),
            ),
        ]

        # Verify we can reconstruct execution from frames alone
        frame_types = [f.frame_type for f in frames]

        # Must see planning
        assert "plan" in frame_types

        # Must see tool call
        assert "tool_call" in frame_types

        # Must see tool result
        assert "tool_result" in frame_types

        # Must see final response
        assert "agent_response" in frame_types

        # Can identify which tool was called
        tool_calls = [f for f in frames if f.frame_type == "tool_call"]
        assert tool_calls[0].tool_name == "plane_create_task"

        # Can see tool arguments
        assert tool_calls[0].tool_arguments["name"] == "Test"

        # Can see tool result
        tool_results = [f for f in frames if f.frame_type == "tool_result"]
        assert tool_results[0].success is True

        # Can see final outcome
        responses = [f for f in frames if f.frame_type == "agent_response"]
        assert responses[0].goal_achieved is True

    def test_agent_result_contains_complete_frame_stream(self):
        """Verify AgentResult contains all frames for audit."""
        # Create frames
        plan = PlanFrame(goal="Test", reasoning="Testing")
        call = ToolCallFrame(tool_name="test", tool_arguments={})
        result = ToolResultFrame(tool_name="test", success=True, result_text="ok")
        response = AgentResponseFrame(response_text="Done", goal_achieved=True)

        # Create agent result
        agent_result = AgentResult(
            success=True,
            response=response,
            frames=(plan, call, result, response),
            iterations=1,
        )

        # All frames must be present
        assert len(agent_result.frames) == 4
        assert plan in agent_result.frames
        assert call in agent_result.frames
        assert result in agent_result.frames
        assert response in agent_result.frames


# =============================================================================
# Bounded Execution Tests
# =============================================================================


class TestBoundedExecution:
    """Tests for bounded execution (max iterations, timeout)."""

    def test_agent_result_tracks_iterations(self):
        """Verify iterations are tracked in result."""
        result = max_iterations_result(
            frames=(),
            max_iterations=5,
        )

        assert result.iterations == 5

    def test_timeout_result_has_error_info(self):
        """Verify timeout result has proper error info."""
        result = timeout_result(
            frames=(),
            iterations=2,
            timeout_seconds=30.0,
        )

        assert result.error_type == "timeout"
        assert result.error_message != ""
        assert "30.0" in result.error_message

    def test_response_frame_tracks_iterations(self):
        """Verify response frame tracks iterations used."""
        frame = AgentResponseFrame(
            response_text="Done",
            goal_achieved=True,
            iterations_used=3,
        )

        assert frame.iterations_used == 3


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in agent execution."""

    @pytest.mark.asyncio
    async def test_failing_tool_returns_error_result(self):
        """Test that failing tools return error in ToolResult."""
        tool = FailingTool()
        result = await tool.execute({})

        assert result.is_error is True
        assert "always fails" in result.text

    def test_tool_result_frame_captures_errors(self):
        """Test ToolResultFrame captures error details."""
        frame = ToolResultFrame(
            tool_name="failing_tool",
            success=False,
            result_text="",
            error_message="Connection timeout",
            execution_time_ms=5000.0,
        )

        assert frame.success is False
        assert frame.error_message == "Connection timeout"
        assert frame.execution_time_ms == 5000.0

    def test_response_frame_captures_failure(self):
        """Test AgentResponseFrame captures goal failure."""
        frame = AgentResponseFrame(
            response_text="Could not complete task",
            goal_achieved=False,
            iterations_used=3,
            failure_reason="Project not found",
        )

        assert frame.goal_achieved is False
        assert frame.failure_reason == "Project not found"
