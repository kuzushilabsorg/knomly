"""
Agent Frames for v2 Agentic Layer.

These frames capture every agent decision for observability:
- PlanFrame: Agent's reasoning state
- ToolCallFrame: Decision to call a tool
- ToolResultFrame: Result from tool execution
- AgentResponseFrame: Final response to user
- AgentControlFrame: Internal control signals

Design Principle (ADR-004):
    "If I can't explain an execution by looking only at the Frame stream,
     the design is broken."

    Every agent decision MUST emit a frame. This enables:
    - Complete audit trail
    - Replayable execution
    - Observable reasoning

Usage:
    # Agent decides to call a tool
    call_frame = ToolCallFrame(
        tool_name="plane_create_task",
        tool_arguments={"name": "Fix bug", "project": "Mobile"},
        reasoning="User asked to create a task",
        iteration=0,
    )

    # Tool returns result
    result_frame = ToolResultFrame(
        tool_name="plane_create_task",
        success=True,
        result_text="Created task MOB-123",
        tool_call_frame_id=str(call_frame.id),
    )

    # Agent responds
    response_frame = AgentResponseFrame(
        response_text="I've created task MOB-123 in Mobile App.",
        goal_achieved=True,
        iterations_used=1,
        tools_called=("plane_create_task",),
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from knomly.pipeline.frames.base import Frame


class AgentAction(Enum):
    """
    Type of action the agent is taking.

    Used in PlanFrame and AgentControlFrame to indicate
    what the agent decided to do.
    """

    PLAN = "plan"  # Agent is reasoning/planning
    TOOL_CALL = "tool_call"  # Agent decided to call a tool
    RESPOND = "respond"  # Agent has final response for user
    ASK_USER = "ask_user"  # Agent needs clarification from user
    ERROR = "error"  # Agent encountered an error


@dataclass(frozen=True, kw_only=True, slots=True)
class PlanFrame(Frame):
    """
    Agent's reasoning state.

    This frame captures WHAT the agent is thinking. It makes the
    agent's reasoning process observable and auditable.

    Emitted when the agent is:
    - Starting to work on a goal
    - Revising its plan after tool results
    - Deciding on next action

    Example:
        PlanFrame(
            goal="Create a high-priority task for Mobile App",
            reasoning="User wants a task created. I have the project name.",
            observations=("Project 'Mobile App' exists",),
            next_action=AgentAction.TOOL_CALL,
            iteration=0,
        )
    """

    # What the agent is trying to achieve
    goal: str

    # Agent's current reasoning (chain of thought)
    reasoning: str

    # What the agent has observed (from previous steps)
    observations: tuple[str, ...] = ()

    # What the agent will do next
    next_action: AgentAction = AgentAction.PLAN

    # Current iteration in the agent loop
    iteration: int = 0

    # Maximum iterations allowed (for observability)
    max_iterations: int = 5

    @property
    def frame_type(self) -> str:
        return "plan"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallFrame(Frame):
    """
    Agent's decision to call a tool.

    This frame is emitted BEFORE tool execution, capturing:
    - Which tool was selected
    - What arguments were passed
    - Why this tool was chosen

    This enables observability of the decision-making process.

    Example:
        ToolCallFrame(
            tool_name="plane_create_task",
            tool_arguments={
                "name": "Fix login bug",
                "project": "Mobile App",
                "priority": "high",
            },
            reasoning="User wants a task created with high priority",
            iteration=0,
        )
    """

    # Which tool to call
    tool_name: str

    # Arguments to pass to the tool
    tool_arguments: dict[str, Any] = field(default_factory=dict)

    # Why this tool was chosen (chain of thought)
    reasoning: str = ""

    # Current iteration in agent loop
    iteration: int = 0

    # Link to the PlanFrame that led to this decision
    plan_frame_id: str | None = None

    @property
    def frame_type(self) -> str:
        return "tool_call"


@dataclass(frozen=True, kw_only=True, slots=True)
class ToolResultFrame(Frame):
    """
    Result from tool execution.

    This frame is emitted AFTER tool execution, capturing:
    - Whether the tool succeeded
    - What the tool returned
    - How long it took

    This enables debugging and replay of tool executions.

    Example:
        ToolResultFrame(
            tool_name="plane_create_task",
            success=True,
            result_text="Created task 'Fix login bug' (MOB-123)",
            structured_result={
                "task_id": "uuid-123",
                "identifier": "MOB-123",
            },
            execution_time_ms=245.5,
            tool_call_frame_id="call-frame-uuid",
        )
    """

    # Which tool was called
    tool_name: str

    # Whether execution succeeded
    success: bool

    # Human-readable result text
    result_text: str

    # Structured result data (for programmatic use)
    structured_result: dict[str, Any] | None = None

    # Error message if success=False
    error_message: str = ""

    # Execution time in milliseconds
    execution_time_ms: float = 0.0

    # Link to the ToolCallFrame that triggered this
    tool_call_frame_id: str | None = None

    @property
    def frame_type(self) -> str:
        return "tool_result"


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentResponseFrame(Frame):
    """
    Agent's final response to the user.

    This frame is emitted when the agent has completed its task
    and has a response ready for the user.

    Example:
        AgentResponseFrame(
            response_text="I've created task MOB-123 'Fix login bug' in Mobile App with high priority.",
            goal_achieved=True,
            iterations_used=1,
            tools_called=("plane_create_task",),
            reasoning_trace="Identified task creation intent. Called plane_create_task. Success.",
        )
    """

    # What to tell the user
    response_text: str

    # Whether the goal was achieved
    goal_achieved: bool = True

    # How many iterations were used
    iterations_used: int = 0

    # Which tools were called (in order)
    tools_called: tuple[str, ...] = ()

    # Summary of reasoning process
    reasoning_trace: str = ""

    # If goal not achieved, why
    failure_reason: str = ""

    @property
    def frame_type(self) -> str:
        return "agent_response"


@dataclass(frozen=True, kw_only=True, slots=True)
class AgentControlFrame(Frame):
    """
    Control signal for agent execution.

    Used internally by AgentExecutor to manage loop state.
    Captures control decisions for observability.

    This frame is NOT typically visible to users, but is
    included in the frame stream for debugging.

    Example:
        AgentControlFrame(
            action=AgentAction.TOOL_CALL,
            should_continue=True,
            iteration=0,
        )
    """

    # What action was taken
    action: AgentAction

    # Whether the loop should continue
    should_continue: bool = True

    # Current iteration
    iteration: int = 0

    # Whether max iterations was reached
    max_iterations_reached: bool = False

    # Whether timeout was reached
    timeout_reached: bool = False

    # Error message if action=ERROR
    error_message: str = ""

    @property
    def frame_type(self) -> str:
        return "agent_control"


# =============================================================================
# Factory Functions
# =============================================================================


def create_initial_plan(
    goal: str,
    *,
    max_iterations: int = 5,
    initial_observations: tuple[str, ...] = (),
) -> PlanFrame:
    """
    Create initial PlanFrame for starting an agent loop.

    Args:
        goal: What the agent should achieve
        max_iterations: Maximum loop iterations
        initial_observations: Any context from prior frames

    Returns:
        PlanFrame ready for agent processing
    """
    return PlanFrame(
        goal=goal,
        reasoning="Starting agent loop. Analyzing goal and available tools.",
        observations=initial_observations,
        next_action=AgentAction.PLAN,
        iteration=0,
        max_iterations=max_iterations,
    )


def create_tool_result_from_tool_output(
    tool_name: str,
    tool_call_frame: ToolCallFrame,
    result: "ToolResult",  # from tools.base
    execution_time_ms: float,
) -> ToolResultFrame:
    """
    Create ToolResultFrame from a ToolResult.

    Args:
        tool_name: Name of the tool that was called
        tool_call_frame: The ToolCallFrame that triggered this
        result: ToolResult from tool execution
        execution_time_ms: How long execution took

    Returns:
        ToolResultFrame for the frame stream
    """
    from knomly.tools.base import ToolResult as ToolResultType

    return ToolResultFrame(
        tool_name=tool_name,
        success=not result.is_error,
        result_text=result.text,
        structured_result=result.structured_content,
        error_message="" if not result.is_error else result.text,
        execution_time_ms=execution_time_ms,
        tool_call_frame_id=str(tool_call_frame.id),
    )
