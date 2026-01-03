# ADR-005: v2 Agentic Layer Architecture

## Status

Proposed

## Context

With v1.9 complete and hardened, Knomly is ready to evolve from **Data Pipelines** (linear) to **Agentic Workflows** (cyclic). The v1 pipeline framework provides:

- Robust frame-based data flow
- Provider abstractions (STT, LLM, Chat)
- Context-aware extraction with graceful degradation
- Platform-agnostic task management (TaskFrame)

However, v1 cannot:
- Make autonomous decisions
- Select and invoke tools based on user intent
- Loop until a goal is achieved
- Reason about multi-step workflows

v2 adds an **Agentic Layer** that sits ON TOP of v1, treating v1 as infrastructure.

## Decision

We will implement a v2 Agentic Layer with the following architecture:

### Core Principle

> **v2 is a CLIENT of v1, not an extension of v1.**

v2 may: plan, loop, decide, call tools, consult memory.
v2 may NOT: mutate pipelines, hide execution state, bypass Frames, rely on Context for meaning.

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            v2 AGENTIC LAYER                                  │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        AgentExecutor (Loop)                             │ │
│  │                                                                         │ │
│  │   Input: ExtractionFrame/TaskFrame + Goal                               │ │
│  │                          │                                               │ │
│  │                          ▼                                               │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │              AgentProcessor (Decision Engine)                    │   │ │
│  │   │                                                                  │   │ │
│  │   │   LLM decides: ToolCall? Respond? AskUser?                      │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                          │                                               │ │
│  │             ┌────────────┼────────────┐                                  │ │
│  │             ▼            ▼            ▼                                  │ │
│  │        PlanFrame   ToolCallFrame  ResponseFrame                          │ │
│  │                          │                                               │ │
│  │                          ▼                                               │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │                    Tool Registry                                 │   │ │
│  │   │                                                                  │   │ │
│  │   │   PlaneCreateTaskTool | PlaneQueryTaskTool | ZulipSendTool      │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                          │                                               │ │
│  │                          ▼                                               │ │
│  │                  ToolResultFrame                                         │ │
│  │                          │                                               │ │
│  │                          ▼                                               │ │
│  │               [Loop back to AgentProcessor]                              │ │
│  │                                                                         │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│   Output: AgentResponseFrame (with full trace of Frames)                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    │ Uses (not extends)
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            v1 PIPELINE LAYER                                 │
│                                                                              │
│   Processors: TranscriptionProcessor, ExtractionProcessor, PlaneProcessor   │
│   Frames: AudioInputFrame, TranscriptionFrame, TaskFrame, TaskResultFrame   │
│   Context: PipelineContext (utilities only)                                  │
│   Providers: STTProvider, LLMProvider, ChatProvider                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Specification

### 1. Tool Interface (MCP-Aligned)

Tools are the "hands" of the agent. They wrap existing v1 logic into callable units.

```python
# tools/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from enum import Enum

class ContentType(Enum):
    TEXT = "text"
    IMAGE = "image"
    RESOURCE = "resource"

@dataclass(frozen=True)
class ContentBlock:
    """Content block in tool result (MCP-aligned)."""
    type: ContentType
    text: str | None = None
    data: bytes | None = None  # For images
    mime_type: str | None = None
    uri: str | None = None  # For resources

    @classmethod
    def text(cls, content: str) -> "ContentBlock":
        return cls(type=ContentType.TEXT, text=content)

@dataclass(frozen=True)
class ToolAnnotations:
    """Behavioral hints for tools (MCP-aligned)."""
    title: str | None = None
    read_only_hint: bool = False      # Does not modify environment
    destructive_hint: bool = True     # May destroy data (default for writes)
    idempotent_hint: bool = False     # Repeated calls have same effect
    open_world_hint: bool = False     # Interacts with external systems

@dataclass(frozen=True)
class ToolResult:
    """Result from tool execution (MCP-aligned)."""
    content: tuple[ContentBlock, ...]
    is_error: bool = False
    structured_content: dict[str, Any] | None = None

    @classmethod
    def success(cls, text: str, structured: dict | None = None) -> "ToolResult":
        return cls(
            content=(ContentBlock.text(text),),
            is_error=False,
            structured_content=structured,
        )

    @classmethod
    def error(cls, message: str) -> "ToolResult":
        return cls(
            content=(ContentBlock.text(f"Error: {message}"),),
            is_error=True,
        )

class Tool(ABC):
    """Base class for all tools (MCP-aligned)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique identifier for the tool."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @property
    @abstractmethod
    def input_schema(self) -> dict[str, Any]:
        """JSON Schema for the tool's arguments."""
        ...

    @property
    def output_schema(self) -> dict[str, Any] | None:
        """Optional JSON Schema for output structure."""
        return None

    @property
    def annotations(self) -> ToolAnnotations:
        """Behavioral hints for the tool."""
        return ToolAnnotations()

    @abstractmethod
    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Execute the tool with the given arguments."""
        ...

    def to_llm_schema(self) -> dict[str, Any]:
        """Convert to schema format for LLM tool use."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }
```

### 2. Tool Registry

```python
# tools/registry.py

class ToolRegistry:
    """Registry of available tools for agents."""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[Tool]:
        """List all registered tools."""
        return list(self._tools.values())

    def to_llm_schemas(self) -> list[dict[str, Any]]:
        """Get all tool schemas for LLM."""
        return [tool.to_llm_schema() for tool in self._tools.values()]
```

### 3. Agent Frames

Every agent decision MUST emit a frame for observability:

```python
# agent/frames.py

from dataclasses import dataclass, field
from typing import Any
from enum import Enum
from knomly.pipeline.frames.base import Frame

class AgentAction(Enum):
    PLAN = "plan"           # Agent is reasoning
    TOOL_CALL = "tool_call" # Agent decided to call a tool
    RESPOND = "respond"     # Agent has final response
    ASK_USER = "ask_user"   # Agent needs clarification

@dataclass(frozen=True, kw_only=True, slots=True)
class PlanFrame(Frame):
    """
    Agent's reasoning state.

    This frame captures WHAT the agent is thinking.
    Required for ADR-004 compliance: execution explainable from frames alone.
    """
    goal: str                           # What the agent is trying to achieve
    reasoning: str                      # Agent's current reasoning
    observations: tuple[str, ...] = ()  # What the agent has observed
    next_action: AgentAction = AgentAction.PLAN
    iteration: int = 0                  # Current iteration in the loop
    max_iterations: int = 5             # Bounded execution

@dataclass(frozen=True, kw_only=True, slots=True)
class ToolCallFrame(Frame):
    """
    Agent's decision to call a tool.

    This frame captures WHAT tool and with WHAT arguments.
    Emitted BEFORE tool execution for observability.
    """
    tool_name: str
    tool_arguments: dict[str, Any] = field(default_factory=dict)
    reasoning: str = ""                 # Why this tool was chosen
    iteration: int = 0
    plan_frame_id: str | None = None    # Links to the PlanFrame

@dataclass(frozen=True, kw_only=True, slots=True)
class ToolResultFrame(Frame):
    """
    Result from tool execution.

    This frame captures WHAT happened when the tool ran.
    Links to ToolCallFrame for complete trace.
    """
    tool_name: str
    success: bool
    result_text: str                    # Human-readable result
    structured_result: dict[str, Any] | None = None
    error_message: str = ""
    execution_time_ms: float = 0.0
    tool_call_frame_id: str | None = None  # Links to ToolCallFrame

@dataclass(frozen=True, kw_only=True, slots=True)
class AgentResponseFrame(Frame):
    """
    Agent's final response to the user.

    This frame captures the CONCLUSION of the agent loop.
    """
    response_text: str                  # What to tell the user
    goal_achieved: bool = True          # Did we achieve the goal?
    iterations_used: int = 0            # How many loop iterations
    tools_called: tuple[str, ...] = ()  # Which tools were invoked
    reasoning_trace: str = ""           # Summary of reasoning

@dataclass(frozen=True, kw_only=True, slots=True)
class AgentControlFrame(Frame):
    """
    Control signal for agent execution.

    Used internally by AgentExecutor to manage loop state.
    """
    action: AgentAction
    should_continue: bool = True
    iteration: int = 0
    max_iterations_reached: bool = False
    timeout_reached: bool = False
```

### 4. AgentProcessor

The "brain" that decides what to do:

```python
# agent/processor.py

class AgentProcessor:
    """
    Decision engine for agent execution.

    Takes a goal and available tools, uses LLM to decide:
    - Call a tool (ToolCallFrame)
    - Respond to user (AgentResponseFrame)
    - Ask for clarification (AskUserFrame)

    All decisions emit frames for ADR-004 compliance.
    """

    def __init__(
        self,
        *,
        llm_provider: LLMProvider,
        tools: ToolRegistry,
        max_iterations: int = 5,
    ):
        self._llm = llm_provider
        self._tools = tools
        self._max_iterations = max_iterations

    async def decide(
        self,
        goal: str,
        history: list[Frame],
        iteration: int,
    ) -> PlanFrame | ToolCallFrame | AgentResponseFrame:
        """
        Make a decision based on goal and history.

        Args:
            goal: What the agent is trying to achieve
            history: Previous frames in this agent loop
            iteration: Current iteration number

        Returns:
            Frame representing the decision
        """
        # Build prompt with goal, tools, and history
        messages = self._build_messages(goal, history, iteration)

        # Get LLM decision (with tool use)
        response = await self._llm.complete(
            messages=messages,
            config=LLMConfig(
                temperature=0.1,  # Deterministic decisions
                max_tokens=1024,
                tools=self._tools.to_llm_schemas(),
            ),
        )

        # Parse LLM response into frame
        return self._parse_decision(response, goal, iteration)
```

### 5. AgentExecutor

The while loop that runs the agent:

```python
# agent/executor.py

class AgentExecutor:
    """
    Executes agent loops with bounded iteration.

    This is the v2 engine. Unlike Pipeline (DAG), this runs a while loop.
    Every step emits frames for complete observability.

    Invariants (ADR-004):
    - All decisions visible in frame stream
    - Bounded iteration (max_iterations)
    - No hidden state
    """

    def __init__(
        self,
        *,
        processor: AgentProcessor,
        tools: ToolRegistry,
        max_iterations: int = 5,
        timeout_seconds: float = 60.0,
    ):
        self._processor = processor
        self._tools = tools
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds

    async def run(
        self,
        goal: str,
        initial_context: Frame | None = None,
    ) -> AgentResult:
        """
        Run the agent loop until goal is achieved or limit reached.

        Args:
            goal: What the agent should achieve
            initial_context: Optional context frame (e.g., ExtractionFrame)

        Returns:
            AgentResult with all emitted frames
        """
        frames: list[Frame] = []
        tools_called: list[str] = []

        if initial_context:
            frames.append(initial_context)

        start_time = time.time()

        for iteration in range(self._max_iterations):
            # Check timeout
            if time.time() - start_time > self._timeout:
                return self._timeout_result(frames, tools_called, iteration)

            # Get agent decision
            decision = await self._processor.decide(
                goal=goal,
                history=frames,
                iteration=iteration,
            )
            frames.append(decision)

            # Handle decision
            if isinstance(decision, AgentResponseFrame):
                # Agent is done
                return AgentResult(
                    success=True,
                    response=decision,
                    frames=tuple(frames),
                    iterations=iteration + 1,
                    tools_called=tuple(tools_called),
                )

            elif isinstance(decision, ToolCallFrame):
                # Execute tool
                result_frame = await self._execute_tool(decision)
                frames.append(result_frame)
                tools_called.append(decision.tool_name)

                if result_frame.success is False:
                    # Tool failed - agent can retry or respond with error
                    pass

            elif isinstance(decision, PlanFrame):
                # Agent is still thinking, continue loop
                pass

        # Max iterations reached
        return self._max_iterations_result(frames, tools_called)

    async def _execute_tool(self, call: ToolCallFrame) -> ToolResultFrame:
        """Execute a tool and return result frame."""
        tool = self._tools.get(call.tool_name)

        if tool is None:
            return ToolResultFrame(
                tool_name=call.tool_name,
                success=False,
                result_text="",
                error_message=f"Unknown tool: {call.tool_name}",
                tool_call_frame_id=str(call.id),
            )

        start = time.time()
        try:
            result = await tool.execute(call.tool_arguments)
            duration = (time.time() - start) * 1000

            return ToolResultFrame(
                tool_name=call.tool_name,
                success=not result.is_error,
                result_text=result.content[0].text if result.content else "",
                structured_result=result.structured_content,
                error_message="" if not result.is_error else result.content[0].text,
                execution_time_ms=duration,
                tool_call_frame_id=str(call.id),
            )
        except Exception as e:
            duration = (time.time() - start) * 1000
            return ToolResultFrame(
                tool_name=call.tool_name,
                success=False,
                result_text="",
                error_message=str(e),
                execution_time_ms=duration,
                tool_call_frame_id=str(call.id),
            )
```

### 6. First Tool: PlaneCreateTaskTool

Wraps existing Plane logic:

```python
# tools/plane/create_task.py

class PlaneCreateTaskTool(Tool):
    """Tool for creating tasks in Plane."""

    def __init__(self, client: PlaneClient, cache: PlaneEntityCache):
        self._client = client
        self._cache = cache

    @property
    def name(self) -> str:
        return "plane_create_task"

    @property
    def description(self) -> str:
        return (
            "Create a new task in Plane project management. "
            "Use this when the user wants to create a task, issue, or work item."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Task title/name",
                },
                "description": {
                    "type": "string",
                    "description": "Task description (optional)",
                    "default": "",
                },
                "project": {
                    "type": "string",
                    "description": "Project name or identifier",
                },
                "priority": {
                    "type": "string",
                    "enum": ["urgent", "high", "medium", "low", "none"],
                    "description": "Task priority",
                    "default": "medium",
                },
                "assignee": {
                    "type": "string",
                    "description": "User to assign (name or email)",
                    "default": "",
                },
            },
            "required": ["name", "project"],
        }

    @property
    def annotations(self) -> ToolAnnotations:
        return ToolAnnotations(
            title="Create Plane Task",
            read_only_hint=False,
            destructive_hint=False,  # Creates, doesn't destroy
            idempotent_hint=False,   # Creates new task each time
            open_world_hint=True,    # Calls external Plane API
        )

    async def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Create task in Plane."""
        try:
            # Resolve project name to ID using cache
            project_name = arguments.get("project", "")
            project_id = self._cache.resolve_project(project_name)

            if not project_id:
                return ToolResult.error(
                    f"Unknown project: {project_name}. "
                    f"Valid projects: {list(self._cache.get_project_mapping().keys())}"
                )

            # Create the task
            issue = await self._client.create_issue(
                project_id=project_id,
                name=arguments["name"],
                description=arguments.get("description", ""),
                priority=self._map_priority(arguments.get("priority", "medium")),
            )

            return ToolResult.success(
                text=f"Created task '{issue.name}' in project (ID: {issue.id})",
                structured={
                    "task_id": issue.id,
                    "task_name": issue.name,
                    "project_id": project_id,
                    "sequence_id": issue.sequence_id,
                },
            )

        except Exception as e:
            return ToolResult.error(str(e))
```

## File Structure

```
knomly/
├── tools/
│   ├── __init__.py
│   ├── base.py              # Tool, ToolResult, ToolAnnotations
│   ├── registry.py          # ToolRegistry
│   └── plane/
│       ├── __init__.py
│       ├── create_task.py   # PlaneCreateTaskTool
│       └── query_tasks.py   # PlaneQueryTasksTool
├── agent/
│   ├── __init__.py
│   ├── frames.py            # PlanFrame, ToolCallFrame, etc.
│   ├── processor.py         # AgentProcessor
│   ├── executor.py          # AgentExecutor
│   └── result.py            # AgentResult
├── pipeline/                 # UNCHANGED - v1 preserved
└── providers/                # UNCHANGED - v1 preserved
```

## Implementation Phases

### Phase 2.0: Tool Foundation (Week 1)
- [x] Design Tool interface (this ADR)
- [ ] Implement `tools/base.py`
- [ ] Implement `tools/registry.py`
- [ ] Implement `PlaneCreateTaskTool`
- [ ] Write unit tests for Tool execution

### Phase 2.1: Agent Frames (Week 1)
- [ ] Implement `agent/frames.py`
- [ ] Add frame derivation methods
- [ ] Write ADR-004 compliance tests

### Phase 2.2: Agent Processor (Week 2)
- [ ] Implement `AgentProcessor`
- [ ] Implement LLM prompt building
- [ ] Implement decision parsing
- [ ] Write unit tests with mocked LLM

### Phase 2.3: Agent Executor (Week 2)
- [ ] Implement `AgentExecutor`
- [ ] Implement bounded iteration
- [ ] Implement timeout protection
- [ ] Write integration tests

### Phase 2.4: Integration (Week 3)
- [ ] Connect agent to v1 pipeline output
- [ ] Implement PlaneQueryTasksTool
- [ ] End-to-end testing
- [ ] Documentation

## Testing Strategy

### ADR-004 Compliance Test

```python
def test_agent_loop_is_explainable_from_frames():
    """
    Core test: Can we explain execution from frames alone?

    This is the v2 equivalent of v1's frame stream test.
    """
    result = await agent.run(goal="Create a task for Mobile App")

    # Every decision must be visible in frames
    frame_types = [type(f).__name__ for f in result.frames]

    # Must have planning frame(s)
    assert "PlanFrame" in frame_types or "ToolCallFrame" in frame_types

    # Must have tool calls (if any were made)
    if result.tools_called:
        assert "ToolCallFrame" in frame_types
        assert "ToolResultFrame" in frame_types

    # Must have final response
    assert "AgentResponseFrame" in frame_types

    # Can reconstruct full execution from frames
    for frame in result.frames:
        # Each frame must be self-describing
        assert frame.id is not None
        assert frame.created_at is not None
```

### Bounded Iteration Test

```python
def test_agent_loop_is_bounded():
    """Agent loop must terminate within max_iterations."""
    agent = AgentExecutor(max_iterations=3)

    # Mock a tool that always returns "need more info"
    result = await agent.run(goal="Impossible task")

    assert result.iterations <= 3
    assert result.frames[-1].max_iterations_reached or result.success
```

## Consequences

### Positive

- **Explainable**: Every agent decision is a frame
- **Bounded**: Max iterations prevent runaway loops
- **Composable**: Tools are independent, testable units
- **Safe**: v1 pipeline unchanged, v2 is a client

### Negative

- **Verbosity**: More frames, more explicit state
- **Complexity**: Agent loop adds cognitive overhead

### Mitigations

- Frame-based approach makes debugging EASIER despite verbosity
- Clear separation of concerns reduces overall complexity

## References

- [ADR-004: v1/v2 Layer Invariants](./ADR-004-v1-v2-invariants.md)
- [MCP Tool Specification](https://modelcontextprotocol.io/specification/)
- [Pipecat Framework](https://docs.pipecat.ai/)
