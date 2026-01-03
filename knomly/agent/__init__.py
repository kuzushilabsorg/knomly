"""
Knomly Agent Layer (v2).

The Agent Layer provides agentic capabilities on top of the v1 pipeline:
- AgentProcessor: Decision engine (LLM decides tool call or respond)
- AgentExecutor: While-loop executor with bounded iteration
- Agent Frames: Observable state for every decision

Design Principle (ADR-005):
    v2 is a CLIENT of v1, not an extension.
    Every agent decision emits a Frame for ADR-004 compliance.

Architecture:
    ┌──────────────────────────────────────────────────────┐
    │                   AgentExecutor                       │
    │                        │                              │
    │   Input: Goal + Context Frame                         │
    │                        │                              │
    │   ┌──────────────────────────────────────────────┐   │
    │   │           AgentProcessor (Loop)               │   │
    │   │                    │                          │   │
    │   │   LLM decides → ToolCallFrame or ResponseFrame│   │
    │   └──────────────────────────────────────────────┘   │
    │                        │                              │
    │              ┌─────────┼─────────┐                    │
    │              ▼                   ▼                    │
    │      ToolCallFrame        AgentResponseFrame          │
    │              │                   │                    │
    │         [Execute]            [Done]                   │
    │              │                                        │
    │       ToolResultFrame                                 │
    │              │                                        │
    │         [Loop back]                                   │
    └──────────────────────────────────────────────────────┘

Usage:
    from knomly.agent import AgentExecutor, AgentProcessor
    from knomly.tools import ToolRegistry
    from knomly.tools.plane import PlaneCreateTaskTool

    # Setup tools
    registry = ToolRegistry()
    registry.register(PlaneCreateTaskTool(client, cache))

    # Create agent
    processor = AgentProcessor(llm=llm_provider, tools=registry)
    executor = AgentExecutor(processor=processor, tools=registry)

    # Run
    result = await executor.run(
        goal="Create a task for Mobile App called 'Fix login bug'",
        initial_context=extraction_frame,
    )

    # All decisions are in result.frames
    for frame in result.frames:
        print(f"{frame.frame_type}: {frame.id}")
"""

from .frames import (
    AgentAction,
    PlanFrame,
    ToolCallFrame,
    ToolResultFrame,
    AgentResponseFrame,
    AgentControlFrame,
)
from .result import AgentResult

from .processor import AgentProcessor
from .executor import AgentExecutor, create_agent

from .memory import (
    Message,
    Conversation,
    MemoryProtocol,
    InMemoryStorage,
    RedisMemory,
    MemoryManager,
    create_memory,
    # v2.2 Frame Persistence
    ExecutionMemory,
    frame_to_message,
    message_to_frame,
)

__all__ = [
    # Frames
    "AgentAction",
    "PlanFrame",
    "ToolCallFrame",
    "ToolResultFrame",
    "AgentResponseFrame",
    "AgentControlFrame",
    # Result
    "AgentResult",
    # Processor and Executor
    "AgentProcessor",
    "AgentExecutor",
    "create_agent",
    # Memory (Phase 2.6)
    "Message",
    "Conversation",
    "MemoryProtocol",
    "InMemoryStorage",
    "RedisMemory",
    "MemoryManager",
    "create_memory",
    # Frame Persistence (v2.2)
    "ExecutionMemory",
    "frame_to_message",
    "message_to_frame",
]
