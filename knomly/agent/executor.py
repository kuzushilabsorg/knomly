"""
Agent Executor (While-Loop Engine).

The AgentExecutor runs the agent loop:
1. Get decision from AgentProcessor
2. If ToolCallFrame → execute tool → loop back
3. If AgentResponseFrame → done

Design Principle (ADR-005):
    - Bounded iteration (max_iterations)
    - Timeout protection
    - Every step emits a frame
    - Complete frame stream in result

v2.2 Enhancement:
    - Optional persistent memory for crash recovery
    - session_id for resumable executions
    - Frames persisted to Redis (or other backend)

This is the v2 engine. Unlike Pipeline (DAG), this runs a while loop.

Usage:
    # Basic usage (in-memory, no persistence)
    executor = AgentExecutor(
        processor=agent_processor,
        tools=tool_registry,
        max_iterations=5,
        timeout_seconds=60.0,
    )

    result = await executor.run(
        goal="Create a task for Mobile App",
        initial_context=extraction_frame,
    )

    # With persistent memory (crash-safe)
    from knomly.agent.memory import ExecutionMemory, RedisMemory

    executor = AgentExecutor(
        processor=agent_processor,
        tools=tool_registry,
        memory=ExecutionMemory(storage=RedisMemory()),
    )

    result = await executor.run(
        goal="Create a task",
        session_id="exec-123",  # Resumable
    )

    if result.success:
        print(result.response.response_text)
    else:
        print(f"Failed: {result.error_message}")

    # Audit: every decision is in result.frames
    for frame in result.frames:
        print(f"{frame.frame_type}: {frame.id}")
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from .frames import (
    AgentAction,
    AgentControlFrame,
    AgentResponseFrame,
    PlanFrame,
    ToolCallFrame,
    ToolResultFrame,
)
from .result import (
    AgentResult,
    error_result,
    max_iterations_result,
    success_result,
    timeout_result,
)

if TYPE_CHECKING:
    from knomly.pipeline.frames.base import Frame
    from knomly.tools import ToolRegistry
    from .processor import AgentProcessor
    from .memory import ExecutionMemory

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes agent loops with bounded iteration.

    This is the v2 engine. Unlike Pipeline (DAG), this runs a while loop.
    Every step emits frames for complete observability.

    Invariants (ADR-004/ADR-005):
    - All decisions visible in frame stream
    - Bounded iteration (max_iterations)
    - Timeout protection
    - No hidden state

    v2.2 Enhancement:
    - Optional persistent memory for crash recovery
    - session_id for resumable executions

    Example:
        executor = AgentExecutor(
            processor=processor,
            tools=registry,
            max_iterations=5,
        )

        result = await executor.run(goal="Create a task")

        # Check success
        if result.success:
            print(result.response.response_text)

        # Audit frames
        for frame in result.frames:
            print(f"{frame.frame_type}")

        # With persistence
        from knomly.agent.memory import ExecutionMemory, RedisMemory

        executor = AgentExecutor(
            processor=processor,
            tools=registry,
            memory=ExecutionMemory(storage=RedisMemory()),
        )

        result = await executor.run(
            goal="Create a task",
            session_id="user-123-task-1",  # Resumable
        )
    """

    def __init__(
        self,
        *,
        processor: "AgentProcessor",
        tools: "ToolRegistry",
        max_iterations: int = 5,
        timeout_seconds: float = 60.0,
        memory: "ExecutionMemory | None" = None,
    ):
        """
        Initialize the executor.

        Args:
            processor: AgentProcessor for decision making
            tools: ToolRegistry for tool execution
            max_iterations: Maximum loop iterations
            timeout_seconds: Maximum execution time
            memory: Optional ExecutionMemory for persistence (v2.2)
        """
        self._processor = processor
        self._tools = tools
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds
        self._memory = memory

    async def run(
        self,
        goal: str,
        initial_context: "Frame | None" = None,
        session_id: str | None = None,
    ) -> AgentResult:
        """
        Run the agent loop until goal is achieved or limit reached.

        Args:
            goal: What the agent should achieve
            initial_context: Optional context frame (e.g., ExtractionFrame)
            session_id: Optional session ID for persistent memory (v2.2).
                       If provided with memory, execution state is persisted
                       and can be resumed after crashes.

        Returns:
            AgentResult with success status and all emitted frames
        """
        # Generate session_id if memory is enabled but no ID provided
        effective_session_id = session_id or (str(uuid4()) if self._memory else None)

        frames: list[Frame] = []
        tools_called: list[str] = []
        started_at = datetime.utcnow()
        restored_iterations = 0

        # Try to restore from memory if session exists
        if self._memory and effective_session_id:
            restored = await self._restore_from_memory(effective_session_id)
            if restored:
                frames = restored["frames"]
                tools_called = restored["tools_called"]
                restored_iterations = restored["iterations"]
                logger.info(
                    f"[agent_executor] Restored session {effective_session_id}: "
                    f"{len(frames)} frames, {restored_iterations} iterations"
                )

        # Add initial context to frames (if not already present from restore)
        if initial_context and initial_context not in frames:
            frames.append(initial_context)
            if self._memory and effective_session_id:
                await self._memory.persist_frame(effective_session_id, initial_context)
            logger.info(
                f"[agent_executor] Starting with context frame: "
                f"{initial_context.frame_type}"
            )

        logger.info(
            f"[agent_executor] Starting agent loop. "
            f"Goal: {goal[:50]}... "
            f"Max iterations: {self._max_iterations}"
            f"{f' (session: {effective_session_id})' if effective_session_id else ''}"
        )

        start_time = time.time()

        try:
            # Adjust loop range to account for restored iterations
            remaining_iterations = self._max_iterations - restored_iterations

            for iteration in range(remaining_iterations):
                effective_iteration = iteration + restored_iterations

                # Check timeout
                elapsed = time.time() - start_time
                if elapsed > self._timeout:
                    logger.warning(
                        f"[agent_executor] Timeout after {elapsed:.1f}s "
                        f"at iteration {effective_iteration}"
                    )
                    return timeout_result(
                        frames=tuple(frames),
                        iterations=effective_iteration,
                        tools_called=tuple(tools_called),
                        timeout_seconds=self._timeout,
                        started_at=started_at,
                    )

                logger.info(
                    f"[agent_executor] Iteration {effective_iteration + 1}/{self._max_iterations}"
                )

                # Get decision from processor
                decision = await self._processor.decide(
                    goal=goal,
                    history=frames,
                    iteration=effective_iteration,
                )
                frames.append(decision)

                # Persist decision frame
                if self._memory and effective_session_id:
                    await self._memory.persist_frame(effective_session_id, decision)

                # Handle decision
                if isinstance(decision, AgentResponseFrame):
                    # Agent is done
                    logger.info(
                        f"[agent_executor] Agent responded. "
                        f"Goal achieved: {decision.goal_achieved}"
                    )
                    return success_result(
                        response=decision,
                        frames=tuple(frames),
                        iterations=effective_iteration + 1,
                        tools_called=tuple(tools_called),
                        started_at=started_at,
                    )

                elif isinstance(decision, ToolCallFrame):
                    # Execute the tool
                    logger.info(
                        f"[agent_executor] Executing tool: {decision.tool_name}"
                    )

                    result_frame = await self._execute_tool(decision)
                    frames.append(result_frame)
                    tools_called.append(decision.tool_name)

                    # Persist tool result frame
                    if self._memory and effective_session_id:
                        await self._memory.persist_frame(effective_session_id, result_frame)

                    # Log result
                    if result_frame.success:
                        logger.info(
                            f"[agent_executor] Tool succeeded: "
                            f"{result_frame.result_text[:100]}..."
                        )
                    else:
                        logger.warning(
                            f"[agent_executor] Tool failed: "
                            f"{result_frame.error_message}"
                        )

                    # Continue loop to let agent process result

                elif isinstance(decision, PlanFrame):
                    # Agent is still thinking
                    logger.debug(
                        f"[agent_executor] Agent planning: "
                        f"{decision.reasoning[:100]}..."
                    )

                    # If agent keeps planning without action, it might be stuck
                    # The processor should handle this by eventually responding

                else:
                    # Unknown frame type
                    logger.warning(
                        f"[agent_executor] Unknown decision type: "
                        f"{type(decision).__name__}"
                    )

                # Emit control frame for observability
                control_frame = AgentControlFrame(
                    action=self._get_action_from_decision(decision),
                    should_continue=True,
                    iteration=effective_iteration,
                )
                frames.append(control_frame)

                # Persist control frame
                if self._memory and effective_session_id:
                    await self._memory.persist_frame(effective_session_id, control_frame)

            # Max iterations reached
            logger.warning(
                f"[agent_executor] Max iterations ({self._max_iterations}) reached"
            )
            return max_iterations_result(
                frames=tuple(frames),
                max_iterations=self._max_iterations,
                tools_called=tuple(tools_called),
                started_at=started_at,
            )

        except asyncio.CancelledError:
            logger.info("[agent_executor] Execution cancelled")
            raise

        except Exception as e:
            logger.error(f"[agent_executor] Unexpected error: {e}")
            return error_result(
                error=e,
                frames=tuple(frames),
                iterations=len([f for f in frames if isinstance(f, ToolCallFrame)]),
                tools_called=tuple(tools_called),
                started_at=started_at,
            )

    async def _execute_tool(self, call: ToolCallFrame) -> ToolResultFrame:
        """
        Execute a tool and return result frame.

        Args:
            call: ToolCallFrame with tool name and arguments

        Returns:
            ToolResultFrame with execution result
        """
        tool = self._tools.get(call.tool_name)

        if tool is None:
            logger.error(f"[agent_executor] Unknown tool: {call.tool_name}")
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
                result_text=result.text,
                structured_result=result.structured_content,
                error_message="" if not result.is_error else result.text,
                execution_time_ms=duration,
                tool_call_frame_id=str(call.id),
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"[agent_executor] Tool execution error: {e}")

            return ToolResultFrame(
                tool_name=call.tool_name,
                success=False,
                result_text="",
                error_message=str(e),
                execution_time_ms=duration,
                tool_call_frame_id=str(call.id),
            )

    def _get_action_from_decision(
        self,
        decision: PlanFrame | ToolCallFrame | AgentResponseFrame,
    ) -> AgentAction:
        """Get AgentAction enum from decision frame."""
        if isinstance(decision, ToolCallFrame):
            return AgentAction.TOOL_CALL
        elif isinstance(decision, AgentResponseFrame):
            return AgentAction.RESPOND
        elif isinstance(decision, PlanFrame):
            return decision.next_action
        else:
            return AgentAction.PLAN

    async def _restore_from_memory(
        self,
        session_id: str,
    ) -> dict | None:
        """
        Restore execution state from memory.

        Args:
            session_id: Session to restore

        Returns:
            Dict with frames, tools_called, iterations, or None if no session
        """
        if not self._memory:
            return None

        try:
            # Restore frames
            frames = await self._memory.restore_history(session_id)
            if not frames:
                return None

            # Extract tools called and iteration count
            tools_called = [
                f.tool_name
                for f in frames
                if hasattr(f, "frame_type") and f.frame_type == "tool_call"
            ]

            iterations = len(tools_called)

            return {
                "frames": frames,
                "tools_called": tools_called,
                "iterations": iterations,
            }

        except Exception as e:
            logger.warning(
                f"[agent_executor] Failed to restore from memory: {e}"
            )
            return None


# =============================================================================
# Factory Functions
# =============================================================================


def create_agent(
    *,
    llm: "LLMProvider",
    tools: "ToolRegistry",
    max_iterations: int = 5,
    timeout_seconds: float = 60.0,
    memory: "ExecutionMemory | None" = None,
) -> AgentExecutor:
    """
    Create a fully configured agent.

    Args:
        llm: LLM provider for decision making
        tools: Registry of available tools
        max_iterations: Maximum loop iterations
        timeout_seconds: Maximum execution time
        memory: Optional ExecutionMemory for persistent execution (v2.2)

    Returns:
        Configured AgentExecutor ready to run

    Example:
        # Basic agent (in-memory only)
        agent = create_agent(llm=llm, tools=registry)

        # Persistent agent (crash-safe)
        from knomly.agent.memory import ExecutionMemory, RedisMemory

        agent = create_agent(
            llm=llm,
            tools=registry,
            memory=ExecutionMemory(storage=RedisMemory()),
        )

        result = await agent.run(
            goal="Create a task",
            session_id="user-123",  # Resumable
        )
    """
    from .processor import AgentProcessor

    processor = AgentProcessor(
        llm=llm,
        tools=tools,
        max_iterations=max_iterations,
    )

    return AgentExecutor(
        processor=processor,
        tools=tools,
        max_iterations=max_iterations,
        timeout_seconds=timeout_seconds,
        memory=memory,
    )
