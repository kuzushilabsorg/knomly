"""
Agent Bridge Processor.

This processor bridges the v1 pipeline to the v2 agent layer.
It wraps the AgentExecutor and integrates it into the pipeline flow.

Design Principle (ADR-005):
    v2 is a CLIENT of v1, not an extension.
    This bridge allows the agent to consume v1 frames while
    preserving ADR-004 compliance (frame-based observability).

Multi-Tenancy (v2.1 - ADR-007):
    Tools are now built per-request via ToolFactory.
    This enables multi-tenant deployments where each user's
    request uses their specific API credentials.

Persistent Memory (v2.2):
    Optional ExecutionMemory enables crash recovery.
    Agent state is persisted to Redis and can be resumed.

Usage:
    # Static tools (single-tenant / testing)
    builder.add(AgentBridgeProcessor(
        tools=[PlaneCreateTaskTool(client, cache)],
        llm_provider=llm,
    ))

    # Dynamic tools (multi-tenant production)
    builder.add(AgentBridgeProcessor(
        tool_factory=PlaneToolFactory(),
        llm_provider=llm,
    ))

    # With persistent memory (crash-safe)
    builder.add(AgentBridgeProcessor(
        tool_factory=PlaneToolFactory(),
        llm_provider=llm,
        memory=ExecutionMemory(storage=RedisMemory()),
    ))

Flow:
    TranscriptionFrame (enriched with context)
        ↓
    AgentBridgeProcessor
        ↓ (extract ToolContext from Frame)
        ↓ (build tools via factory)
    [v2 AgentExecutor runs loop]
        ↓
    AgentResponseFrame → UserResponseFrame
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from knomly.pipeline.frames.action import UserResponseFrame
from knomly.pipeline.processor import Processor
from knomly.tools.factory import (
    StaticToolFactory,
    ToolContext,
    ToolFactory,
    extract_tool_context_from_frame,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from knomly.agent import AgentExecutor, ExecutionMemory
    from knomly.agent.result import AgentResult
    from knomly.integrations.plane.cache import PlaneEntityCache
    from knomly.integrations.plane.client import PlaneClient
    from knomly.pipeline.context import PipelineContext
    from knomly.pipeline.frames.base import Frame
    from knomly.providers.llm import LLMProvider
    from knomly.tools import Tool, ToolRegistry

logger = logging.getLogger(__name__)


class AgentBridgeProcessor(Processor):
    """
    Bridge between v1 pipeline and v2 agent layer.

    This processor:
    1. Receives a frame from v1 (typically TranscriptionFrame)
    2. Extracts user context from frame (for multi-tenancy)
    3. Builds tools via factory (with user's credentials)
    4. Runs the v2 AgentExecutor
    5. Converts AgentResponseFrame to v1-compatible output

    Multi-Tenancy (v2.1):
        Tools are now built per-request. Two modes:

        1. Static tools (backwards compatible, single-tenant):
           AgentBridgeProcessor(tools=[...])

        2. Dynamic tools (multi-tenant, recommended):
           AgentBridgeProcessor(tool_factory=PlaneToolFactory())

    ADR Compliance:
    - ADR-004: Agent frames logged for observability
    - ADR-005: Agent is client of pipeline, not extension
    - ADR-007: Tools built per-request with user credentials

    Example (static - single-tenant):
        processor = AgentBridgeProcessor(
            tools=[PlaneCreateTaskTool(client, cache)],
            llm_provider=llm,
        )

    Example (dynamic - multi-tenant):
        processor = AgentBridgeProcessor(
            tool_factory=PlaneToolFactory(base_url="..."),
            llm_provider=llm,
        )
    """

    def __init__(
        self,
        *,
        tools: list[Tool] | None = None,
        tool_factory: ToolFactory | None = None,
        tool_registry: ToolRegistry | None = None,
        llm_provider: LLMProvider | None = None,
        max_iterations: int = 5,
        timeout_seconds: float = 60.0,
        memory: ExecutionMemory | None = None,
        secret_provider: callable[[str], dict[str, str]] | None = None,
    ):
        """
        Initialize the bridge.

        Args:
            tools: Static tools for the agent (single-tenant mode)
            tool_factory: Factory to build tools per-request (multi-tenant)
            tool_registry: Pre-configured tool registry (alternative to tools)
            llm_provider: LLM provider for agent decisions
            max_iterations: Maximum agent loop iterations
            timeout_seconds: Maximum execution time
            memory: Optional ExecutionMemory for persistent execution (v2.2)
            secret_provider: Callback (user_id) -> dict for secrets (v3.2 security)

        Note:
            Provide either `tools` OR `tool_factory`, not both.
            If neither provided, tools must come from context.

        Security (v3.2):
            Secrets are NEVER read from Frame metadata.
            Use secret_provider callback to fetch secrets at runtime.
        """
        # Validate mutual exclusivity
        if tools and tool_factory:
            raise ValueError(
                "Provide either 'tools' OR 'tool_factory', not both. "
                "For multi-tenant, use tool_factory."
            )

        # Convert static tools to factory for uniform handling
        if tools:
            self._tool_factory: ToolFactory = StaticToolFactory(tools)
        elif tool_factory:
            self._tool_factory = tool_factory
        else:
            # No tools provided - will need to get from context or fail
            self._tool_factory = StaticToolFactory([])

        self._tool_registry = tool_registry
        self._llm_provider = llm_provider
        self._max_iterations = max_iterations
        self._timeout = timeout_seconds
        self._memory = memory
        self._secret_provider = secret_provider

        # Executor is now built per-request (for dynamic tools)
        # but we cache the LLM provider
        self._llm: LLMProvider | None = None

    @property
    def name(self) -> str:
        return "agent_bridge"

    async def initialize(self, ctx: PipelineContext) -> None:
        """
        Initialize the bridge.

        In v2.1, we no longer build the executor here (tools are dynamic).
        We only resolve the LLM provider.
        """
        # Get LLM provider from context if not provided
        llm = self._llm_provider
        if llm is None and ctx.providers:
            llm = ctx.providers.get_llm()

        if llm is None:
            raise RuntimeError(
                "AgentBridgeProcessor requires an LLM provider. "
                "Provide via constructor or ensure ctx.providers is configured."
            )

        self._llm = llm
        logger.info(f"[agent_bridge] Initialized | max_iterations={self._max_iterations}")

    def _build_executor(self, tools: Sequence[Tool]) -> AgentExecutor:
        """
        Build an executor for this request's tools.

        Called per-request to create executor with user-scoped tools.
        """
        from knomly.agent import AgentExecutor, AgentProcessor
        from knomly.tools import ToolRegistry

        # Build registry from tools
        registry = ToolRegistry()
        for tool in tools:
            registry.register(tool)

        # Create processor and executor
        processor = AgentProcessor(
            llm=self._llm,
            tools=registry,
            max_iterations=self._max_iterations,
        )

        return AgentExecutor(
            processor=processor,
            tools=registry,
            max_iterations=self._max_iterations,
            timeout_seconds=self._timeout,
            memory=self._memory,  # v2.2: Pass memory for persistence
        )

    def _generate_session_id(self, frame: Frame, tool_context: ToolContext) -> str:
        """
        Generate a deterministic session ID for this execution.

        CRITICAL: Session IDs MUST be idempotent. If the same request
        is retried (e.g., Twilio webhook retry), we MUST generate the
        EXACT SAME session ID to resume the existing session instead
        of starting a new one.

        Priority for external message ID:
        1. frame.metadata["message_sid"] - Twilio Message SID
        2. frame.metadata["external_id"] - Generic external ID
        3. frame.id - Fallback to internal UUID (less reliable for retries)

        NO timestamps or random values - these break idempotency.

        Format: sess_{user_id}_{message_id}
        """
        user_id = tool_context.user_id

        # Prefer external message ID for idempotency
        # Twilio stores MessageSid in metadata
        message_id = (
            frame.metadata.get("message_sid") or frame.metadata.get("external_id") or str(frame.id)
        )

        # Create a deterministic session ID
        return f"sess_{user_id}_{message_id}"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | Sequence[Frame] | None:
        """
        Process a frame through the v2 agent layer.

        Steps:
        1. Ensure LLM is initialized
        2. Extract ToolContext from Frame (user_id, secrets)
        3. Build tools via factory (with user's credentials)
        4. Build executor for this request
        5. Run agent loop
        6. Convert agent response to v1-compatible frame

        Args:
            frame: Input frame (typically TranscriptionFrame)
            ctx: Pipeline context

        Returns:
            UserResponseFrame with agent's response
        """
        start_time = time.monotonic()

        # Ensure LLM is initialized
        if self._llm is None:
            await self.initialize(ctx)

        # Extract goal from frame
        goal = self._extract_goal(frame)
        if not goal:
            logger.warning("[agent_bridge] No goal extracted from frame")
            return UserResponseFrame(
                message="I couldn't understand what you'd like me to do.",
                sender_phone=getattr(frame, "sender_phone", ""),
                source_frame_id=frame.id,
            )

        # Extract ToolContext from frame (v2.1 multi-tenancy)
        # SECURITY (v3.2): Use secret_provider callback, never from frame metadata
        tool_context = extract_tool_context_from_frame(
            frame,
            secret_provider=self._secret_provider,
        )

        # =================================================================
        # v3: Check for dynamic tools from PipelineResolver
        # =================================================================
        # Dynamic tools are injected into context.metadata by the webhook
        # when KNOMLY_USE_RESOLVER is enabled
        dynamic_tools: list = ctx.metadata.get("dynamic_tools", [])

        # Build tools for this user/request (factory-based)
        factory_tools = list(self._tool_factory.build_tools(tool_context))

        # =================================================================
        # Tool Deduplication with Precedence (v3.2)
        # =================================================================
        # Rule: Dynamic tools (from resolver) WIN over factory tools
        # This ensures database-configured tools override hardcoded ones
        seen_names: set[str] = set()
        tools: list = []

        # 1. First add dynamic tools (they win)
        for tool in dynamic_tools:
            if tool.name not in seen_names:
                tools.append(tool)
                seen_names.add(tool.name)

        # 2. Then add factory tools (only if name not already present)
        for tool in factory_tools:
            if tool.name not in seen_names:
                tools.append(tool)
                seen_names.add(tool.name)
            else:
                logger.warning(
                    f"[agent_bridge] Tool '{tool.name}' from factory SHADOWED by dynamic tool. "
                    f"Dynamic tool takes precedence. Review if this is intentional."
                )

        if dynamic_tools:
            logger.info(
                f"[agent_bridge] Using dynamic tools from resolver: "
                f"{[t.name for t in dynamic_tools]} | "
                f"factory tools: {[t.name for t in factory_tools]} | "
                f"final: {[t.name for t in tools]}"
            )

        if not tools:
            logger.warning(f"[agent_bridge] No tools built for user {tool_context.user_id}")
            return UserResponseFrame(
                message="I don't have the tools to help with that right now.",
                sender_phone=getattr(frame, "sender_phone", ""),
                source_frame_id=frame.id,
            )

        # Build executor for this request
        executor = self._build_executor(tools)

        # Generate session_id for persistent memory (v2.2)
        session_id = None
        if self._memory:
            session_id = self._generate_session_id(frame, tool_context)

        # Log context for observability
        context_info = self._extract_context_info(frame)
        logger.info(
            f"[agent_bridge] Running agent | "
            f"user={tool_context.user_id} | "
            f"tools={len(tools)} | "
            f"goal: {goal[:50]}... | "
            f"context_keys: {list(context_info.keys())}"
            f"{f' | session={session_id}' if session_id else ''}"
        )

        # Run agent executor
        result = await executor.run(
            goal=goal,
            initial_context=frame,  # Pass the enriched frame
            session_id=session_id,  # v2.2: Enable persistence
        )

        # Log result for observability
        duration_ms = (time.monotonic() - start_time) * 1000
        logger.info(
            f"[agent_bridge] Agent completed | "
            f"user={tool_context.user_id} | "
            f"success={result.success} | "
            f"iterations={result.iterations} | "
            f"tools_called={result.tools_called} | "
            f"duration_ms={duration_ms:.1f}"
        )

        # Convert to v1-compatible frame
        return self._create_response_frame(frame, result)

    def _extract_goal(self, frame: Frame) -> str:
        """
        Extract the goal/instruction from the frame.

        Tries multiple fields to find the user's intent.
        """
        # Try common text fields
        for attr in ["text", "english_text", "original_text", "message"]:
            if hasattr(frame, attr):
                text = getattr(frame, attr)
                if text and isinstance(text, str):
                    return text.strip()

        # Try metadata
        if "text" in frame.metadata:
            return str(frame.metadata["text"]).strip()

        return ""

    def _extract_context_info(self, frame: Frame) -> dict[str, Any]:
        """
        Extract context information from frame metadata.

        Returns a summary of available context for logging.
        """
        info: dict[str, Any] = {}

        # Check for plane context
        plane_ctx = frame.metadata.get("plane_context")
        if plane_ctx:
            info["plane_projects"] = len(plane_ctx.get("projects", []))
            info["plane_users"] = len(plane_ctx.get("users", []))
            info["cache_healthy"] = plane_ctx.get("cache_healthy", False)

        # Check for intent
        if "intent" in frame.metadata:
            info["intent"] = frame.metadata["intent"]

        # Check for extracted data
        if "extracted_project" in frame.metadata:
            info["extracted_project"] = frame.metadata["extracted_project"]

        return info

    def _create_response_frame(
        self,
        input_frame: Frame,
        result: AgentResult,
    ) -> UserResponseFrame:
        """
        Convert agent result to v1-compatible UserResponseFrame.
        """

        # Build response message
        if result.success and result.response:
            message = result.response.response_text
        else:
            message = (
                f"I encountered an issue: {result.error_message or 'Unknown error'}. "
                f"Please try again or rephrase your request."
            )

        # Include agent metadata for observability
        metadata = {
            "agent_success": result.success,
            "agent_iterations": result.iterations,
            "agent_tools_called": list(result.tools_called),
            "agent_duration_ms": result.duration_ms,
            "agent_frame_count": len(result.frames),
        }

        if result.error_type:
            metadata["agent_error_type"] = result.error_type

        return UserResponseFrame(
            message=message,
            sender_phone=getattr(input_frame, "sender_phone", ""),
            source_frame_id=input_frame.id,
            metadata={**input_frame.metadata, **metadata},
        )


# =============================================================================
# Factory Functions
# =============================================================================


def create_task_agent_bridge(
    plane_client: PlaneClient,
    plane_cache: PlaneEntityCache,
    llm_provider: LLMProvider,
    *,
    max_iterations: int = 5,
) -> AgentBridgeProcessor:
    """
    Create an AgentBridgeProcessor configured for task management.

    Args:
        plane_client: Plane API client
        plane_cache: Plane entity cache for name resolution
        llm_provider: LLM provider for agent decisions
        max_iterations: Maximum agent loop iterations

    Returns:
        Configured AgentBridgeProcessor with Plane tools
    """
    from knomly.tools.plane import PlaneCreateTaskTool, PlaneQueryTasksTool

    tools = [
        PlaneCreateTaskTool(client=plane_client, cache=plane_cache),
        PlaneQueryTasksTool(client=plane_client, cache=plane_cache),
    ]

    return AgentBridgeProcessor(
        tools=tools,
        llm_provider=llm_provider,
        max_iterations=max_iterations,
    )
