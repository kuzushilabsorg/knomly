"""
Agent Processor (Decision Engine).

The AgentProcessor is the "brain" of the agent. It uses an LLM to:
- Analyze the goal and context
- Decide whether to call a tool or respond
- Generate appropriate frames for each decision

Design Principle (ADR-005):
    Every decision emits a frame. The processor does NOT execute tools;
    it only decides WHAT to do. Execution is handled by AgentExecutor.

Security:
    LLM responses are validated with Pydantic to prevent malformed
    responses from causing unexpected behavior.

Usage:
    processor = AgentProcessor(llm=llm_provider, tools=tool_registry)

    decision = await processor.decide(
        goal="Create a task for Mobile App",
        history=[extraction_frame],
        iteration=0,
    )

    if isinstance(decision, ToolCallFrame):
        # Execute the tool
        ...
    elif isinstance(decision, AgentResponseFrame):
        # Done
        ...
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, Literal, Optional

from pydantic import BaseModel, Field, ValidationError

from .frames import (
    AgentAction,
    AgentResponseFrame,
    PlanFrame,
    ToolCallFrame,
)

if TYPE_CHECKING:
    from knomly.pipeline.frames.base import Frame
    from knomly.providers.llm import LLMProvider
    from knomly.tools import ToolRegistry

logger = logging.getLogger(__name__)


# =============================================================================
# LLM Response Schemas (Pydantic Validation)
# =============================================================================


class ToolCallResponse(BaseModel):
    """Schema for tool_call action from LLM."""

    action: Literal["tool_call"]
    tool: str = Field(..., min_length=1, description="Name of the tool to call")
    arguments: dict[str, Any] = Field(default_factory=dict, description="Tool arguments")
    reasoning: str = Field(default="", description="Why this tool was chosen")


class RespondResponse(BaseModel):
    """Schema for respond action from LLM."""

    action: Literal["respond"]
    message: str = Field(..., min_length=1, description="Response message to user")
    reasoning: str = Field(default="", description="Reasoning for this response")


class AskUserResponse(BaseModel):
    """Schema for ask_user action from LLM."""

    action: Literal["ask_user"]
    question: str = Field(..., min_length=1, description="Question to ask the user")
    reasoning: str = Field(default="", description="Why clarification is needed")


class LLMDecision(BaseModel):
    """
    Validated LLM decision schema.

    This is the union type that validates all possible actions.
    Use model_validate to parse raw JSON from LLM.
    """

    action: str = Field(..., description="Action type: tool_call, respond, or ask_user")
    tool: Optional[str] = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    message: Optional[str] = None
    question: Optional[str] = None
    reasoning: str = Field(default="")

    def is_tool_call(self) -> bool:
        return self.action == "tool_call" and self.tool is not None

    def is_respond(self) -> bool:
        return self.action == "respond" and self.message is not None

    def is_ask_user(self) -> bool:
        return self.action == "ask_user" and self.question is not None


# =============================================================================
# System Prompts
# =============================================================================

AGENT_SYSTEM_PROMPT = """You are an AI agent that helps users accomplish tasks by using available tools.

Your role is to:
1. Analyze the user's goal and any context provided
2. Decide whether to call a tool or respond to the user
3. If calling a tool, provide the exact arguments needed
4. If responding, provide a helpful message

Available tools:
{tools_section}

{entity_context}

Important rules:
- Only call one tool at a time
- If you have enough information to respond, do so directly
- If you need more information, call the appropriate tool
- When referencing projects or users, use the exact names from the "Valid Entities" section above
- Be concise and helpful in your responses

Current context:
{context_section}

Respond in JSON format with exactly one of these structures:

For calling a tool:
{{"action": "tool_call", "tool": "<tool_name>", "arguments": {{...}}, "reasoning": "<why>"}}

For responding to the user:
{{"action": "respond", "message": "<response>", "reasoning": "<why>"}}

For asking the user a question:
{{"action": "ask_user", "question": "<question>", "reasoning": "<why>"}}
"""

TOOLS_SECTION_TEMPLATE = """
Tool: {name}
Description: {description}
Arguments: {arguments}
"""


# =============================================================================
# Agent Processor
# =============================================================================


class AgentProcessor:
    """
    Decision engine for agent execution.

    The processor uses an LLM to decide what action to take:
    - Call a tool (ToolCallFrame)
    - Respond to user (AgentResponseFrame)
    - Ask for clarification (AgentResponseFrame with question)

    All decisions emit frames for ADR-004 compliance.

    Example:
        processor = AgentProcessor(llm=llm_provider, tools=registry)

        decision = await processor.decide(
            goal="Create a task called 'Fix bug' in Mobile App",
            history=[],
            iteration=0,
        )

        if isinstance(decision, ToolCallFrame):
            print(f"Calling {decision.tool_name}")
        elif isinstance(decision, AgentResponseFrame):
            print(f"Response: {decision.response_text}")
    """

    def __init__(
        self,
        *,
        llm: "LLMProvider",
        tools: "ToolRegistry",
        max_iterations: int = 5,
    ):
        """
        Initialize the processor.

        Args:
            llm: LLM provider for decision making
            tools: Registry of available tools
            max_iterations: Maximum iterations (for frame metadata)
        """
        self._llm = llm
        self._tools = tools
        self._max_iterations = max_iterations

    async def decide(
        self,
        goal: str,
        history: list["Frame"],
        iteration: int,
    ) -> PlanFrame | ToolCallFrame | AgentResponseFrame:
        """
        Make a decision based on goal and history.

        Args:
            goal: What the agent is trying to achieve
            history: Previous frames in this agent loop
            iteration: Current iteration number

        Returns:
            Frame representing the decision:
            - PlanFrame: Agent is still thinking
            - ToolCallFrame: Agent decided to call a tool
            - AgentResponseFrame: Agent has final response
        """
        # Build the prompt
        system_prompt = self._build_system_prompt(goal, history)
        user_message = self._build_user_message(goal, history, iteration)

        logger.info(
            f"[agent_processor] Iteration {iteration}: "
            f"Deciding action for goal: {goal[:50]}..."
        )

        # Call LLM
        from knomly.providers.llm import LLMConfig, Message

        messages = [
            Message.system(system_prompt),
            Message.user(user_message),
        ]

        try:
            response = await self._llm.complete(
                messages=messages,
                config=LLMConfig(
                    temperature=0.1,  # Low temperature for deterministic decisions
                    max_tokens=1024,
                    response_format="json",
                ),
            )

            # Parse the response
            return self._parse_response(response.content, goal, iteration, history)

        except Exception as e:
            logger.error(f"[agent_processor] LLM call failed: {e}")
            # Return error response
            return AgentResponseFrame(
                response_text=f"I encountered an error: {e}",
                goal_achieved=False,
                iterations_used=iteration,
                failure_reason=str(e),
            )

    def _build_system_prompt(
        self,
        goal: str,
        history: list["Frame"],
    ) -> str:
        """Build the system prompt with tools and context."""
        # Build tools section
        tools_section = self._build_tools_section()

        # Build entity context from frame metadata (plane_context, etc.)
        entity_context = self._build_entity_context(history)

        # Build execution context section from history
        context_section = self._build_context_section(history)

        return AGENT_SYSTEM_PROMPT.format(
            tools_section=tools_section,
            entity_context=entity_context,
            context_section=context_section,
        )

    def _build_tools_section(self) -> str:
        """Build the tools description for the prompt."""
        sections = []

        for tool in self._tools.list_tools():
            # Format arguments from input_schema
            props = tool.input_schema.get("properties", {})
            required = tool.input_schema.get("required", [])

            args_list = []
            for name, spec in props.items():
                req = "(required)" if name in required else "(optional)"
                desc = spec.get("description", "")
                type_ = spec.get("type", "any")
                args_list.append(f"  - {name}: {type_} {req} - {desc}")

            arguments = "\n".join(args_list) if args_list else "  (no arguments)"

            sections.append(
                TOOLS_SECTION_TEMPLATE.format(
                    name=tool.name,
                    description=tool.description,
                    arguments=arguments,
                )
            )

        return "\n".join(sections) if sections else "No tools available."

    def _build_entity_context(self, history: list["Frame"]) -> str:
        """
        Build entity context from frame metadata.

        This extracts plane_context (projects, users) from enriched frames
        and formats them for the LLM. This is the "Context Handoff" that
        enables the agent to know valid entity names/IDs.

        The context flows through Frame.metadata (ADR-004 compliant).
        """
        # Find plane_context in any frame's metadata
        plane_context = None
        for frame in history:
            if hasattr(frame, "metadata") and "plane_context" in frame.metadata:
                plane_context = frame.metadata["plane_context"]
                break

        if not plane_context:
            return ""

        lines = ["**Valid Entities (use these exact names when calling tools):**"]

        # Add projects
        projects = plane_context.get("projects", [])
        if projects:
            lines.append("\nProjects:")
            for p in projects[:10]:  # Limit to 10 for prompt size
                name = p.get("name", "")
                identifier = p.get("identifier", "")
                if identifier:
                    lines.append(f"  - {name} ({identifier})")
                else:
                    lines.append(f"  - {name}")

        # Add users
        users = plane_context.get("users", [])
        if users:
            lines.append("\nTeam Members:")
            for u in users[:10]:  # Limit to 10 for prompt size
                name = u.get("name", "")
                email = u.get("email", "")
                if email:
                    lines.append(f"  - {name} ({email})")
                else:
                    lines.append(f"  - {name}")

        # Add cache health status for observability
        cache_healthy = plane_context.get("cache_healthy", False)
        if not cache_healthy:
            cache_error = plane_context.get("cache_error", "Unknown")
            lines.append(f"\n⚠️ Entity cache is degraded: {cache_error}")
            lines.append("Some entity names may not resolve correctly.")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _build_context_section(self, history: list["Frame"]) -> str:
        """Build context from frame history."""
        if not history:
            return "No previous context."

        lines = []
        for frame in history[-5:]:  # Last 5 frames for context window
            if hasattr(frame, "frame_type"):
                if frame.frame_type == "tool_result":
                    lines.append(
                        f"[Tool Result] {frame.tool_name}: "
                        f"{'Success' if frame.success else 'Failed'} - "
                        f"{frame.result_text[:100]}..."
                    )
                elif frame.frame_type == "tool_call":
                    lines.append(
                        f"[Tool Call] {frame.tool_name}: "
                        f"{json.dumps(frame.tool_arguments)}"
                    )
                elif frame.frame_type == "extraction":
                    if hasattr(frame, "summary"):
                        lines.append(f"[Extraction] Summary: {frame.summary}")
                elif frame.frame_type == "transcription":
                    if hasattr(frame, "text"):
                        lines.append(f"[Transcription] {frame.text[:200]}...")
                else:
                    lines.append(f"[{frame.frame_type}] Frame ID: {frame.id}")

        return "\n".join(lines) if lines else "No relevant context."

    def _build_user_message(
        self,
        goal: str,
        history: list["Frame"],
        iteration: int,
    ) -> str:
        """Build the user message with goal and iteration."""
        parts = [f"Goal: {goal}"]

        if iteration > 0:
            parts.append(f"This is iteration {iteration + 1}.")

            # Add last tool result if available
            for frame in reversed(history):
                if hasattr(frame, "frame_type") and frame.frame_type == "tool_result":
                    if frame.success:
                        parts.append(f"Last tool result: {frame.result_text}")
                    else:
                        parts.append(f"Last tool failed: {frame.error_message}")
                    break

        parts.append("What should I do next?")
        return "\n\n".join(parts)

    def _parse_response(
        self,
        content: str,
        goal: str,
        iteration: int,
        history: list["Frame"],
    ) -> PlanFrame | ToolCallFrame | AgentResponseFrame:
        """
        Parse and validate LLM response into appropriate frame.

        Uses Pydantic validation to ensure the response conforms to
        expected schema, preventing malformed responses from causing
        unexpected behavior.
        """
        try:
            # Parse JSON response
            data = json.loads(content)

            # Validate with Pydantic schema
            try:
                decision = LLMDecision.model_validate(data)
            except ValidationError as e:
                logger.warning(
                    f"[agent_processor] LLM response validation failed: {e.error_count()} errors"
                )
                logger.debug(f"[agent_processor] Validation errors: {e.errors()}")
                # Fall back to raw data if validation fails
                decision = LLMDecision(
                    action=data.get("action", ""),
                    tool=data.get("tool"),
                    arguments=data.get("arguments", {}),
                    message=data.get("message"),
                    question=data.get("question"),
                    reasoning=data.get("reasoning", ""),
                )

            action = decision.action.lower()

            if action == "tool_call" and decision.tool:
                tool_name = decision.tool
                arguments = decision.arguments
                reasoning = decision.reasoning

                # Validate tool exists
                if not self._tools.get(tool_name):
                    logger.warning(
                        f"[agent_processor] Unknown tool: {tool_name}. "
                        f"Returning error response."
                    )
                    return AgentResponseFrame(
                        response_text=f"I tried to use an unknown tool: {tool_name}",
                        goal_achieved=False,
                        iterations_used=iteration,
                        failure_reason=f"Unknown tool: {tool_name}",
                    )

                logger.info(
                    f"[agent_processor] Decided to call tool: {tool_name}"
                )

                return ToolCallFrame(
                    tool_name=tool_name,
                    tool_arguments=arguments,
                    reasoning=reasoning,
                    iteration=iteration,
                )

            elif action == "respond" and decision.message:
                message = decision.message
                reasoning = decision.reasoning

                logger.info("[agent_processor] Decided to respond to user")

                # Collect tools called from history
                tools_called = tuple(
                    f.tool_name
                    for f in history
                    if hasattr(f, "frame_type") and f.frame_type == "tool_call"
                )

                return AgentResponseFrame(
                    response_text=message,
                    goal_achieved=True,
                    iterations_used=iteration + 1,
                    tools_called=tools_called,
                    reasoning_trace=reasoning,
                )

            elif action == "ask_user" and decision.question:
                question = decision.question
                reasoning = decision.reasoning

                logger.info("[agent_processor] Decided to ask user for clarification")

                return AgentResponseFrame(
                    response_text=question,
                    goal_achieved=False,
                    iterations_used=iteration + 1,
                    failure_reason="Need clarification",
                    reasoning_trace=reasoning,
                )

            else:
                # Unknown action or missing required fields - return planning frame
                logger.warning(
                    f"[agent_processor] Unknown/incomplete action: {action}. "
                    f"Returning plan frame."
                )
                return PlanFrame(
                    goal=goal,
                    reasoning=f"Analyzing response: {content[:100]}",
                    next_action=AgentAction.PLAN,
                    iteration=iteration,
                    max_iterations=self._max_iterations,
                )

        except json.JSONDecodeError as e:
            logger.error(f"[agent_processor] Failed to parse JSON: {e}")
            logger.debug(f"[agent_processor] Raw content: {content}")

            # Try to extract useful information from non-JSON response
            return AgentResponseFrame(
                response_text=content[:500] if content else "Unable to process request",
                goal_achieved=False,
                iterations_used=iteration,
                failure_reason=f"Invalid JSON response: {e}",
            )

        except Exception as e:
            logger.error(f"[agent_processor] Error parsing response: {e}")
            return AgentResponseFrame(
                response_text=f"Error processing: {e}",
                goal_achieved=False,
                iterations_used=iteration,
                failure_reason=str(e),
            )
