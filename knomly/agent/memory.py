"""
Memory Persistence (Phase 2.6).

Memory enables agents to persist and recall conversation history
across sessions. This is essential for:
- Long-running conversations
- Multi-session workflows
- User context persistence
- Agent learning from past interactions

Design:
    - Memory Protocol defines the interface
    - Multiple backends: InMemory (test), Redis (production)
    - Messages stored as structured data (JSON serializable)
    - Support for conversation threads (session_id)

Usage:
    # Create memory instance
    memory = RedisMemory(redis_url="redis://localhost:6379")

    # Store messages
    await memory.add_message(
        session_id="user-123",
        role="user",
        content="Create a task for the mobile app",
    )

    # Recall conversation
    history = await memory.get_messages("user-123")

    # Clear session
    await memory.clear("user-123")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Literal, Protocol
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


# =============================================================================
# Message Types
# =============================================================================


MessageRole = Literal["system", "user", "assistant", "tool"]


@dataclass(frozen=True)
class Message:
    """
    A single message in conversation history.

    Attributes:
        role: The role of the message sender
        content: The message content (text or structured)
        id: Unique message identifier
        timestamp: When the message was created
        metadata: Additional context (tool calls, frame refs, etc.)

    Note:
        Messages are immutable (frozen) to ensure integrity.
    """

    role: MessageRole
    content: str
    id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "role": self.role,
            "content": self.content,
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        """Create Message from dict."""
        return cls(
            role=data["role"],
            content=data["content"],
            id=data.get("id", str(uuid4())),
            timestamp=datetime.fromisoformat(data["timestamp"])
            if "timestamp" in data
            else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )

    def to_llm_format(self) -> dict[str, str]:
        """Convert to LLM-compatible format (role + content only)."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """
    A conversation (thread) containing multiple messages.

    Attributes:
        session_id: Unique identifier for this conversation
        messages: Ordered list of messages
        created_at: When the conversation started
        updated_at: When the last message was added
        metadata: Additional conversation context
    """

    session_id: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_message(self, message: Message) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)
        self.updated_at = datetime.now(UTC)

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "session_id": self.session_id,
            "messages": [m.to_dict() for m in self.messages],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Conversation:
        """Create Conversation from dict."""
        return cls(
            session_id=data["session_id"],
            messages=[Message.from_dict(m) for m in data.get("messages", [])],
            created_at=datetime.fromisoformat(data["created_at"])
            if "created_at" in data
            else datetime.now(UTC),
            updated_at=datetime.fromisoformat(data["updated_at"])
            if "updated_at" in data
            else datetime.now(UTC),
            metadata=data.get("metadata", {}),
        )

    def to_llm_messages(self) -> list[dict[str, str]]:
        """Convert all messages to LLM format."""
        return [m.to_llm_format() for m in self.messages]


# =============================================================================
# Memory Protocol
# =============================================================================


class MemoryProtocol(Protocol):
    """
    Protocol for memory backends.

    Memory stores conversation history keyed by session_id.
    Sessions can be per-user, per-conversation, or per-task.
    """

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """
        Add a message to a session's history.

        Args:
            session_id: Conversation/session identifier
            role: Message role (system, user, assistant, tool)
            content: Message content
            metadata: Optional additional context

        Returns:
            The created Message object
        """
        ...

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> Sequence[Message]:
        """
        Get messages for a session.

        Args:
            session_id: Conversation/session identifier
            limit: Optional limit on number of messages (most recent)

        Returns:
            Sequence of messages in chronological order
        """
        ...

    async def get_conversation(self, session_id: str) -> Conversation | None:
        """
        Get the full conversation object for a session.

        Args:
            session_id: Conversation/session identifier

        Returns:
            Conversation object or None if not found
        """
        ...

    async def clear(self, session_id: str) -> None:
        """
        Clear all messages for a session.

        Args:
            session_id: Conversation/session identifier
        """
        ...

    async def delete_message(self, session_id: str, message_id: str) -> bool:
        """
        Delete a specific message.

        Args:
            session_id: Conversation/session identifier
            message_id: Message ID to delete

        Returns:
            True if deleted, False if not found
        """
        ...


# =============================================================================
# In-Memory Implementation
# =============================================================================


class InMemoryStorage:
    """
    In-memory message storage for testing and development.

    Messages are stored in a dict keyed by session_id.
    Not suitable for production (data lost on restart).

    Example:
        memory = InMemoryStorage()
        await memory.add_message("user-123", "user", "Hello!")
        messages = await memory.get_messages("user-123")
    """

    def __init__(self, max_messages_per_session: int = 100) -> None:
        """
        Initialize in-memory storage.

        Args:
            max_messages_per_session: Maximum messages to store per session
                                      (oldest are evicted when exceeded)
        """
        self._conversations: dict[str, Conversation] = {}
        self._max_messages = max_messages_per_session

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to session history."""
        # Get or create conversation
        if session_id not in self._conversations:
            self._conversations[session_id] = Conversation(session_id=session_id)

        conv = self._conversations[session_id]

        # Create message
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        # Add and evict old messages if needed
        conv.add_message(message)
        if len(conv.messages) > self._max_messages:
            conv.messages = conv.messages[-self._max_messages :]

        logger.debug(f"[memory:inmemory] Added {role} message to session {session_id}")
        return message

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> Sequence[Message]:
        """Get messages for a session."""
        conv = self._conversations.get(session_id)
        if conv is None:
            return ()

        messages = conv.messages
        if limit is not None:
            messages = messages[-limit:]

        return tuple(messages)

    async def get_conversation(self, session_id: str) -> Conversation | None:
        """Get the full conversation object."""
        return self._conversations.get(session_id)

    async def clear(self, session_id: str) -> None:
        """Clear all messages for a session."""
        if session_id in self._conversations:
            del self._conversations[session_id]
            logger.debug(f"[memory:inmemory] Cleared session {session_id}")

    async def delete_message(self, session_id: str, message_id: str) -> bool:
        """Delete a specific message."""
        conv = self._conversations.get(session_id)
        if conv is None:
            return False

        for i, msg in enumerate(conv.messages):
            if msg.id == message_id:
                conv.messages.pop(i)
                return True

        return False

    def session_count(self) -> int:
        """Get number of active sessions (for testing)."""
        return len(self._conversations)


# =============================================================================
# Redis Implementation
# =============================================================================


class RedisMemory:
    """
    Redis-backed persistent memory.

    Stores conversations in Redis with automatic expiration.
    Suitable for production use with proper Redis setup.

    Storage Format:
        - Key: f"knomly:memory:{session_id}"
        - Value: JSON serialized Conversation

    Example:
        memory = RedisMemory(redis_url="redis://localhost:6379")
        await memory.add_message("user-123", "user", "Hello!")

        # Later (even after restart)
        messages = await memory.get_messages("user-123")
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        key_prefix: str = "knomly:memory",
        ttl_seconds: int = 86400 * 7,  # 7 days default
        max_messages_per_session: int = 100,
    ) -> None:
        """
        Initialize Redis memory.

        Args:
            redis_url: Redis connection URL
            key_prefix: Prefix for all keys
            ttl_seconds: Time-to-live for conversations (0 = no expiry)
            max_messages_per_session: Maximum messages per session
        """
        self._redis_url = redis_url
        self._key_prefix = key_prefix
        self._ttl = ttl_seconds
        self._max_messages = max_messages_per_session
        self._client: Any = None  # redis.asyncio.Redis

    async def _get_client(self) -> Any:
        """Get or create Redis client."""
        if self._client is None:
            try:
                import redis.asyncio as aioredis

                self._client = aioredis.from_url(
                    self._redis_url,
                    encoding="utf-8",
                    decode_responses=True,
                )
            except ImportError:
                raise ImportError(
                    "redis package required for RedisMemory. Install with: pip install redis"
                )
        return self._client

    def _key(self, session_id: str) -> str:
        """Generate Redis key for session."""
        return f"{self._key_prefix}:{session_id}"

    async def add_message(
        self,
        session_id: str,
        role: MessageRole,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> Message:
        """Add a message to session history."""
        client = await self._get_client()
        key = self._key(session_id)

        # Get or create conversation
        conv = await self.get_conversation(session_id)
        if conv is None:
            conv = Conversation(session_id=session_id)

        # Create message
        message = Message(
            role=role,
            content=content,
            metadata=metadata or {},
        )

        # Add and evict if needed
        conv.add_message(message)
        if len(conv.messages) > self._max_messages:
            conv.messages = conv.messages[-self._max_messages :]

        # Serialize and store
        data = json.dumps(conv.to_dict())
        if self._ttl > 0:
            await client.setex(key, self._ttl, data)
        else:
            await client.set(key, data)

        logger.debug(f"[memory:redis] Added {role} message to session {session_id}")
        return message

    async def get_messages(
        self,
        session_id: str,
        limit: int | None = None,
    ) -> Sequence[Message]:
        """Get messages for a session."""
        conv = await self.get_conversation(session_id)
        if conv is None:
            return ()

        messages = conv.messages
        if limit is not None:
            messages = messages[-limit:]

        return tuple(messages)

    async def get_conversation(self, session_id: str) -> Conversation | None:
        """Get the full conversation object."""
        client = await self._get_client()
        key = self._key(session_id)

        data = await client.get(key)
        if data is None:
            return None

        try:
            return Conversation.from_dict(json.loads(data))
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning(f"[memory:redis] Failed to parse conversation: {e}")
            return None

    async def clear(self, session_id: str) -> None:
        """Clear all messages for a session."""
        client = await self._get_client()
        key = self._key(session_id)
        await client.delete(key)
        logger.debug(f"[memory:redis] Cleared session {session_id}")

    async def delete_message(self, session_id: str, message_id: str) -> bool:
        """Delete a specific message."""
        conv = await self.get_conversation(session_id)
        if conv is None:
            return False

        for i, msg in enumerate(conv.messages):
            if msg.id == message_id:
                conv.messages.pop(i)
                # Re-save
                client = await self._get_client()
                key = self._key(session_id)
                data = json.dumps(conv.to_dict())
                if self._ttl > 0:
                    await client.setex(key, self._ttl, data)
                else:
                    await client.set(key, data)
                return True

        return False

    async def close(self) -> None:
        """Close the Redis connection."""
        if self._client is not None:
            await self._client.close()
            self._client = None


# =============================================================================
# Memory Manager
# =============================================================================


class MemoryManager:
    """
    High-level memory manager for agent conversations.

    Provides convenient methods for common memory operations
    and integrates with agent execution.

    Example:
        manager = MemoryManager(memory=RedisMemory())

        # Add user message
        await manager.add_user_message(
            session_id="user-123",
            content="Create a task",
        )

        # Add assistant response
        await manager.add_assistant_message(
            session_id="user-123",
            content="I've created the task.",
            tool_calls=["plane_create_task"],
        )

        # Get conversation for LLM
        messages = await manager.get_llm_messages("user-123")
    """

    def __init__(
        self,
        memory: MemoryProtocol | None = None,
        system_prompt: str | None = None,
    ) -> None:
        """
        Initialize memory manager.

        Args:
            memory: Memory backend (defaults to InMemoryStorage)
            system_prompt: Default system prompt to prepend
        """
        self._memory = memory or InMemoryStorage()
        self._system_prompt = system_prompt

    async def add_user_message(
        self,
        session_id: str,
        content: str,
        **metadata: Any,
    ) -> Message:
        """Add a user message."""
        return await self._memory.add_message(
            session_id=session_id,
            role="user",
            content=content,
            metadata=metadata,
        )

    async def add_assistant_message(
        self,
        session_id: str,
        content: str,
        tool_calls: list[str] | None = None,
        **metadata: Any,
    ) -> Message:
        """Add an assistant message."""
        if tool_calls:
            metadata["tool_calls"] = tool_calls

        return await self._memory.add_message(
            session_id=session_id,
            role="assistant",
            content=content,
            metadata=metadata,
        )

    async def add_tool_result(
        self,
        session_id: str,
        tool_name: str,
        result: str,
        success: bool = True,
        **metadata: Any,
    ) -> Message:
        """Add a tool result message."""
        metadata["tool_name"] = tool_name
        metadata["success"] = success

        return await self._memory.add_message(
            session_id=session_id,
            role="tool",
            content=result,
            metadata=metadata,
        )

    async def get_llm_messages(
        self,
        session_id: str,
        limit: int | None = None,
        include_system: bool = True,
    ) -> list[dict[str, str]]:
        """
        Get messages formatted for LLM consumption.

        Args:
            session_id: Session identifier
            limit: Maximum messages to return
            include_system: Whether to prepend system prompt

        Returns:
            List of {role, content} dicts
        """
        messages = await self._memory.get_messages(session_id, limit)
        result = [m.to_llm_format() for m in messages]

        # Prepend system prompt if configured
        if include_system and self._system_prompt:
            result.insert(0, {"role": "system", "content": self._system_prompt})

        return result

    async def clear_session(self, session_id: str) -> None:
        """Clear a session's history."""
        await self._memory.clear(session_id)

    async def get_conversation(self, session_id: str) -> Conversation | None:
        """Get the full conversation object."""
        return await self._memory.get_conversation(session_id)


# =============================================================================
# Convenience Functions
# =============================================================================


def create_memory(
    backend: Literal["inmemory", "redis"] = "inmemory",
    **kwargs: Any,
) -> InMemoryStorage | RedisMemory:
    """
    Create a memory backend.

    Args:
        backend: "inmemory" or "redis"
        **kwargs: Backend-specific configuration

    Returns:
        Memory backend instance

    Example:
        # Development
        memory = create_memory("inmemory")

        # Production
        memory = create_memory("redis", redis_url="redis://localhost:6379")
    """
    if backend == "inmemory":
        return InMemoryStorage(**kwargs)
    elif backend == "redis":
        return RedisMemory(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Frame Persistence (v2.2)
# =============================================================================


def frame_to_message(frame: Any) -> Message:
    """
    Convert an agent Frame to a Message for persistence.

    This bridges the Frame system (agent execution) with the Message system
    (memory persistence). Each Frame type is converted to an appropriate
    message role with serialized metadata.

    Args:
        frame: Any Frame object (ToolCallFrame, ToolResultFrame, etc.)

    Returns:
        Message with serialized frame data

    Note:
        Frame data is stored in metadata["frame_data"] for full restoration.
    """
    frame_type = getattr(frame, "frame_type", "unknown")
    frame_id = str(getattr(frame, "id", ""))

    # Normalize frame_type (class name → lower case for comparison)
    frame_type_lower = frame_type.lower()

    # Determine role and content based on frame type
    if frame_type_lower == "toolcallframe" or frame_type == "tool_call":
        role = "assistant"
        content = f"[Tool Call] {frame.tool_name}: {json.dumps(frame.tool_arguments)}"
        frame_type = "tool_call"  # Normalize for storage
    elif frame_type_lower == "toolresultframe" or frame_type == "tool_result":
        role = "tool"
        if frame.success:
            content = f"[Tool Result] {frame.tool_name}: {frame.result_text}"
        else:
            content = f"[Tool Error] {frame.tool_name}: {frame.error_message}"
        frame_type = "tool_result"
    elif frame_type_lower == "agentresponseframe" or frame_type == "agent_response":
        role = "assistant"
        content = frame.response_text
        frame_type = "agent_response"
    elif frame_type_lower == "planframe" or frame_type == "plan":
        role = "assistant"
        content = f"[Plan] {frame.reasoning}"
        frame_type = "plan"
    elif frame_type_lower == "transcriptionframe" or frame_type == "transcription":
        role = "user"
        text = getattr(frame, "english_text", "") or getattr(frame, "text", "")
        content = text
        frame_type = "transcription"
    elif frame_type_lower == "extractionframe" or frame_type == "extraction":
        role = "user"
        content = f"[Context] {getattr(frame, 'summary', str(frame))}"
        frame_type = "extraction"
    elif frame_type_lower == "agentcontrolframe" or frame_type == "agent_control":
        role = "assistant"
        content = f"[Control] iteration={getattr(frame, 'iteration', 0)}"
        frame_type = "agent_control"
    else:
        role = "user"
        content = f"[{frame_type}] Frame: {frame_id}"

    # Serialize frame data for full restoration
    frame_data = _serialize_frame(frame)

    return Message(
        role=role,
        content=content,
        id=frame_id or str(uuid4()),
        metadata={
            "frame_type": frame_type,
            "frame_data": frame_data,
            "serialized": True,
        },
    )


def message_to_frame(message: Message) -> Any | None:
    """
    Restore a Frame from a persisted Message.

    Args:
        message: Message with frame_data in metadata

    Returns:
        Restored Frame object, or None if not restorable
    """
    if not message.metadata.get("serialized"):
        return None

    frame_data = message.metadata.get("frame_data")
    if not frame_data:
        return None

    return _deserialize_frame(frame_data)


def _serialize_frame(frame: Any) -> dict[str, Any]:
    """Serialize a Frame to a JSON-compatible dict."""
    frame_type = getattr(frame, "frame_type", "unknown")
    frame_type_lower = frame_type.lower()
    frame_id = str(getattr(frame, "id", ""))
    created_at = getattr(frame, "created_at", None)

    # Normalize frame type for storage
    if frame_type_lower == "toolcallframe":
        normalized_type = "tool_call"
    elif frame_type_lower == "toolresultframe":
        normalized_type = "tool_result"
    elif frame_type_lower == "agentresponseframe":
        normalized_type = "agent_response"
    elif frame_type_lower == "planframe":
        normalized_type = "plan"
    elif frame_type_lower == "transcriptionframe":
        normalized_type = "transcription"
    elif frame_type_lower == "agentcontrolframe":
        normalized_type = "agent_control"
    else:
        normalized_type = frame_type

    base = {
        "type": normalized_type,
        "id": frame_id,
        "created_at": created_at.isoformat() if created_at else None,
    }

    if normalized_type == "tool_call":
        base.update(
            {
                "tool_name": frame.tool_name,
                "tool_arguments": frame.tool_arguments,
                "reasoning": getattr(frame, "reasoning", ""),
                "iteration": getattr(frame, "iteration", 0),
            }
        )
    elif normalized_type == "tool_result":
        base.update(
            {
                "tool_name": frame.tool_name,
                "success": frame.success,
                "result_text": frame.result_text,
                "error_message": getattr(frame, "error_message", ""),
                "execution_time_ms": getattr(frame, "execution_time_ms", 0),
                "tool_call_frame_id": getattr(frame, "tool_call_frame_id", ""),
            }
        )
    elif normalized_type == "agent_response":
        base.update(
            {
                "response_text": frame.response_text,
                "goal_achieved": frame.goal_achieved,
                "iterations_used": getattr(frame, "iterations_used", 0),
                "tools_called": list(getattr(frame, "tools_called", ())),
                "reasoning_trace": getattr(frame, "reasoning_trace", ""),
            }
        )
    elif normalized_type == "plan":
        base.update(
            {
                "goal": getattr(frame, "goal", ""),
                "reasoning": frame.reasoning,
                "next_action": str(getattr(frame, "next_action", "")),
                "iteration": getattr(frame, "iteration", 0),
            }
        )
    elif normalized_type == "transcription":
        base.update(
            {
                "original_text": getattr(frame, "original_text", ""),
                "english_text": getattr(frame, "english_text", ""),
                "detected_language": getattr(frame, "detected_language", ""),
                "confidence": getattr(frame, "confidence", 0.0),
                "sender_phone": getattr(frame, "sender_phone", "") or "",
                "metadata": getattr(frame, "metadata", {}),
            }
        )
    else:
        # Generic fallback - store what we can
        base["metadata"] = getattr(frame, "metadata", {})

    return base


def _deserialize_frame(data: dict[str, Any]) -> Any | None:
    """Deserialize a dict back to a Frame object."""
    from knomly.agent.frames import (
        AgentAction,
        AgentResponseFrame,
        PlanFrame,
        ToolCallFrame,
        ToolResultFrame,
    )
    from knomly.pipeline.frames import TranscriptionFrame

    frame_type = data.get("type")

    try:
        if frame_type == "tool_call":
            return ToolCallFrame(
                tool_name=data["tool_name"],
                tool_arguments=data["tool_arguments"],
                reasoning=data.get("reasoning", ""),
                iteration=data.get("iteration", 0),
            )
        elif frame_type == "tool_result":
            return ToolResultFrame(
                tool_name=data["tool_name"],
                success=data["success"],
                result_text=data["result_text"],
                error_message=data.get("error_message", ""),
                execution_time_ms=data.get("execution_time_ms", 0),
                tool_call_frame_id=data.get("tool_call_frame_id", ""),
            )
        elif frame_type == "agent_response":
            return AgentResponseFrame(
                response_text=data["response_text"],
                goal_achieved=data["goal_achieved"],
                iterations_used=data.get("iterations_used", 0),
                tools_called=tuple(data.get("tools_called", ())),
                reasoning_trace=data.get("reasoning_trace", ""),
            )
        elif frame_type == "plan":
            action_str = data.get("next_action", "plan")
            try:
                next_action = AgentAction[action_str.upper()]
            except (KeyError, AttributeError):
                next_action = AgentAction.PLAN

            return PlanFrame(
                goal=data.get("goal", ""),
                reasoning=data["reasoning"],
                next_action=next_action,
                iteration=data.get("iteration", 0),
            )
        elif frame_type == "transcription":
            return TranscriptionFrame(
                original_text=data.get("original_text", ""),
                english_text=data.get("english_text", ""),
                detected_language=data.get("detected_language", ""),
                confidence=data.get("confidence", 0.0),
                sender_phone=data.get("sender_phone", ""),
                metadata=data.get("metadata", {}),
            )
        else:
            logger.warning(f"[memory] Unknown frame type: {frame_type}")
            return None

    except Exception as e:
        logger.error(f"[memory] Failed to deserialize frame: {e}")
        return None


class ExecutionMemory:
    """
    High-level memory manager for agent executions.

    Bridges the gap between AgentExecutor (frames) and Memory (messages).
    Supports persisting and resuming agent execution state.

    Example:
        memory = ExecutionMemory(
            storage=RedisMemory(redis_url="redis://localhost:6379"),
        )

        # During execution
        await memory.persist_frame(session_id="exec-123", frame=tool_call_frame)

        # Resume after restart
        history = await memory.restore_history(session_id="exec-123")
        # Continue execution with restored history
    """

    def __init__(
        self,
        storage: MemoryProtocol | None = None,
    ) -> None:
        """
        Initialize execution memory.

        Args:
            storage: Memory backend (defaults to InMemoryStorage)
        """
        self._storage = storage or InMemoryStorage()

    async def persist_frame(self, session_id: str, frame: Any) -> Message:
        """
        Persist a frame to memory.

        Args:
            session_id: Execution session identifier
            frame: Frame to persist

        Returns:
            The created Message
        """
        message = frame_to_message(frame)
        await self._storage.add_message(
            session_id=session_id,
            role=message.role,
            content=message.content,
            metadata=message.metadata,
        )
        logger.debug(
            f"[execution_memory] Persisted {frame.frame_type} frame to session {session_id}"
        )
        return message

    async def restore_history(self, session_id: str) -> list[Any]:
        """
        Restore frame history from memory.

        Args:
            session_id: Execution session identifier

        Returns:
            List of restored Frame objects
        """
        messages = await self._storage.get_messages(session_id)
        frames = []

        for msg in messages:
            frame = message_to_frame(msg)
            if frame is not None:
                frames.append(frame)
            else:
                logger.debug(f"[execution_memory] Could not restore frame from message: {msg.id}")

        logger.info(f"[execution_memory] Restored {len(frames)} frames from session {session_id}")
        return frames

    async def get_last_state(self, session_id: str) -> dict[str, Any] | None:
        """
        Get the last execution state for a session.

        Returns:
            Dict with iteration count, tools_called, etc. or None if no session
        """
        conversation = await self._storage.get_conversation(session_id)
        if conversation is None:
            return None

        messages = conversation.messages
        if not messages:
            return None

        # Count iterations (tool_call frames)
        iterations = sum(1 for m in messages if m.metadata.get("frame_type") == "tool_call")

        # Get tools called
        tools_called = [
            m.metadata.get("frame_data", {}).get("tool_name", "")
            for m in messages
            if m.metadata.get("frame_type") == "tool_call"
        ]

        # Check if completed
        completed = any(
            m.metadata.get("frame_type") == "agent_response"
            and m.metadata.get("frame_data", {}).get("goal_achieved", False)
            for m in messages
        )

        return {
            "session_id": session_id,
            "iterations": iterations,
            "tools_called": tools_called,
            "completed": completed,
            "message_count": len(messages),
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
        }

    async def clear_session(self, session_id: str) -> None:
        """Clear a session's execution history."""
        await self._storage.clear(session_id)
        logger.info(f"[execution_memory] Cleared session {session_id}")
