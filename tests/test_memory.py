"""
Tests for Memory Persistence (Phase 2.6).

Tests cover:
- Message and Conversation dataclasses
- InMemoryStorage operations
- RedisMemory operations (mocked)
- MemoryManager high-level API
- Serialization/deserialization
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from knomly.agent.memory import (
    Conversation,
    InMemoryStorage,
    MemoryManager,
    Message,
    RedisMemory,
    create_memory,
)

# =============================================================================
# Message Tests
# =============================================================================


class TestMessage:
    """Tests for Message dataclass."""

    def test_create_message(self):
        """Create a basic message."""
        msg = Message(role="user", content="Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"
        assert msg.id is not None
        assert msg.timestamp is not None
        assert msg.metadata == {}

    def test_message_is_immutable(self):
        """Message should be frozen (immutable)."""
        msg = Message(role="user", content="Hello!")

        with pytest.raises(Exception):  # FrozenInstanceError
            msg.content = "Changed!"

    def test_message_with_metadata(self):
        """Create message with metadata."""
        msg = Message(
            role="assistant",
            content="Done!",
            metadata={"tool_calls": ["plane_create_task"]},
        )

        assert msg.metadata["tool_calls"] == ["plane_create_task"]

    def test_message_to_dict(self):
        """Convert message to dict."""
        msg = Message(
            role="user",
            content="Hello!",
            metadata={"source": "whatsapp"},
        )

        data = msg.to_dict()

        assert data["role"] == "user"
        assert data["content"] == "Hello!"
        assert data["id"] == msg.id
        assert "timestamp" in data
        assert data["metadata"]["source"] == "whatsapp"

    def test_message_from_dict(self):
        """Create message from dict."""
        data = {
            "role": "assistant",
            "content": "Task created!",
            "id": "msg-123",
            "timestamp": "2026-01-03T10:00:00+00:00",
            "metadata": {"success": True},
        }

        msg = Message.from_dict(data)

        assert msg.role == "assistant"
        assert msg.content == "Task created!"
        assert msg.id == "msg-123"
        assert msg.metadata["success"] is True

    def test_message_to_llm_format(self):
        """Convert to LLM format (role + content only)."""
        msg = Message(
            role="user",
            content="Create a task",
            metadata={"extra": "ignored"},
        )

        llm_format = msg.to_llm_format()

        assert llm_format == {"role": "user", "content": "Create a task"}

    def test_message_roundtrip(self):
        """Serialize and deserialize message."""
        original = Message(
            role="tool",
            content="Tool result",
            metadata={"tool_name": "test"},
        )

        data = original.to_dict()
        restored = Message.from_dict(data)

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.id == original.id
        assert restored.metadata == original.metadata


# =============================================================================
# Conversation Tests
# =============================================================================


class TestConversation:
    """Tests for Conversation dataclass."""

    def test_create_conversation(self):
        """Create empty conversation."""
        conv = Conversation(session_id="user-123")

        assert conv.session_id == "user-123"
        assert len(conv.messages) == 0
        assert conv.created_at is not None
        assert conv.updated_at is not None

    def test_add_message(self):
        """Add message to conversation."""
        conv = Conversation(session_id="user-123")
        msg = Message(role="user", content="Hello!")

        initial_updated = conv.updated_at
        conv.add_message(msg)

        assert len(conv.messages) == 1
        assert conv.messages[0] == msg
        assert conv.updated_at >= initial_updated

    def test_conversation_to_dict(self):
        """Convert conversation to dict."""
        conv = Conversation(session_id="user-123")
        conv.add_message(Message(role="user", content="Hello!"))
        conv.add_message(Message(role="assistant", content="Hi!"))

        data = conv.to_dict()

        assert data["session_id"] == "user-123"
        assert len(data["messages"]) == 2
        assert "created_at" in data
        assert "updated_at" in data

    def test_conversation_from_dict(self):
        """Create conversation from dict."""
        data = {
            "session_id": "user-123",
            "messages": [
                {"role": "user", "content": "Hello!", "timestamp": "2026-01-03T10:00:00+00:00"},
                {"role": "assistant", "content": "Hi!", "timestamp": "2026-01-03T10:00:01+00:00"},
            ],
            "created_at": "2026-01-03T10:00:00+00:00",
            "updated_at": "2026-01-03T10:00:01+00:00",
            "metadata": {"user_phone": "+123"},
        }

        conv = Conversation.from_dict(data)

        assert conv.session_id == "user-123"
        assert len(conv.messages) == 2
        assert conv.messages[0].role == "user"
        assert conv.metadata["user_phone"] == "+123"

    def test_conversation_to_llm_messages(self):
        """Convert conversation to LLM message format."""
        conv = Conversation(session_id="user-123")
        conv.add_message(Message(role="user", content="Create task"))
        conv.add_message(Message(role="assistant", content="Done!"))

        llm_messages = conv.to_llm_messages()

        assert llm_messages == [
            {"role": "user", "content": "Create task"},
            {"role": "assistant", "content": "Done!"},
        ]


# =============================================================================
# InMemoryStorage Tests
# =============================================================================


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    @pytest.mark.asyncio
    async def test_add_and_get_message(self):
        """Add and retrieve messages."""
        memory = InMemoryStorage()

        msg = await memory.add_message("session-1", "user", "Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"

        messages = await memory.get_messages("session-1")

        assert len(messages) == 1
        assert messages[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_multiple_messages(self):
        """Add multiple messages to session."""
        memory = InMemoryStorage()

        await memory.add_message("session-1", "user", "Message 1")
        await memory.add_message("session-1", "assistant", "Message 2")
        await memory.add_message("session-1", "user", "Message 3")

        messages = await memory.get_messages("session-1")

        assert len(messages) == 3
        assert messages[0].content == "Message 1"
        assert messages[2].content == "Message 3"

    @pytest.mark.asyncio
    async def test_multiple_sessions(self):
        """Messages are isolated by session."""
        memory = InMemoryStorage()

        await memory.add_message("session-1", "user", "Session 1 message")
        await memory.add_message("session-2", "user", "Session 2 message")

        messages1 = await memory.get_messages("session-1")
        messages2 = await memory.get_messages("session-2")

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0].content == "Session 1 message"
        assert messages2[0].content == "Session 2 message"

    @pytest.mark.asyncio
    async def test_get_messages_with_limit(self):
        """Get messages with limit."""
        memory = InMemoryStorage()

        for i in range(5):
            await memory.add_message("session-1", "user", f"Message {i}")

        messages = await memory.get_messages("session-1", limit=3)

        assert len(messages) == 3
        # Should get most recent
        assert messages[0].content == "Message 2"
        assert messages[2].content == "Message 4"

    @pytest.mark.asyncio
    async def test_get_messages_nonexistent_session(self):
        """Get messages from non-existent session."""
        memory = InMemoryStorage()

        messages = await memory.get_messages("nonexistent")

        assert messages == ()

    @pytest.mark.asyncio
    async def test_get_conversation(self):
        """Get full conversation object."""
        memory = InMemoryStorage()

        await memory.add_message("session-1", "user", "Hello!")

        conv = await memory.get_conversation("session-1")

        assert conv is not None
        assert conv.session_id == "session-1"
        assert len(conv.messages) == 1

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Clear session messages."""
        memory = InMemoryStorage()

        await memory.add_message("session-1", "user", "Hello!")
        await memory.clear("session-1")

        messages = await memory.get_messages("session-1")

        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_delete_message(self):
        """Delete specific message."""
        memory = InMemoryStorage()

        msg = await memory.add_message("session-1", "user", "Hello!")
        await memory.add_message("session-1", "user", "World!")

        result = await memory.delete_message("session-1", msg.id)

        assert result is True
        messages = await memory.get_messages("session-1")
        assert len(messages) == 1
        assert messages[0].content == "World!"

    @pytest.mark.asyncio
    async def test_delete_message_not_found(self):
        """Delete non-existent message."""
        memory = InMemoryStorage()

        await memory.add_message("session-1", "user", "Hello!")

        result = await memory.delete_message("session-1", "nonexistent")

        assert result is False

    @pytest.mark.asyncio
    async def test_max_messages_eviction(self):
        """Old messages are evicted when max reached."""
        memory = InMemoryStorage(max_messages_per_session=3)

        for i in range(5):
            await memory.add_message("session-1", "user", f"Message {i}")

        messages = await memory.get_messages("session-1")

        assert len(messages) == 3
        # Should keep most recent
        assert messages[0].content == "Message 2"
        assert messages[2].content == "Message 4"

    @pytest.mark.asyncio
    async def test_add_message_with_metadata(self):
        """Add message with metadata."""
        memory = InMemoryStorage()

        msg = await memory.add_message(
            "session-1",
            "user",
            "Hello!",
            metadata={"source": "whatsapp", "phone": "+123"},
        )

        assert msg.metadata["source"] == "whatsapp"
        assert msg.metadata["phone"] == "+123"

    def test_session_count(self):
        """Get number of active sessions."""
        memory = InMemoryStorage()

        assert memory.session_count() == 0


# =============================================================================
# RedisMemory Tests (Mocked)
# =============================================================================


class TestRedisMemory:
    """Tests for RedisMemory with mocked Redis."""

    @pytest.fixture
    def mock_redis_client(self):
        """Create mock Redis client."""
        client = AsyncMock()
        client.get = AsyncMock(return_value=None)
        client.set = AsyncMock()
        client.setex = AsyncMock()
        client.delete = AsyncMock()
        return client

    @pytest.mark.asyncio
    async def test_add_message_new_session(self, mock_redis_client):
        """Add message to new session."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory()

            msg = await memory.add_message("session-1", "user", "Hello!")

            assert msg.role == "user"
            assert msg.content == "Hello!"
            mock_redis_client.setex.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_message_existing_session(self, mock_redis_client):
        """Add message to existing session."""
        # Mock existing conversation
        existing_conv = Conversation(session_id="session-1")
        existing_conv.add_message(Message(role="user", content="Previous"))
        mock_redis_client.get = AsyncMock(return_value=json.dumps(existing_conv.to_dict()))

        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory()

            await memory.add_message("session-1", "assistant", "New!")

            # Should have stored updated conversation
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            stored_data = json.loads(call_args[0][2])
            assert len(stored_data["messages"]) == 2

    @pytest.mark.asyncio
    async def test_get_messages(self, mock_redis_client):
        """Get messages from session."""
        conv = Conversation(session_id="session-1")
        conv.add_message(Message(role="user", content="Hello!"))
        mock_redis_client.get = AsyncMock(return_value=json.dumps(conv.to_dict()))

        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory()

            messages = await memory.get_messages("session-1")

            assert len(messages) == 1
            assert messages[0].content == "Hello!"

    @pytest.mark.asyncio
    async def test_get_messages_nonexistent(self, mock_redis_client):
        """Get messages from non-existent session."""
        mock_redis_client.get = AsyncMock(return_value=None)

        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory()

            messages = await memory.get_messages("nonexistent")

            assert messages == ()

    @pytest.mark.asyncio
    async def test_clear_session(self, mock_redis_client):
        """Clear session."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory()

            await memory.clear("session-1")

            mock_redis_client.delete.assert_called_once()

    @pytest.mark.asyncio
    async def test_ttl_applied(self, mock_redis_client):
        """TTL is applied to stored conversations."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory(ttl_seconds=3600)

            await memory.add_message("session-1", "user", "Hello!")

            # setex is called with TTL
            mock_redis_client.setex.assert_called_once()
            call_args = mock_redis_client.setex.call_args
            assert call_args[0][1] == 3600  # TTL

    @pytest.mark.asyncio
    async def test_no_ttl(self, mock_redis_client):
        """No TTL when set to 0."""
        with patch("redis.asyncio.from_url", return_value=mock_redis_client):
            memory = RedisMemory(ttl_seconds=0)

            await memory.add_message("session-1", "user", "Hello!")

            # set is called instead of setex
            mock_redis_client.set.assert_called_once()
            mock_redis_client.setex.assert_not_called()


# =============================================================================
# MemoryManager Tests
# =============================================================================


class TestMemoryManager:
    """Tests for MemoryManager."""

    @pytest.mark.asyncio
    async def test_add_user_message(self):
        """Add user message via manager."""
        manager = MemoryManager()

        msg = await manager.add_user_message("session-1", "Hello!")

        assert msg.role == "user"
        assert msg.content == "Hello!"

    @pytest.mark.asyncio
    async def test_add_assistant_message(self):
        """Add assistant message via manager."""
        manager = MemoryManager()

        msg = await manager.add_assistant_message(
            "session-1",
            "Task created!",
            tool_calls=["plane_create_task"],
        )

        assert msg.role == "assistant"
        assert msg.metadata["tool_calls"] == ["plane_create_task"]

    @pytest.mark.asyncio
    async def test_add_tool_result(self):
        """Add tool result via manager."""
        manager = MemoryManager()

        msg = await manager.add_tool_result(
            "session-1",
            tool_name="plane_create_task",
            result="Created task #123",
            success=True,
        )

        assert msg.role == "tool"
        assert msg.metadata["tool_name"] == "plane_create_task"
        assert msg.metadata["success"] is True

    @pytest.mark.asyncio
    async def test_get_llm_messages(self):
        """Get messages in LLM format."""
        manager = MemoryManager()

        await manager.add_user_message("session-1", "Create task")
        await manager.add_assistant_message("session-1", "Done!")

        llm_messages = await manager.get_llm_messages("session-1")

        assert len(llm_messages) == 2
        assert llm_messages[0] == {"role": "user", "content": "Create task"}
        assert llm_messages[1] == {"role": "assistant", "content": "Done!"}

    @pytest.mark.asyncio
    async def test_get_llm_messages_with_system(self):
        """Get messages with system prompt."""
        manager = MemoryManager(system_prompt="You are a helpful assistant.")

        await manager.add_user_message("session-1", "Hello!")

        llm_messages = await manager.get_llm_messages("session-1")

        assert len(llm_messages) == 2
        assert llm_messages[0]["role"] == "system"
        assert llm_messages[0]["content"] == "You are a helpful assistant."
        assert llm_messages[1]["role"] == "user"

    @pytest.mark.asyncio
    async def test_get_llm_messages_without_system(self):
        """Get messages without system prompt."""
        manager = MemoryManager(system_prompt="You are helpful.")

        await manager.add_user_message("session-1", "Hello!")

        llm_messages = await manager.get_llm_messages("session-1", include_system=False)

        assert len(llm_messages) == 1
        assert llm_messages[0]["role"] == "user"

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Clear session via manager."""
        manager = MemoryManager()

        await manager.add_user_message("session-1", "Hello!")
        await manager.clear_session("session-1")

        messages = await manager.get_llm_messages("session-1", include_system=False)
        assert len(messages) == 0

    @pytest.mark.asyncio
    async def test_custom_memory_backend(self):
        """Use custom memory backend."""
        memory = InMemoryStorage(max_messages_per_session=5)
        manager = MemoryManager(memory=memory)

        for i in range(10):
            await manager.add_user_message("session-1", f"Message {i}")

        messages = await manager.get_llm_messages("session-1", include_system=False)
        assert len(messages) == 5


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateMemory:
    """Tests for create_memory factory function."""

    def test_create_inmemory(self):
        """Create in-memory storage."""
        memory = create_memory("inmemory")

        assert isinstance(memory, InMemoryStorage)

    def test_create_inmemory_with_options(self):
        """Create in-memory storage with options."""
        memory = create_memory("inmemory", max_messages_per_session=50)

        assert memory._max_messages == 50

    def test_create_redis(self):
        """Create Redis storage."""
        memory = create_memory("redis", redis_url="redis://localhost:6379")

        assert isinstance(memory, RedisMemory)
        assert memory._redis_url == "redis://localhost:6379"

    def test_create_redis_with_options(self):
        """Create Redis storage with options."""
        memory = create_memory(
            "redis",
            redis_url="redis://localhost:6379",
            key_prefix="myapp:memory",
            ttl_seconds=3600,
        )

        assert memory._key_prefix == "myapp:memory"
        assert memory._ttl == 3600

    def test_create_unknown_backend(self):
        """Unknown backend raises error."""
        with pytest.raises(ValueError, match="Unknown backend"):
            create_memory("unknown")


# =============================================================================
# Integration Tests
# =============================================================================


class TestMemoryIntegration:
    """Integration tests for memory in typical workflows."""

    @pytest.mark.asyncio
    async def test_complete_conversation_flow(self):
        """Test complete conversation workflow."""
        manager = MemoryManager(system_prompt="You are Knomly, a voice assistant.")

        # User sends voice note
        await manager.add_user_message(
            "user-123",
            "Create a task for the mobile app feature",
            source="whatsapp",
            phone="+1234567890",
        )

        # Agent processes and uses tool
        await manager.add_tool_result(
            "user-123",
            tool_name="plane_create_task",
            result='{"id": "task-456", "name": "Mobile app feature"}',
            success=True,
        )

        # Agent responds
        await manager.add_assistant_message(
            "user-123",
            "I've created task #456 for the mobile app feature.",
            tool_calls=["plane_create_task"],
        )

        # Get conversation for LLM
        messages = await manager.get_llm_messages("user-123")

        assert len(messages) == 4  # system + user + tool + assistant
        assert messages[0]["role"] == "system"
        assert "voice assistant" in messages[0]["content"]

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self):
        """Test multi-turn conversation."""
        manager = MemoryManager()

        # Turn 1
        await manager.add_user_message("user-123", "What tasks are open?")
        await manager.add_assistant_message("user-123", "You have 3 open tasks...")

        # Turn 2
        await manager.add_user_message("user-123", "Close the first one")
        await manager.add_assistant_message("user-123", "Done, task #1 is now closed.")

        # Full history available
        conv = await manager.get_conversation("user-123")
        assert len(conv.messages) == 4

        # LLM sees full context
        messages = await manager.get_llm_messages("user-123")
        assert len(messages) == 4

    @pytest.mark.asyncio
    async def test_session_isolation(self):
        """Different users have isolated sessions."""
        manager = MemoryManager()

        await manager.add_user_message("user-1", "User 1 message")
        await manager.add_user_message("user-2", "User 2 message")

        messages1 = await manager.get_llm_messages("user-1", include_system=False)
        messages2 = await manager.get_llm_messages("user-2", include_system=False)

        assert len(messages1) == 1
        assert len(messages2) == 1
        assert messages1[0]["content"] == "User 1 message"
        assert messages2[0]["content"] == "User 2 message"


# =============================================================================
# Frame Persistence Tests (v2.2)
# =============================================================================


class TestFrameToMessage:
    """Tests for frame_to_message conversion."""

    def test_tool_call_frame(self):
        """Convert ToolCallFrame to message."""
        from knomly.agent.frames import ToolCallFrame
        from knomly.agent.memory import frame_to_message

        frame = ToolCallFrame(
            tool_name="plane_create_task",
            tool_arguments={"name": "Test task"},
            reasoning="Need to create a task",
            iteration=0,
        )

        msg = frame_to_message(frame)

        assert msg.role == "assistant"
        assert "plane_create_task" in msg.content
        assert msg.metadata["frame_type"] == "tool_call"
        assert msg.metadata["serialized"] is True
        assert "frame_data" in msg.metadata

    def test_tool_result_frame_success(self):
        """Convert successful ToolResultFrame to message."""
        from knomly.agent.frames import ToolResultFrame
        from knomly.agent.memory import frame_to_message

        frame = ToolResultFrame(
            tool_name="plane_create_task",
            success=True,
            result_text="Created task #123",
            error_message="",
        )

        msg = frame_to_message(frame)

        assert msg.role == "tool"
        assert "Created task #123" in msg.content
        assert msg.metadata["frame_type"] == "tool_result"

    def test_tool_result_frame_failure(self):
        """Convert failed ToolResultFrame to message."""
        from knomly.agent.frames import ToolResultFrame
        from knomly.agent.memory import frame_to_message

        frame = ToolResultFrame(
            tool_name="plane_create_task",
            success=False,
            result_text="",
            error_message="Permission denied",
        )

        msg = frame_to_message(frame)

        assert msg.role == "tool"
        assert "Permission denied" in msg.content
        assert "[Tool Error]" in msg.content

    def test_agent_response_frame(self):
        """Convert AgentResponseFrame to message."""
        from knomly.agent.frames import AgentResponseFrame
        from knomly.agent.memory import frame_to_message

        frame = AgentResponseFrame(
            response_text="Task created successfully!",
            goal_achieved=True,
            iterations_used=2,
        )

        msg = frame_to_message(frame)

        assert msg.role == "assistant"
        assert msg.content == "Task created successfully!"
        assert msg.metadata["frame_type"] == "agent_response"

    def test_plan_frame(self):
        """Convert PlanFrame to message."""
        from knomly.agent.frames import AgentAction, PlanFrame
        from knomly.agent.memory import frame_to_message

        frame = PlanFrame(
            goal="Create a task",
            reasoning="Analyzing project requirements",
            next_action=AgentAction.TOOL_CALL,
            iteration=1,
        )

        msg = frame_to_message(frame)

        assert msg.role == "assistant"
        assert "[Plan]" in msg.content
        assert "Analyzing" in msg.content

    def test_transcription_frame(self):
        """Convert TranscriptionFrame to message."""
        from knomly.agent.memory import frame_to_message
        from knomly.pipeline.frames import TranscriptionFrame

        frame = TranscriptionFrame(
            original_text="crear una tarea",
            english_text="create a task",
            detected_language="es",
            confidence=0.95,
            sender_phone="+1234567890",
        )

        msg = frame_to_message(frame)

        assert msg.role == "user"
        assert msg.content == "create a task"


class TestMessageToFrame:
    """Tests for message_to_frame restoration."""

    def test_restore_tool_call_frame(self):
        """Restore ToolCallFrame from message."""
        from knomly.agent.frames import ToolCallFrame
        from knomly.agent.memory import frame_to_message, message_to_frame

        original = ToolCallFrame(
            tool_name="plane_create_task",
            tool_arguments={"name": "Test"},
            reasoning="Testing",
            iteration=0,
        )

        msg = frame_to_message(original)
        restored = message_to_frame(msg)

        assert isinstance(restored, ToolCallFrame)
        assert restored.tool_name == "plane_create_task"
        assert restored.tool_arguments == {"name": "Test"}

    def test_restore_tool_result_frame(self):
        """Restore ToolResultFrame from message."""
        from knomly.agent.frames import ToolResultFrame
        from knomly.agent.memory import frame_to_message, message_to_frame

        original = ToolResultFrame(
            tool_name="plane_create_task",
            success=True,
            result_text="Created task #123",
            error_message="",
        )

        msg = frame_to_message(original)
        restored = message_to_frame(msg)

        assert isinstance(restored, ToolResultFrame)
        assert restored.tool_name == "plane_create_task"
        assert restored.success is True
        assert restored.result_text == "Created task #123"

    def test_restore_agent_response_frame(self):
        """Restore AgentResponseFrame from message."""
        from knomly.agent.frames import AgentResponseFrame
        from knomly.agent.memory import frame_to_message, message_to_frame

        original = AgentResponseFrame(
            response_text="Done!",
            goal_achieved=True,
            iterations_used=3,
            tools_called=("tool1", "tool2"),
        )

        msg = frame_to_message(original)
        restored = message_to_frame(msg)

        assert isinstance(restored, AgentResponseFrame)
        assert restored.response_text == "Done!"
        assert restored.goal_achieved is True
        assert restored.tools_called == ("tool1", "tool2")

    def test_restore_returns_none_for_non_serialized(self):
        """Non-serialized messages return None."""
        from knomly.agent.memory import Message, message_to_frame

        msg = Message(role="user", content="Hello!")

        restored = message_to_frame(msg)

        assert restored is None


class TestExecutionMemory:
    """Tests for ExecutionMemory high-level API."""

    @pytest.mark.asyncio
    async def test_persist_and_restore_frames(self):
        """Persist and restore frame history."""
        from knomly.agent.frames import ToolCallFrame, ToolResultFrame
        from knomly.agent.memory import ExecutionMemory

        memory = ExecutionMemory()

        # Persist frames
        frame1 = ToolCallFrame(
            tool_name="tool1",
            tool_arguments={"arg": "value"},
            reasoning="Test",
            iteration=0,
        )
        frame2 = ToolResultFrame(
            tool_name="tool1",
            success=True,
            result_text="Success!",
            error_message="",
        )

        await memory.persist_frame("session-1", frame1)
        await memory.persist_frame("session-1", frame2)

        # Restore
        history = await memory.restore_history("session-1")

        assert len(history) == 2
        assert isinstance(history[0], ToolCallFrame)
        assert isinstance(history[1], ToolResultFrame)
        assert history[0].tool_name == "tool1"
        assert history[1].result_text == "Success!"

    @pytest.mark.asyncio
    async def test_get_last_state(self):
        """Get execution state summary."""
        from knomly.agent.frames import AgentResponseFrame, ToolCallFrame, ToolResultFrame
        from knomly.agent.memory import ExecutionMemory

        memory = ExecutionMemory()

        # Persist a complete execution
        await memory.persist_frame(
            "session-1",
            ToolCallFrame(
                tool_name="tool1",
                tool_arguments={},
                reasoning="",
                iteration=0,
            ),
        )
        await memory.persist_frame(
            "session-1",
            ToolResultFrame(
                tool_name="tool1",
                success=True,
                result_text="Done!",
                error_message="",
            ),
        )
        await memory.persist_frame(
            "session-1",
            AgentResponseFrame(
                response_text="Task completed!",
                goal_achieved=True,
                iterations_used=1,
            ),
        )

        state = await memory.get_last_state("session-1")

        assert state is not None
        assert state["session_id"] == "session-1"
        assert state["iterations"] == 1
        assert state["tools_called"] == ["tool1"]
        assert state["completed"] is True
        assert state["message_count"] == 3

    @pytest.mark.asyncio
    async def test_get_last_state_nonexistent(self):
        """Get state for non-existent session."""
        from knomly.agent.memory import ExecutionMemory

        memory = ExecutionMemory()

        state = await memory.get_last_state("nonexistent")

        assert state is None

    @pytest.mark.asyncio
    async def test_clear_session(self):
        """Clear execution session."""
        from knomly.agent.frames import ToolCallFrame
        from knomly.agent.memory import ExecutionMemory

        memory = ExecutionMemory()

        await memory.persist_frame(
            "session-1",
            ToolCallFrame(
                tool_name="tool1",
                tool_arguments={},
                reasoning="",
                iteration=0,
            ),
        )

        await memory.clear_session("session-1")

        history = await memory.restore_history("session-1")
        assert history == []

    @pytest.mark.asyncio
    async def test_with_redis_storage(self):
        """ExecutionMemory works with Redis storage."""
        from knomly.agent.frames import ToolCallFrame
        from knomly.agent.memory import ExecutionMemory, InMemoryStorage

        # Use InMemoryStorage as mock for Redis
        storage = InMemoryStorage()
        memory = ExecutionMemory(storage=storage)

        await memory.persist_frame(
            "session-1",
            ToolCallFrame(
                tool_name="tool1",
                tool_arguments={},
                reasoning="",
                iteration=0,
            ),
        )

        # Verify frame was stored
        messages = await storage.get_messages("session-1")
        assert len(messages) == 1


class TestExecutorMemoryIntegration:
    """Integration tests for AgentExecutor with memory."""

    @pytest.mark.asyncio
    async def test_executor_persists_frames(self):
        """Executor persists frames when memory is provided."""
        from unittest.mock import AsyncMock

        from knomly.agent import AgentExecutor
        from knomly.agent.frames import AgentResponseFrame
        from knomly.agent.memory import ExecutionMemory, InMemoryStorage

        # Create mock processor
        mock_processor = MagicMock()
        mock_processor.decide = AsyncMock(
            return_value=AgentResponseFrame(
                response_text="Done!",
                goal_achieved=True,
                iterations_used=1,
            )
        )

        # Create mock registry
        mock_registry = MagicMock()

        # Create executor with memory
        storage = InMemoryStorage()
        memory = ExecutionMemory(storage=storage)

        executor = AgentExecutor(
            processor=mock_processor,
            tools=mock_registry,
            max_iterations=5,
            memory=memory,
        )

        # Run executor
        result = await executor.run(
            goal="Test goal",
            session_id="test-session",
        )

        # Verify frames were persisted
        messages = await storage.get_messages("test-session")
        assert len(messages) >= 1  # At least the response frame

    @pytest.mark.asyncio
    async def test_executor_generates_session_id_when_memory_enabled(self):
        """Executor generates session_id when memory enabled but no ID provided."""
        from unittest.mock import AsyncMock

        from knomly.agent import AgentExecutor
        from knomly.agent.frames import AgentResponseFrame
        from knomly.agent.memory import ExecutionMemory, InMemoryStorage

        mock_processor = MagicMock()
        mock_processor.decide = AsyncMock(
            return_value=AgentResponseFrame(
                response_text="Done!",
                goal_achieved=True,
                iterations_used=1,
            )
        )

        storage = InMemoryStorage()
        memory = ExecutionMemory(storage=storage)

        executor = AgentExecutor(
            processor=mock_processor,
            tools=MagicMock(),
            memory=memory,
        )

        # Run without explicit session_id
        await executor.run(goal="Test goal")

        # Verify session was created (storage has 1 session)
        assert storage.session_count() == 1

    @pytest.mark.asyncio
    async def test_executor_without_memory_has_no_persistence(self):
        """Executor without memory doesn't persist anything."""
        from unittest.mock import AsyncMock

        from knomly.agent import AgentExecutor
        from knomly.agent.frames import AgentResponseFrame

        mock_processor = MagicMock()
        mock_processor.decide = AsyncMock(
            return_value=AgentResponseFrame(
                response_text="Done!",
                goal_achieved=True,
                iterations_used=1,
            )
        )

        # No memory provided
        executor = AgentExecutor(
            processor=mock_processor,
            tools=MagicMock(),
        )

        # This should work without errors
        result = await executor.run(goal="Test goal")

        assert result.success is True
