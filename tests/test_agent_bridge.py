"""
Integration Tests for v1 → v2 Agent Bridge.

Tests cover:
- AgentBridgeProcessor initialization and processing
- Context handoff from v1 frames to v2 agent
- Pipeline builder TASK intent routing
- End-to-end flow from TranscriptionFrame to UserResponseFrame
- Multi-tenancy via ToolFactory pattern (v2.1)

Core Invariant (ADR-004):
    "If I can't explain an execution by looking only at the Frame stream,
     the design is broken."

All agent interactions must be visible through frame metadata.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from knomly.pipeline.processors.agent_bridge import (
    AgentBridgeProcessor,
    create_task_agent_bridge,
)
from knomly.pipeline.frames import (
    Frame,
    TranscriptionFrame,
    UserResponseFrame,
)
from knomly.tools.base import Tool, ToolResult
from knomly.tools.registry import ToolRegistry
from knomly.tools.factory import StaticToolFactory, ToolContext
from knomly.agent.frames import AgentResponseFrame
from knomly.agent.result import AgentResult, success_result


# =============================================================================
# Test Fixtures
# =============================================================================


class MockLLMProvider:
    """Mock LLM provider for testing."""

    def __init__(self, responses: list[str] | None = None):
        self._responses = responses or ['{"action": "respond", "message": "Done!", "reasoning": "Test"}']
        self._call_count = 0

    async def complete(self, messages, config=None):
        response = self._responses[min(self._call_count, len(self._responses) - 1)]
        self._call_count += 1

        # Return mock response
        mock_response = MagicMock()
        mock_response.content = response
        return mock_response


class MockTool(Tool):
    """Mock tool for testing."""

    def __init__(self, name: str = "mock_tool"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return "A mock tool for testing"

    @property
    def input_schema(self) -> dict:
        return {"type": "object", "properties": {}}

    async def execute(self, arguments: dict) -> ToolResult:
        return ToolResult.success("Mock executed")


@pytest.fixture
def mock_llm():
    """Create mock LLM provider."""
    return MockLLMProvider()


@pytest.fixture
def mock_plane_client():
    """Create mock Plane client."""
    client = AsyncMock()
    client.create_work_item = AsyncMock(
        return_value=MagicMock(
            id="task-uuid-123",
            name="Test Task",
            sequence_id=123,
            project_identifier="MOB",
        )
    )
    client.list_work_items = AsyncMock(
        return_value=MagicMock(
            results=[],
            has_next=False,
        )
    )
    return client


@pytest.fixture
def mock_plane_cache():
    """Create mock Plane entity cache."""
    cache = MagicMock()
    cache.resolve_project = MagicMock(return_value="project-uuid-123")
    cache.resolve_user = MagicMock(return_value="user-uuid-123")
    cache.get_project_mapping = MagicMock(
        return_value={"Mobile App": "project-uuid-123"}
    )
    return cache


@pytest.fixture
def mock_context():
    """Create mock pipeline context."""
    ctx = MagicMock()
    ctx.providers = MagicMock()
    ctx.providers.get_llm = MagicMock(return_value=None)
    return ctx


@pytest.fixture
def enriched_transcription_frame():
    """Create a transcription frame with plane context metadata."""
    return TranscriptionFrame(
        original_text="Create a task for Mobile App",
        english_text="Create a task for Mobile App",
        detected_language="en",
        confidence=0.95,
        sender_phone="+1234567890",
        metadata={
            "intent": "task",
            "intent_confidence": 0.92,
            "plane_context": {
                "projects": [
                    {"id": "proj-1", "name": "Mobile App", "identifier": "MOB"},
                    {"id": "proj-2", "name": "Backend API", "identifier": "API"},
                ],
                "users": [
                    {"id": "user-1", "name": "Alice", "email": "alice@example.com"},
                ],
                "cache_healthy": True,
            },
        },
    )


# =============================================================================
# AgentBridgeProcessor Tests
# =============================================================================


class TestAgentBridgeProcessor:
    """Tests for AgentBridgeProcessor."""

    def test_processor_name(self):
        """Test processor has correct name."""
        processor = AgentBridgeProcessor(tools=[])
        assert processor.name == "agent_bridge"

    @pytest.mark.asyncio
    async def test_initialize_resolves_llm(
        self, mock_llm, mock_context
    ):
        """Test initialization resolves LLM provider."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
            max_iterations=3,
        )

        await processor.initialize(mock_context)

        # LLM is resolved and stored
        assert processor._llm is not None

    @pytest.mark.asyncio
    async def test_process_extracts_goal_from_text(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Test that goal is extracted from frame text."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        # Create mock executor
        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Created task",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
            )
        )

        # Patch _build_executor to return our mock
        with patch.object(processor, "_build_executor", return_value=mock_executor):
            await processor.process(enriched_transcription_frame, mock_context)

        # Verify goal was extracted from english_text
        mock_executor.run.assert_called_once()
        call_kwargs = mock_executor.run.call_args.kwargs
        assert call_kwargs["goal"] == "Create a task for Mobile App"

    @pytest.mark.asyncio
    async def test_process_passes_frame_with_context(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Test that enriched frame is passed to executor."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Done",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
            )
        )

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            await processor.process(enriched_transcription_frame, mock_context)

        # Verify the frame with context was passed
        call_kwargs = mock_executor.run.call_args.kwargs
        initial_context = call_kwargs["initial_context"]

        assert "plane_context" in initial_context.metadata
        assert len(initial_context.metadata["plane_context"]["projects"]) == 2

    @pytest.mark.asyncio
    async def test_process_returns_user_response_frame(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Test that processing returns UserResponseFrame."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Task created successfully!",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
                tools_called=("plane_create_task",),
            )
        )

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            result = await processor.process(enriched_transcription_frame, mock_context)

        assert isinstance(result, UserResponseFrame)
        assert result.message == "Task created successfully!"
        assert result.sender_phone == "+1234567890"

    @pytest.mark.asyncio
    async def test_process_includes_agent_metadata(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Test that agent metadata is included in response frame."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Done",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=2,
                tools_called=("tool1", "tool2"),
            )
        )

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            result = await processor.process(enriched_transcription_frame, mock_context)

        # Verify agent metadata is included
        assert result.metadata["agent_success"] is True
        assert result.metadata["agent_iterations"] == 2
        assert result.metadata["agent_tools_called"] == ["tool1", "tool2"]
        assert "agent_duration_ms" in result.metadata  # Just check it exists

    @pytest.mark.asyncio
    async def test_process_handles_empty_goal(
        self, mock_llm, mock_context
    ):
        """Test handling frame with no extractable goal."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        # Create frame with empty text
        empty_frame = TranscriptionFrame(
            original_text="",
            english_text="",
            detected_language="en",
            confidence=0.0,
            sender_phone="+1234567890",
        )

        result = await processor.process(empty_frame, mock_context)

        assert isinstance(result, UserResponseFrame)
        assert "couldn't understand" in result.message.lower()


# =============================================================================
# Context Handoff Tests (ADR-004 Compliance)
# =============================================================================


class TestContextHandoff:
    """
    Tests verifying context flows correctly from v1 to v2.

    ADR-004: Context must flow through Frame metadata, not PipelineContext.
    """

    @pytest.mark.asyncio
    async def test_plane_context_preserved_in_frame(
        self, enriched_transcription_frame
    ):
        """Verify plane_context is in frame metadata."""
        assert "plane_context" in enriched_transcription_frame.metadata

        plane_ctx = enriched_transcription_frame.metadata["plane_context"]
        assert len(plane_ctx["projects"]) == 2
        assert plane_ctx["projects"][0]["name"] == "Mobile App"

    @pytest.mark.asyncio
    async def test_context_visible_in_agent_history(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Verify context is visible when agent processes frame."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        # Capture what gets passed to executor
        captured_context = None

        async def capture_run(goal, initial_context, session_id=None):
            nonlocal captured_context
            captured_context = initial_context
            return AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Done",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
            )

        mock_executor = MagicMock()
        mock_executor.run = capture_run

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            await processor.process(enriched_transcription_frame, mock_context)

        # Verify context was passed through
        assert captured_context is not None
        assert "plane_context" in captured_context.metadata
        assert captured_context.metadata["plane_context"]["cache_healthy"] is True


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTaskAgentBridge:
    """Tests for create_task_agent_bridge factory function."""

    def test_creates_processor_with_factory(
        self, mock_plane_client, mock_plane_cache, mock_llm
    ):
        """Test factory creates processor with StaticToolFactory."""
        processor = create_task_agent_bridge(
            plane_client=mock_plane_client,
            plane_cache=mock_plane_cache,
            llm_provider=mock_llm,
        )

        assert isinstance(processor, AgentBridgeProcessor)
        # Should have a StaticToolFactory with 2 tools
        assert isinstance(processor._tool_factory, StaticToolFactory)
        tools = processor._tool_factory.build_tools(ToolContext(user_id="test"))
        assert len(tools) == 2  # PlaneCreateTaskTool, PlaneQueryTasksTool

    def test_respects_max_iterations(
        self, mock_plane_client, mock_plane_cache, mock_llm
    ):
        """Test factory respects max_iterations parameter."""
        processor = create_task_agent_bridge(
            plane_client=mock_plane_client,
            plane_cache=mock_plane_cache,
            llm_provider=mock_llm,
            max_iterations=10,
        )

        assert processor._max_iterations == 10


# =============================================================================
# Pipeline Builder Integration Tests
# =============================================================================


class TestPipelineBuilderIntegration:
    """Tests for pipeline builder TASK intent routing."""

    def test_task_intent_disabled_without_dependencies(self):
        """Test TASK intent is disabled when dependencies missing."""
        from knomly.pipeline.builder import PipelineFactory

        # Create mock settings
        settings = MagicMock()
        settings.twilio_account_sid = "test"
        settings.twilio_auth_token = "test"
        settings.twilio_whatsapp_number = "+1234567890"

        factory = PipelineFactory(settings)
        cases = factory._build_intent_cases()

        # TASK should not be in cases without plane dependencies
        assert "task" not in cases

    def test_task_intent_enabled_with_dependencies(
        self, mock_plane_client, mock_plane_cache, mock_llm
    ):
        """Test TASK intent is enabled when dependencies provided."""
        from knomly.pipeline.builder import PipelineFactory

        settings = MagicMock()
        settings.twilio_account_sid = "test"
        settings.twilio_auth_token = "test"
        settings.twilio_whatsapp_number = "+1234567890"

        factory = PipelineFactory(
            settings,
            plane_client=mock_plane_client,
            plane_cache=mock_plane_cache,
            llm_provider=mock_llm,
        )
        cases = factory._build_intent_cases()

        assert "task" in cases
        assert isinstance(cases["task"], AgentBridgeProcessor)

    def test_context_enrichment_added_with_cache(self, mock_plane_cache):
        """Test ContextEnrichmentProcessor added when cache provided."""
        from knomly.pipeline.builder import PipelineFactory

        settings = MagicMock()
        settings.twilio_account_sid = "test"
        settings.twilio_auth_token = "test"
        settings.twilio_whatsapp_number = "+1234567890"

        factory = PipelineFactory(
            settings,
            plane_cache=mock_plane_cache,
        )

        # Build the pipeline
        pipeline = factory.create_voice_pipeline()

        # Verify ContextEnrichmentProcessor is in the pipeline
        processor_names = [p.name for p in pipeline.processors]
        assert "context_enrichment" in processor_names


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestAgentBridgeErrorHandling:
    """Tests for error handling in agent bridge."""

    @pytest.mark.asyncio
    async def test_agent_failure_returns_error_response(
        self, mock_llm, mock_context, enriched_transcription_frame
    ):
        """Test that agent failure returns helpful error response."""
        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
        )

        await processor.initialize(mock_context)

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=False,
                response=None,
                frames=(),
                iterations=5,
                error_message="Maximum iterations reached",
                error_type="max_iterations",
            )
        )

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            result = await processor.process(enriched_transcription_frame, mock_context)

        assert isinstance(result, UserResponseFrame)
        assert "issue" in result.message.lower()
        assert result.metadata["agent_success"] is False
        assert result.metadata["agent_error_type"] == "max_iterations"

    @pytest.mark.asyncio
    async def test_missing_llm_raises_on_initialize(self, mock_context):
        """Test that missing LLM raises RuntimeError on initialize."""
        processor = AgentBridgeProcessor(tools=[MockTool()])

        # Ensure no LLM in context
        mock_context.providers = None

        with pytest.raises(RuntimeError, match="LLM provider"):
            await processor.initialize(mock_context)


# =============================================================================
# Multi-Tenant Isolation Tests (v3.2 Security Hardening)
# =============================================================================


class TestMultiTenantIsolation:
    """
    Integration tests for multi-tenant isolation.

    Verifies:
    1. Secret provider callback is used (not frame metadata)
    2. Different users get different secrets
    3. Tool deduplication with precedence (dynamic > factory)
    4. User isolation - one user's tools don't leak to another

    Security (ADR v3.2):
        Secrets MUST come from secret_provider callback, never from frames.
        This prevents secrets from leaking via frame serialization/logging.
    """

    @pytest.fixture
    def user_secrets_store(self):
        """Simulated per-user secrets store."""
        return {
            "user-alice": {"plane_api_key": "alice-secret-key"},
            "user-bob": {"plane_api_key": "bob-secret-key"},
            "anonymous": {},  # No secrets for anonymous
        }

    @pytest.fixture
    def mock_secret_provider(self, user_secrets_store):
        """Secret provider callback that returns user-specific secrets."""
        calls = []

        def provider(user_id: str) -> dict[str, str]:
            calls.append(user_id)
            return user_secrets_store.get(user_id, {})

        provider.calls = calls
        return provider

    @pytest.mark.asyncio
    async def test_secret_provider_is_called_with_user_id(
        self, mock_llm, mock_context, mock_secret_provider
    ):
        """Verify secret_provider is called with correct user_id."""
        from knomly.tools.factory import extract_tool_context_from_frame

        # Create frame with user_id in metadata
        frame = TranscriptionFrame(
            original_text="Create a task",
            english_text="Create a task",
            detected_language="en",
            confidence=1.0,
            sender_phone="+1234567890",
            metadata={"user_id": "user-alice"},
        )

        ctx = extract_tool_context_from_frame(
            frame, secret_provider=mock_secret_provider
        )

        # Verify provider was called with alice's user_id
        assert "user-alice" in mock_secret_provider.calls
        assert ctx.secrets == {"plane_api_key": "alice-secret-key"}

    @pytest.mark.asyncio
    async def test_secrets_in_frame_metadata_are_ignored(
        self, mock_llm, mock_context, mock_secret_provider
    ):
        """
        SECURITY: Secrets in frame.metadata are IGNORED.

        This prevents secrets from leaking via serialization.
        """
        from knomly.tools.factory import extract_tool_context_from_frame
        import logging

        # Create frame with secrets in metadata (should be ignored)
        frame = TranscriptionFrame(
            original_text="Create a task",
            english_text="Create a task",
            detected_language="en",
            confidence=1.0,
            sender_phone="+1234567890",
            metadata={
                "user_id": "user-alice",
                "secrets": {"leaked_key": "SHOULD_NOT_APPEAR"},  # Security hole attempt
            },
        )

        # Use a caplog fixture to verify warning is logged
        with patch("knomly.tools.factory.logger") as mock_logger:
            ctx = extract_tool_context_from_frame(
                frame, secret_provider=mock_secret_provider
            )

            # Verify warning was logged about secrets in frame
            mock_logger.warning.assert_called()
            warning_call = str(mock_logger.warning.call_args)
            assert "SECURITY" in warning_call or "secrets" in warning_call.lower()

        # Verify frame secrets were NOT used
        assert "leaked_key" not in ctx.secrets
        # Verify provider secrets WERE used
        assert ctx.secrets == {"plane_api_key": "alice-secret-key"}

    @pytest.mark.asyncio
    async def test_different_users_get_different_secrets(
        self, mock_llm, mock_secret_provider
    ):
        """Verify users are isolated - each gets their own secrets."""
        from knomly.tools.factory import extract_tool_context_from_frame

        # Alice's frame
        alice_frame = TranscriptionFrame(
            original_text="Task for Alice",
            english_text="Task for Alice",
            detected_language="en",
            confidence=1.0,
            sender_phone="+1111111111",
            metadata={"user_id": "user-alice"},
        )

        # Bob's frame
        bob_frame = TranscriptionFrame(
            original_text="Task for Bob",
            english_text="Task for Bob",
            detected_language="en",
            confidence=1.0,
            sender_phone="+2222222222",
            metadata={"user_id": "user-bob"},
        )

        alice_ctx = extract_tool_context_from_frame(
            alice_frame, secret_provider=mock_secret_provider
        )
        bob_ctx = extract_tool_context_from_frame(
            bob_frame, secret_provider=mock_secret_provider
        )

        # Verify isolation
        assert alice_ctx.secrets["plane_api_key"] == "alice-secret-key"
        assert bob_ctx.secrets["plane_api_key"] == "bob-secret-key"
        assert alice_ctx.secrets != bob_ctx.secrets

    @pytest.mark.asyncio
    async def test_tool_deduplication_dynamic_wins(
        self, mock_llm, enriched_transcription_frame
    ):
        """
        Verify tool precedence: dynamic tools (resolver) > factory tools.

        When same-named tool exists in both dynamic and factory,
        the dynamic version is used.
        """
        # Create two tools with the same name
        factory_tool = MockTool(name="create_task")
        factory_tool._source = "factory"  # Tag for identification

        dynamic_tool = MockTool(name="create_task")
        dynamic_tool._source = "dynamic"  # Tag for identification

        # Create context with proper dict metadata (not MagicMock)
        test_context = MagicMock()
        test_context.providers = MagicMock()
        test_context.providers.get_llm = MagicMock(return_value=None)
        test_context.metadata = {"dynamic_tools": [dynamic_tool]}  # Real dict

        processor = AgentBridgeProcessor(
            tools=[factory_tool],  # Factory tool
            llm_provider=mock_llm,
        )

        await processor.initialize(test_context)

        # Track which tool gets used
        used_tools = []

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Done",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
                tools_called=("create_task",),
            )
        )

        # Capture the tools passed to executor
        def capture_executor(tools):
            for t in tools:
                used_tools.append((t.name, getattr(t, "_source", "unknown")))
            return mock_executor

        with patch.object(processor, "_build_executor", side_effect=capture_executor):
            await processor.process(enriched_transcription_frame, test_context)

        # Verify only ONE create_task tool was passed (the dynamic one)
        create_task_tools = [t for t in used_tools if t[0] == "create_task"]
        assert len(create_task_tools) == 1
        assert create_task_tools[0][1] == "dynamic"  # Dynamic won

    @pytest.mark.asyncio
    async def test_secret_provider_failure_graceful_degradation(
        self, mock_llm
    ):
        """Verify system continues if secret_provider fails."""
        from knomly.tools.factory import extract_tool_context_from_frame

        def failing_provider(user_id: str):
            raise RuntimeError("Vault connection failed")

        frame = TranscriptionFrame(
            original_text="Create a task",
            english_text="Create a task",
            detected_language="en",
            confidence=1.0,
            sender_phone="+1234567890",
            metadata={"user_id": "user-alice"},
        )

        # Should not raise - graceful degradation
        ctx = extract_tool_context_from_frame(
            frame, secret_provider=failing_provider
        )

        # Secrets are empty, but user_id is preserved
        assert ctx.user_id == "user-alice"
        assert ctx.secrets == {}

    @pytest.mark.asyncio
    async def test_processor_passes_secret_provider_to_factory(
        self, mock_llm, mock_context, mock_secret_provider
    ):
        """Verify AgentBridgeProcessor uses secret_provider correctly."""
        # Create a frame with user metadata
        frame = TranscriptionFrame(
            original_text="Create a task",
            english_text="Create a task",
            detected_language="en",
            confidence=1.0,
            sender_phone="+1234567890",
            metadata={"user_id": "user-alice"},
        )

        processor = AgentBridgeProcessor(
            tools=[MockTool()],
            llm_provider=mock_llm,
            secret_provider=mock_secret_provider,  # v3.2: Pass provider
        )

        await processor.initialize(mock_context)

        mock_executor = AsyncMock()
        mock_executor.run = AsyncMock(
            return_value=AgentResult(
                success=True,
                response=AgentResponseFrame(
                    response_text="Done",
                    goal_achieved=True,
                ),
                frames=(),
                iterations=1,
            )
        )

        with patch.object(processor, "_build_executor", return_value=mock_executor):
            await processor.process(frame, mock_context)

        # Verify secret provider was called
        assert "user-alice" in mock_secret_provider.calls
