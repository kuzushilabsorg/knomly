"""
Tests for Context Enrichment (v1.5).

Tests cover:
- PlaneEntityCache caching and resolution
- ContextEnrichmentProcessor frame enrichment
- ExtractionProcessor context-aware prompts
- Full pipeline flow with context injection

Design Verification (ADR-004):
- Context flows through Frame metadata
- Not through PipelineContext
- Frame stream is self-describing
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from knomly.integrations.plane.cache import (
    CachedProject,
    CachedUser,
    PlaneEntityCache,
)
from knomly.pipeline.context import PipelineContext
from knomly.pipeline.frames.base import Frame
from knomly.pipeline.frames.processing import TranscriptionFrame
from knomly.pipeline.processors.context_enrichment import (
    ContextEnrichmentProcessor,
    extract_plane_context,
    get_project_prompt_section,
    get_user_prompt_section,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_plane_client():
    """Create mock Plane client."""
    client = MagicMock()
    client.workspace_slug = "test-workspace"
    return client


@pytest.fixture
def populated_cache(mock_plane_client):
    """Create cache with pre-populated data."""
    cache = PlaneEntityCache(client=mock_plane_client)

    # Populate projects
    cache._projects = {
        "uuid-mobile": CachedProject(
            id="uuid-mobile",
            name="Mobile App",
            identifier="MOB",
            description="Mobile application",
        ),
        "uuid-backend": CachedProject(
            id="uuid-backend",
            name="Backend API",
            identifier="API",
            description="Backend services",
        ),
        "uuid-frontend": CachedProject(
            id="uuid-frontend",
            name="Frontend",
            identifier="FE",
            description="Frontend web app",
        ),
    }

    # Populate users
    cache._users = {
        "uuid-steve": CachedUser(
            id="uuid-steve",
            display_name="Steve Jobs",
            email="steve@apple.com",
            username="steve",
        ),
        "uuid-tim": CachedUser(
            id="uuid-tim",
            display_name="Tim Cook",
            email="tim@apple.com",
            username="tim",
        ),
    }

    cache._last_refresh = datetime.now().timestamp()

    return cache


@pytest.fixture
def pipeline_context():
    """Create test pipeline context."""
    return PipelineContext()


# =============================================================================
# PlaneEntityCache Tests
# =============================================================================


class TestPlaneEntityCache:
    """Tests for PlaneEntityCache."""

    def test_get_project_mapping(self, populated_cache):
        """Test project name/identifier to ID mapping."""
        mapping = populated_cache.get_project_mapping()

        # Should map by name (lowercase)
        assert mapping["mobile app"] == "uuid-mobile"
        assert mapping["backend api"] == "uuid-backend"

        # Should map by identifier (lowercase)
        assert mapping["mob"] == "uuid-mobile"
        assert mapping["api"] == "uuid-backend"

    def test_get_user_mapping(self, populated_cache):
        """Test user name/email to ID mapping."""
        mapping = populated_cache.get_user_mapping()

        # Should map by display name
        assert mapping["steve jobs"] == "uuid-steve"
        assert mapping["tim cook"] == "uuid-tim"

        # Should map by email
        assert mapping["steve@apple.com"] == "uuid-steve"

        # Should map by username
        assert mapping["steve"] == "uuid-steve"

    def test_resolve_project_by_name(self, populated_cache):
        """Test resolving project by name."""
        assert populated_cache.resolve_project("Mobile App") == "uuid-mobile"
        assert populated_cache.resolve_project("mobile app") == "uuid-mobile"

    def test_resolve_project_by_identifier(self, populated_cache):
        """Test resolving project by identifier."""
        assert populated_cache.resolve_project("MOB") == "uuid-mobile"
        assert populated_cache.resolve_project("api") == "uuid-backend"

    def test_resolve_project_by_id(self, populated_cache):
        """Test resolving project by UUID."""
        assert populated_cache.resolve_project("uuid-mobile") == "uuid-mobile"

    def test_resolve_project_not_found(self, populated_cache):
        """Test resolving unknown project returns None."""
        assert populated_cache.resolve_project("unknown") is None

    def test_resolve_user_by_name(self, populated_cache):
        """Test resolving user by display name."""
        assert populated_cache.resolve_user("Steve Jobs") == "uuid-steve"
        assert populated_cache.resolve_user("steve jobs") == "uuid-steve"

    def test_resolve_user_by_email(self, populated_cache):
        """Test resolving user by email."""
        assert populated_cache.resolve_user("steve@apple.com") == "uuid-steve"

    def test_get_best_match_project_exact(self, populated_cache):
        """Test exact match returns confidence 1.0."""
        project_id, confidence = populated_cache.get_best_match_project("Mobile App")
        assert project_id == "uuid-mobile"
        assert confidence == 1.0

    def test_get_best_match_project_partial(self, populated_cache):
        """Test partial match returns lower confidence."""
        project_id, confidence = populated_cache.get_best_match_project("Mobile")
        assert project_id == "uuid-mobile"
        assert confidence == 0.8  # Partial match

    def test_get_project_list_for_prompt(self, populated_cache):
        """Test getting project list formatted for LLM prompt."""
        projects = populated_cache.get_project_list_for_prompt()

        assert len(projects) == 3
        assert all("name" in p and "identifier" in p and "id" in p for p in projects)

    def test_to_frame_metadata(self, populated_cache):
        """Test serializing cache to frame metadata."""
        metadata = populated_cache.to_frame_metadata()

        assert "plane_context" in metadata
        context = metadata["plane_context"]

        assert "projects" in context
        assert "users" in context
        assert "project_mapping" in context
        assert "user_mapping" in context
        assert "cache_timestamp" in context

    def test_cache_staleness(self, mock_plane_client):
        """Test cache staleness detection."""
        cache = PlaneEntityCache(client=mock_plane_client, ttl_seconds=1.0)

        # Fresh cache should not be stale
        cache._last_refresh = datetime.now().timestamp()
        assert not cache.is_stale

        # Old cache should be stale
        cache._last_refresh = 0
        assert cache.is_stale


# =============================================================================
# ContextEnrichmentProcessor Tests
# =============================================================================


class TestContextEnrichmentProcessor:
    """Tests for ContextEnrichmentProcessor."""

    @pytest.mark.asyncio
    async def test_enriches_frame_with_plane_context(self, populated_cache, pipeline_context):
        """Test that processor adds plane_context to frame metadata."""
        processor = ContextEnrichmentProcessor(
            plane_cache=populated_cache,
            auto_refresh=False,  # Use pre-populated data
        )

        frame = TranscriptionFrame(
            original_text="Create a task for the Mobile App",
            english_text="Create a task for the Mobile App",
            sender_phone="+1234567890",
        )

        result = await processor.process(frame, pipeline_context)

        # Should return same frame type
        assert isinstance(result, TranscriptionFrame)

        # Should have enriched metadata
        assert "plane_context" in result.metadata
        context = result.metadata["plane_context"]

        assert "projects" in context
        assert len(context["projects"]) == 3

    @pytest.mark.asyncio
    async def test_passes_through_without_cache(self, pipeline_context):
        """Test processor passes through when no cache configured."""
        processor = ContextEnrichmentProcessor()  # No cache

        frame = TranscriptionFrame(
            original_text="Hello world",
            english_text="Hello world",
            sender_phone="+1234567890",
        )

        result = await processor.process(frame, pipeline_context)

        # Should return original frame unchanged
        assert result is frame

    @pytest.mark.asyncio
    async def test_preserves_existing_metadata(self, populated_cache, pipeline_context):
        """Test that enrichment preserves existing frame metadata."""
        processor = ContextEnrichmentProcessor(
            plane_cache=populated_cache,
            auto_refresh=False,
        )

        frame = TranscriptionFrame(
            original_text="Test",
            english_text="Test",
            sender_phone="+1",
            metadata={"existing_key": "existing_value"},
        )

        result = await processor.process(frame, pipeline_context)

        # Should preserve existing metadata
        assert result.metadata["existing_key"] == "existing_value"

        # Should add new context
        assert "plane_context" in result.metadata


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestContextUtilities:
    """Tests for context extraction utilities."""

    def test_extract_plane_context_present(self, populated_cache):
        """Test extracting plane context from enriched frame."""
        metadata = populated_cache.to_frame_metadata()
        frame = Frame(metadata=metadata)

        context = extract_plane_context(frame)

        assert context is not None
        assert "projects" in context

    def test_extract_plane_context_missing(self):
        """Test extracting plane context when not present."""
        frame = Frame(metadata={})

        context = extract_plane_context(frame)

        assert context is None

    def test_get_project_prompt_section(self, populated_cache):
        """Test generating project prompt section."""
        metadata = populated_cache.to_frame_metadata()
        frame = Frame(metadata=metadata)

        section = get_project_prompt_section(frame)

        assert "Valid Projects" in section
        assert "Mobile App" in section
        assert "MOB" in section

    def test_get_user_prompt_section(self, populated_cache):
        """Test generating user prompt section."""
        metadata = populated_cache.to_frame_metadata()
        frame = Frame(metadata=metadata)

        section = get_user_prompt_section(frame)

        assert "Valid Users" in section
        assert "Steve Jobs" in section
        assert "steve@apple.com" in section


# =============================================================================
# ADR-004 Compliance Tests
# =============================================================================


class TestADR004Compliance:
    """
    Tests verifying ADR-004 compliance.

    Core invariant: Context flows through Frame metadata, not PipelineContext.
    """

    @pytest.mark.asyncio
    async def test_context_in_frame_not_pipeline_context(self, populated_cache, pipeline_context):
        """Verify context is in Frame metadata, not PipelineContext."""
        processor = ContextEnrichmentProcessor(
            plane_cache=populated_cache,
            auto_refresh=False,
        )

        frame = TranscriptionFrame(original_text="Test", english_text="Test", sender_phone="+1")

        result = await processor.process(frame, pipeline_context)

        # Context should be in frame metadata
        assert "plane_context" in result.metadata

        # Context should NOT be in PipelineContext
        assert not hasattr(pipeline_context, "plane_context")
        assert not hasattr(pipeline_context, "valid_projects")

    def test_frame_stream_is_self_describing(self, populated_cache):
        """Verify Frame stream can explain execution without external state."""
        # Simulate a frame stream
        metadata = populated_cache.to_frame_metadata()

        # Create frames as they would appear in the stream
        frame1 = TranscriptionFrame(
            original_text="Create task for Mobile App",
            english_text="Create task for Mobile App",
            sender_phone="+1",
        )

        frame2 = TranscriptionFrame(
            original_text="Create task for Mobile App",
            english_text="Create task for Mobile App",
            sender_phone="+1",
            metadata=metadata,  # Enriched
        )

        # Frame stream should show:
        # 1. frame1 has no context
        # 2. frame2 has plane_context with projects

        # This is verifiable without access to any external state
        assert "plane_context" not in frame1.metadata
        assert "plane_context" in frame2.metadata

        # The enrichment is VISIBLE in the Frame stream
        context = frame2.metadata["plane_context"]
        assert len(context["projects"]) == 3


# =============================================================================
# Failure Mode Tests (v1.9 Hardening)
# =============================================================================


class TestFailureModes:
    """
    Tests for graceful degradation and failure handling.

    Core principle: Context is AUXILIARY, not critical.
    Pipeline should NEVER crash due to context unavailability.
    """

    @pytest.mark.asyncio
    async def test_cache_refresh_timeout(self, mock_plane_client):
        """Test that cache refresh respects timeout and degrades gracefully."""
        import asyncio

        cache = PlaneEntityCache(
            client=mock_plane_client,
            refresh_timeout=0.1,  # 100ms timeout
        )

        # Mock a slow API call
        async def slow_list_projects():
            await asyncio.sleep(1.0)  # 1 second, will exceed timeout
            return []

        mock_plane_client.list_projects = slow_list_projects

        # Refresh should return False (failed) but not raise
        result = await cache.refresh(force=True)

        assert result is False
        assert cache._last_error is not None
        assert "Timeout" in cache._last_error

    @pytest.mark.asyncio
    async def test_cache_api_error_graceful_degradation(self, mock_plane_client):
        """Test that API errors don't crash the cache."""
        cache = PlaneEntityCache(client=mock_plane_client)

        # Mock API error
        async def failing_list_projects():
            raise ConnectionError("Plane API unreachable")

        mock_plane_client.list_projects = failing_list_projects

        # Refresh should return False but not raise
        result = await cache.refresh(force=True)

        assert result is False
        assert cache._last_error is not None
        assert "unreachable" in cache._last_error

    @pytest.mark.asyncio
    async def test_safe_get_context_never_throws(self, mock_plane_client):
        """Test that safe_get_context() NEVER throws, returns empty context."""
        cache = PlaneEntityCache(client=mock_plane_client)

        # Mock catastrophic failure
        async def catastrophic_failure():
            raise RuntimeError("Catastrophic failure")

        mock_plane_client.list_projects = catastrophic_failure

        # safe_get_context should return empty context, not raise
        context = await cache.safe_get_context()

        assert "plane_context" in context
        assert context["plane_context"]["cache_healthy"] is False
        assert context["plane_context"]["projects"] == []

    @pytest.mark.asyncio
    async def test_enrichment_processor_handles_cache_failure(
        self, mock_plane_client, pipeline_context
    ):
        """Test that ContextEnrichmentProcessor continues on cache failure."""
        cache = PlaneEntityCache(client=mock_plane_client)

        # Mock API failure
        async def failing_api():
            raise ConnectionError("Network error")

        mock_plane_client.list_projects = failing_api

        processor = ContextEnrichmentProcessor(
            plane_cache=cache,
            auto_refresh=True,  # Will attempt refresh
        )

        frame = TranscriptionFrame(
            original_text="Create a task",
            english_text="Create a task",
            sender_phone="+1234567890",
        )

        # Processor should return frame (possibly with degraded context), not raise
        result = await processor.process(frame, pipeline_context)

        assert result is not None
        # Frame should still have context metadata (even if degraded)
        if "plane_context" in result.metadata:
            assert result.metadata["plane_context"]["cache_healthy"] is False

    @pytest.mark.asyncio
    async def test_context_includes_health_status(self, populated_cache):
        """Test that context includes cache health status for observability."""
        context = await populated_cache.safe_get_context(auto_refresh=False)

        assert "plane_context" in context
        assert "cache_healthy" in context["plane_context"]
        assert "cache_error" in context["plane_context"]

        # Healthy cache should report healthy
        assert context["plane_context"]["cache_healthy"] is True
        assert context["plane_context"]["cache_error"] is None

    @pytest.mark.asyncio
    async def test_degraded_context_includes_error_info(self, mock_plane_client):
        """Test that degraded context includes error info for debugging."""
        cache = PlaneEntityCache(client=mock_plane_client)

        # Simulate a failed refresh
        cache._last_error = "API returned 503"

        context = await cache.safe_get_context(auto_refresh=False)

        assert context["plane_context"]["cache_healthy"] is False
        assert context["plane_context"]["cache_error"] == "API returned 503"

    def test_empty_context_structure(self, mock_plane_client):
        """Test that empty context has proper structure for downstream processors."""
        cache = PlaneEntityCache(client=mock_plane_client)
        empty = cache._empty_context()

        # Should have all required keys
        assert "plane_context" in empty
        ctx = empty["plane_context"]

        assert ctx["projects"] == []
        assert ctx["users"] == []
        assert ctx["project_mapping"] == {}
        assert ctx["user_mapping"] == {}
        assert ctx["cache_timestamp"] == 0
        assert ctx["cache_healthy"] is False
        assert "cache_error" in ctx
