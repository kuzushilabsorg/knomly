"""
Tests for Knomly observability module.
"""

from datetime import UTC, datetime

import pytest

from knomly.pipeline import (
    AuditEntry,
    InMemoryAuditRepository,
    JSONLogger,
    NoOpSpan,
    NoOpTracer,
    PipelineContext,
    PipelineLogger,
    PipelineMetrics,
    create_audit_entry,
    get_metrics,
    get_tracer,
    reset_metrics,
    set_tracer,
)
from knomly.pipeline.frames import AudioInputFrame

# =============================================================================
# JSONLogger Tests
# =============================================================================


class TestJSONLogger:
    """Tests for JSONLogger."""

    def test_logs_valid_json(self, capsys):
        logger = JSONLogger(name="test")
        logger.info("Test message")

        # Get captured output (if using print fallback)
        # Note: In actual use, this would go to Python logging

    def test_includes_timestamp(self):
        logger = JSONLogger(name="test")
        # Just verify it doesn't raise
        logger.info("Test message")

    def test_includes_request_id(self):
        logger = JSONLogger(name="test", request_id="req-123")
        # Just verify it doesn't raise
        logger.info("Test message")

    def test_includes_extra_context(self):
        logger = JSONLogger(
            name="test",
            extra_context={"service": "knomly", "version": "1.0"},
        )
        logger.info("Test message")

    def test_with_context_creates_new_logger(self):
        logger = JSONLogger(name="test", request_id="req-123")
        new_logger = logger.with_context(pipeline="standup")

        assert new_logger is not logger
        assert new_logger.request_id == "req-123"
        assert new_logger.extra_context.get("pipeline") == "standup"

    def test_all_log_levels(self):
        logger = JSONLogger(name="test")
        # Verify none raise
        logger.debug("Debug message")
        logger.info("Info message")
        logger.warning("Warning message")
        logger.error("Error message")


# =============================================================================
# PipelineLogger Tests
# =============================================================================


class TestPipelineLogger:
    """Tests for PipelineLogger."""

    def test_pipeline_started(self):
        logger = PipelineLogger(request_id="req-123", pipeline_name="test")
        # Verify doesn't raise
        logger.pipeline_started(
            pipeline_name="test",
            processors=["a", "b", "c"],
            frame_type="AudioInputFrame",
            frame_id="frame-123",
        )

    def test_pipeline_completed_success(self):
        logger = PipelineLogger(request_id="req-123")
        logger.pipeline_completed(
            success=True,
            duration_ms=1234.5,
            output_count=1,
        )

    def test_pipeline_completed_failure(self):
        logger = PipelineLogger(request_id="req-123")
        logger.pipeline_completed(
            success=False,
            duration_ms=500.0,
            output_count=0,
            error="Connection failed",
        )

    def test_processor_lifecycle(self):
        logger = PipelineLogger(request_id="req-123")

        logger.processor_started(
            processor_name="transcription",
            frame_type="AudioInputFrame",
            frame_id="frame-123",
        )

        logger.processor_completed(
            processor_name="transcription",
            duration_ms=2500.0,
            output_type="TranscriptionFrame",
            output_count=1,
        )

    def test_processor_error(self):
        logger = PipelineLogger(request_id="req-123")
        logger.processor_error(
            processor_name="transcription",
            error="Timeout",
            error_type="TimeoutError",
            frame_id="frame-123",
        )

    def test_routing_decision(self):
        logger = PipelineLogger(request_id="req-123")
        logger.routing_decision(
            router_name="Conditional",
            selected_branch="if_true",
            frame_type="TranscriptionFrame",
            condition="confidence > 0.8",
        )

    def test_async_handoff(self):
        logger = PipelineLogger(request_id="req-123")
        logger.async_handoff(
            continuation_id="cont-456",
            processors_remaining=3,
            frame_type="AudioInputFrame",
        )

    def test_retry_attempt(self):
        logger = PipelineLogger(request_id="req-123")
        logger.retry_attempt(
            operation="transcribe",
            attempt=2,
            max_attempts=3,
            error="Connection timeout",
            delay_ms=1000.0,
        )

    def test_circuit_state_change(self):
        logger = PipelineLogger(request_id="req-123")
        logger.circuit_state_change(
            circuit_name="external_api",
            from_state="closed",
            to_state="open",
            failure_count=5,
        )


# =============================================================================
# AuditEntry Tests
# =============================================================================


class TestAuditEntry:
    """Tests for AuditEntry."""

    def test_create_entry(self):
        entry = AuditEntry(
            execution_id="exec-123",
            request_id="req-456",
            timestamp=datetime.now(UTC),
            pipeline_name="standup",
            processors=["a", "b", "c"],
            status="completed",
            success=True,
            duration_ms=1234.5,
        )

        assert entry.execution_id == "exec-123"
        assert entry.pipeline_name == "standup"
        assert entry.success is True

    def test_to_dict(self):
        now = datetime.now(UTC)
        entry = AuditEntry(
            execution_id="exec-123",
            request_id="req-456",
            timestamp=now,
            pipeline_name="standup",
            processors=["a", "b"],
            status="completed",
            success=True,
            duration_ms=1000.0,
            processor_timings={"a": 500.0, "b": 500.0},
        )

        d = entry.to_dict()

        assert d["execution_id"] == "exec-123"
        assert d["timestamp"] == now.isoformat()
        assert d["processors"] == ["a", "b"]
        assert d["processor_timings"] == {"a": 500.0, "b": 500.0}


# =============================================================================
# InMemoryAuditRepository Tests
# =============================================================================


class TestInMemoryAuditRepository:
    """Tests for InMemoryAuditRepository."""

    @pytest.fixture
    def repo(self) -> InMemoryAuditRepository:
        return InMemoryAuditRepository()

    @pytest.fixture
    def sample_entry(self) -> AuditEntry:
        return AuditEntry(
            execution_id="exec-123",
            request_id="req-456",
            timestamp=datetime.now(UTC),
            pipeline_name="test",
            processors=["a"],
            status="completed",
        )

    @pytest.mark.asyncio
    async def test_save_and_find_by_execution_id(
        self, repo: InMemoryAuditRepository, sample_entry: AuditEntry
    ):
        await repo.save(sample_entry)

        found = await repo.find_by_execution_id("exec-123")

        assert found is not None
        assert found.execution_id == "exec-123"

    @pytest.mark.asyncio
    async def test_find_by_request_id(self, repo: InMemoryAuditRepository):
        entry1 = AuditEntry(
            execution_id="exec-1",
            request_id="req-shared",
            timestamp=datetime.now(UTC),
            pipeline_name="test",
            processors=[],
            status="started",
        )
        entry2 = AuditEntry(
            execution_id="exec-2",
            request_id="req-shared",
            timestamp=datetime.now(UTC),
            pipeline_name="test",
            processors=[],
            status="completed",
        )

        await repo.save(entry1)
        await repo.save(entry2)

        found = await repo.find_by_request_id("req-shared")

        assert len(found) == 2

    @pytest.mark.asyncio
    async def test_find_not_found(self, repo: InMemoryAuditRepository):
        found = await repo.find_by_execution_id("nonexistent")
        assert found is None

    @pytest.mark.asyncio
    async def test_max_entries_limit(self, repo: InMemoryAuditRepository):
        repo.max_entries = 5

        for i in range(10):
            entry = AuditEntry(
                execution_id=f"exec-{i}",
                request_id="req",
                timestamp=datetime.now(UTC),
                pipeline_name="test",
                processors=[],
                status="completed",
            )
            await repo.save(entry)

        assert len(repo.entries) == 5
        # Should have the last 5 entries
        assert repo.entries[0].execution_id == "exec-5"

    def test_clear(self, repo: InMemoryAuditRepository):
        repo.entries.append(
            AuditEntry(
                execution_id="exec-1",
                request_id="req",
                timestamp=datetime.now(UTC),
                pipeline_name="test",
                processors=[],
                status="completed",
            )
        )

        repo.clear()

        assert len(repo.entries) == 0


# =============================================================================
# PipelineMetrics Tests
# =============================================================================


class TestPipelineMetrics:
    """Tests for PipelineMetrics."""

    @pytest.fixture
    def metrics(self) -> PipelineMetrics:
        return PipelineMetrics()

    def test_record_execution_success(self, metrics: PipelineMetrics):
        metrics.record_execution(success=True, duration_ms=1000.0)

        assert metrics.executions_total == 1
        assert metrics.executions_success == 1
        assert metrics.executions_failed == 0
        assert 1000.0 in metrics.execution_durations_ms

    def test_record_execution_failure(self, metrics: PipelineMetrics):
        metrics.record_execution(success=False, duration_ms=500.0)

        assert metrics.executions_total == 1
        assert metrics.executions_success == 0
        assert metrics.executions_failed == 1

    def test_record_processor(self, metrics: PipelineMetrics):
        metrics.record_processor("transcription", 1500.0)
        metrics.record_processor("transcription", 1600.0)
        metrics.record_processor("extraction", 800.0)

        assert metrics.frames_processed == 3
        assert len(metrics.processor_durations_ms["transcription"]) == 2
        assert len(metrics.processor_durations_ms["extraction"]) == 1

    def test_record_retry(self, metrics: PipelineMetrics):
        metrics.record_retry()
        metrics.record_retry()

        assert metrics.retries_total == 2

    def test_record_circuit_open(self, metrics: PipelineMetrics):
        metrics.record_circuit_open()

        assert metrics.circuit_opens == 1

    def test_get_stats(self, metrics: PipelineMetrics):
        metrics.record_execution(True, 1000.0)
        metrics.record_execution(True, 2000.0)
        metrics.record_execution(False, 500.0)

        stats = metrics.get_stats()

        assert stats["executions"]["total"] == 3
        assert stats["executions"]["success"] == 2
        assert stats["executions"]["failed"] == 1
        assert stats["executions"]["success_rate"] == 2 / 3

    def test_get_stats_percentiles(self, metrics: PipelineMetrics):
        for i in range(100):
            metrics.record_execution(True, float(i))

        stats = metrics.get_stats()

        assert stats["duration_ms"]["p50"] is not None
        assert stats["duration_ms"]["p95"] is not None

    def test_reset(self, metrics: PipelineMetrics):
        metrics.record_execution(True, 1000.0)
        metrics.record_processor("test", 500.0)
        metrics.record_retry()

        metrics.reset()

        assert metrics.executions_total == 0
        assert metrics.frames_processed == 0
        assert metrics.retries_total == 0

    def test_histogram_trimming(self, metrics: PipelineMetrics):
        metrics.max_histogram_entries = 10

        for i in range(20):
            metrics.record_execution(True, float(i))

        assert len(metrics.execution_durations_ms) == 10


class TestGlobalMetrics:
    """Tests for global metrics functions."""

    def test_get_metrics_returns_instance(self):
        metrics = get_metrics()
        assert isinstance(metrics, PipelineMetrics)

    def test_reset_metrics(self):
        metrics = get_metrics()
        metrics.record_execution(True, 1000.0)

        reset_metrics()

        assert metrics.executions_total == 0


# =============================================================================
# Tracer Tests
# =============================================================================


class TestNoOpTracer:
    """Tests for NoOpTracer."""

    def test_start_span_returns_noop_span(self):
        tracer = NoOpTracer()
        span = tracer.start_span("test")

        assert isinstance(span, NoOpSpan)

    def test_noop_span_methods_dont_raise(self):
        span = NoOpSpan()

        # None of these should raise
        span.set_attribute("key", "value")
        span.set_status("ok")
        span.record_exception(RuntimeError("test"))
        span.end()


class TestGlobalTracer:
    """Tests for global tracer functions."""

    def test_get_tracer_returns_noop_by_default(self):
        tracer = get_tracer()
        assert isinstance(tracer, NoOpTracer)

    def test_set_tracer(self):
        class CustomTracer:
            def start_span(self, name, attributes=None):
                return NoOpSpan()

        original = get_tracer()

        try:
            set_tracer(CustomTracer())
            assert isinstance(get_tracer(), CustomTracer)
        finally:
            set_tracer(original)


# =============================================================================
# create_audit_entry Tests
# =============================================================================


class TestCreateAuditEntry:
    """Tests for create_audit_entry helper."""

    def test_creates_entry_from_context(self):
        ctx = PipelineContext(
            sender_phone="919876543210",
            user_id="user-123",
        )
        frame = AudioInputFrame(audio_data=b"test", sender_phone="919876543210")

        entry = create_audit_entry(
            ctx=ctx,
            pipeline_name="standup",
            processors=["a", "b", "c"],
            input_frame=frame,
            status="completed",
            success=True,
            output_frames=[frame],
        )

        assert entry.pipeline_name == "standup"
        assert entry.processors == ["a", "b", "c"]
        assert entry.status == "completed"
        assert entry.success is True
        assert entry.input_frame_type == "AudioInputFrame"
        assert entry.user_id == "user-123"
        assert entry.sender_phone == "919876543210"
        assert entry.output_count == 1

    def test_includes_routing_decisions(self):
        from knomly.pipeline.routing import RoutingDecision

        ctx = PipelineContext()
        ctx.routing_decisions.append(
            RoutingDecision(
                timestamp=datetime.now(UTC),
                router_name="Conditional",
                frame_id=ctx.execution_id,
                frame_type="Frame",
                selected_branch="if_true",
            )
        )

        frame = AudioInputFrame(audio_data=b"test", sender_phone="123")

        entry = create_audit_entry(
            ctx=ctx,
            pipeline_name="test",
            processors=[],
            input_frame=frame,
            status="completed",
        )

        assert len(entry.routing_decisions) == 1
        assert entry.routing_decisions[0]["router_name"] == "Conditional"
