# ADR-001: HTTP Pipeline Adaptation from Pipecat Patterns

**Status**: Accepted
**Date**: 2026-01-02
**Author**: Knomly Architecture Team

## Context

We need a modular, extensible pipeline architecture for processing WhatsApp voice notes into structured Zulip standups. Pipecat provides an excellent reference architecture for AI pipelines, but it's designed for **real-time streaming** (WebRTC/WebSocket), while our use case is **HTTP request/response**.

### Pipecat's Design Context
- Continuous audio/video streams
- Sub-100ms latency requirements
- Mid-stream interruptions (user can interrupt bot)
- Bidirectional frame flow (upstream for errors, downstream for data)
- Priority queues (SystemFrames bypass normal queue)
- Push-based model with concurrent frame processing

### Our Design Context (WhatsApp → Zulip)
- Discrete requests (complete voice note arrives as single file)
- Seconds-level latency acceptable
- No mid-stream interruptions (recording is complete)
- Single request = single pipeline execution
- No concurrency within single request
- Request/response is inherently synchronous per-request

## Decision

We adapt Pipecat's **valuable abstractions** while **removing complexity** that doesn't serve HTTP:

### What We Keep

| Pattern | Reason |
|---------|--------|
| **Frame as immutable data container** | Clean data flow, debuggability |
| **Processor as single-responsibility transformer** | Modularity, testability |
| **Pipeline as orchestrator** | Clear execution model |
| **Provider abstraction** | Swappable services |
| **Lineage tracking** | Audit trail, debugging |

### What We Remove/Simplify

| Pipecat Pattern | Our Adaptation | Reason |
|-----------------|----------------|--------|
| Bidirectional flow (upstream/downstream) | **Unidirectional** | No interruptions in HTTP |
| `push_frame()` with direction | **Return value** | Simpler, explicit |
| Priority queue with SystemFrames | **Sequential execution** | Single request, no priority needed |
| PipelineTask with async queues | **Simple async/await chain** | No concurrent frames |
| Transport layer abstraction | **HTTP IS the transport** | FastAPI handles this |
| "Never consume frames" rule | **Processors can return None** | Sometimes we want to stop |

### Core Design Principles

1. **Frames are truly immutable** - Use `@dataclass(frozen=True)` correctly
2. **Processors are pure functions** - `async (Frame, Context) → Frame | list[Frame] | None`
3. **Pipeline is sequential** - Frame flows through processors in order
4. **Errors become ErrorFrames** - Pipeline catches exceptions, wraps in ErrorFrame
5. **Context is request-scoped** - Providers, config, audit log per-request

## Architecture

### Frame Hierarchy

```
Frame (frozen dataclass)
├── InputFrame
│   ├── AudioInputFrame      # Voice note from WhatsApp
│   └── TextInputFrame       # Text message fallback
├── ProcessingFrame
│   ├── TranscriptionFrame   # STT result
│   └── ExtractionFrame      # LLM extraction result
├── ActionFrame
│   ├── ZulipMessageFrame    # Zulip post result
│   └── ConfirmationFrame    # WhatsApp confirmation result
└── ErrorFrame               # Pipeline errors
```

### Processor Interface

```python
class Processor(ABC):
    @property
    def name(self) -> str: ...

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext
    ) -> Frame | list[Frame] | None:
        """
        Transform input frame.

        Returns:
            Frame: Continue with single frame
            list[Frame]: Fan-out to multiple frames
            None: Stop pipeline (frame consumed)

        Raises:
            Exception: Pipeline wraps in ErrorFrame
        """
```

### Pipeline Execution Model

```
                    ┌─────────────────────────────────────────────────┐
                    │              Pipeline.execute()                  │
                    └─────────────────────────────────────────────────┘
                                          │
                                          ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ AudioInput   │───▶│ Transcribe   │───▶│ Extract      │───▶│ Post Zulip   │
│ Frame        │    │ Processor    │    │ Processor    │    │ Processor    │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                           │                   │                   │
                           ▼                   ▼                   ▼
                    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
                    │ Transcription│    │ Extraction   │    │ ZulipMessage │
                    │ Frame        │    │ Frame        │    │ Frame        │
                    └──────────────┘    └──────────────┘    └──────────────┘
                                                                   │
                                          ┌────────────────────────┘
                                          ▼
                                   ┌──────────────┐
                                   │ Confirmation │
                                   │ Processor    │
                                   └──────────────┘
                                          │
                                          ▼
                                   ┌──────────────┐
                                   │ Confirmation │
                                   │ Frame        │
                                   └──────────────┘
```

### Error Handling

```python
# Pipeline catches processor exceptions
try:
    result = await processor.process(frame, ctx)
except Exception as e:
    # Wrap in ErrorFrame, continue or stop based on config
    error_frame = ErrorFrame.from_exception(e, processor.name, frame)
    if error_frame.is_fatal:
        return PipelineResult(error=error_frame)
    frames = [error_frame]  # Continue with error frame
```

### HTTP Integration

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Endpoint                              │
│  POST /api/v1/webhook/twilio                                        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Parse Twilio form data                                          │
│  2. Validate message type (audio)                                   │
│  3. Return HTTP 200 immediately (Twilio requires fast ACK)          │
│  4. Add BackgroundTask for pipeline execution                       │
└─────────────────────────────────────────────────────────────────────┘
                                │
                   ┌────────────┴────────────┐
                   │     BackgroundTask      │
                   └─────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. Create AudioInputFrame with Twilio media URL                    │
│  2. Create PipelineContext with providers, config                   │
│  3. Execute pipeline                                                │
│  4. Log audit record to MongoDB                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Consequences

### Positive
- **Simpler codebase** - No async queues, priority handling, bidirectional flow
- **Easier testing** - Processors are pure functions, easy to unit test
- **Clear data flow** - Frame → Processor → Frame, no hidden state
- **Proper Python** - Uses dataclasses correctly, no metaclass hacks

### Negative
- **Not streaming** - Can't reuse for real-time voice (would need Pipecat)
- **No interruptions** - Can't stop mid-pipeline from external signal
- **Sequential only** - No parallel processor execution (could add later)

### Risks
- **Scaling** - Each request is independent; horizontal scaling via replicas
- **Long-running** - Pipeline runs in BackgroundTask; monitor for timeouts

## Alternatives Considered

### 1. Use Pipecat Directly
**Rejected** - Pipecat requires streaming transport; adding HTTP transport would be significant work and fight the framework's design.

### 2. No Framework (Raw Functions)
**Rejected** - Loses modularity, testability, provider swappability. The Frame/Processor abstraction is valuable even simplified.

### 3. Use Celery for Background Tasks
**Considered** - Could add later if we need distributed execution. For now, FastAPI BackgroundTasks are sufficient.

## References

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [Pipecat Frame Processing](https://docs.pipecat.ai/guides/learn/pipeline)
- [FastAPI Background Tasks](https://fastapi.tiangolo.com/tutorial/background-tasks/)
