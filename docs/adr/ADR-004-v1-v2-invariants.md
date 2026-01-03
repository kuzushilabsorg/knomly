# ADR-004: v1/v2 Layer Invariants

## Status

Accepted

## Context

Knomly has two execution layers:
- **v1 (Pipeline Layer)**: Deterministic, bounded, explicit execution
- **v2 (Agent Layer)**: Agentic, tool-driven, iterative execution

As the system grows, there's risk of **semantic collapse** where:
- Everything "kind of works"
- Nobody can explain why
- Local reasoning becomes impossible

This ADR establishes **inviolable invariants** that protect architectural integrity.

## The Core Test

> "If I can't explain an execution by looking only at the Frame stream, the design is broken."

This is the litmus test for all architectural decisions.

## v1 Layer Invariants

### 1. Processors Do Not Loop

```python
# CORRECT: Single input → output transformation
async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | Sequence[Frame] | None:
    result = await self._do_work(frame)
    return ResultFrame(data=result)

# VIOLATION: Internal retry/loop
async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
    while not success:  # WRONG - hidden loop
        result = await self._do_work(frame)
        if result.ok:
            success = True
    return result
```

### 2. Data Flows Through Frames, Not Context

```python
# CORRECT: Data in frames, utilities in context
async def process(self, frame: TaskFrame, ctx: PipelineContext) -> TaskResultFrame:
    # Context provides utilities (logger, metrics)
    ctx.logger.info(f"Processing task: {frame.name}")

    # Data stays in frames
    return TaskResultFrame(
        task_id="123",
        source_frame_id=frame.id,  # Lineage preserved
    )

# VIOLATION: Using context as data bus
async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
    ctx.state["resolved_id"] = await self._resolve(frame.name)  # WRONG
    return frame  # Data hidden in context!
```

### 3. Execution Is Explicit

```python
# CORRECT: Clear execution path
pipeline = PipelineBuilder()
    .add(TranscriptionProcessor())
    .add(ExtractionProcessor())
    .add(PlaneProcessor(client))
    .build()

result = await pipeline.execute(audio_frame, ctx)  # Explicit execution

# VIOLATION: Hidden implicit execution
class SneakyProcessor:
    async def process(self, frame, ctx):
        await self._call_external_api()  # Hidden side effect
        await ctx.trigger_webhook()       # Hidden side effect
        return frame
```

### 4. Frames Are Self-Describing

Every frame must contain enough information to understand what happened:

```python
@dataclass(frozen=True)
class TaskResultFrame(Frame):
    # What happened
    operation: TaskOperation  # CREATE, UPDATE, DELETE
    success: bool

    # What was affected
    task_id: str
    task_name: str
    platform: str  # "plane", "linear", etc.

    # Lineage (where did this come from?)
    source_frame_id: UUID | None = None

    # Errors (if any)
    error_message: str = ""
```

### 5. No Frame-External State Mutations

```python
# CORRECT: All state changes flow through frames
async def process(self, frame: TaskFrame, ctx: PipelineContext) -> TaskResultFrame:
    work_item = await self._client.create_work_item(...)
    return TaskResultFrame(
        task_id=work_item.id,
        # ... all data in the frame
    )

# VIOLATION: Mutating shared state
class StatefulProcessor:
    def __init__(self):
        self.created_tasks = []  # Shared mutable state!

    async def process(self, frame, ctx):
        self.created_tasks.append(frame.id)  # WRONG - hidden state
        return frame
```

## v2 Layer Invariants

### 6. Agent Loops Are Bounded

```python
# CORRECT: Explicit iteration limit
class AgentExecutor:
    async def run(self, goal: str, max_iterations: int = 10) -> AgentResult:
        for i in range(max_iterations):
            action = await self._plan(goal)
            if action.is_done:
                break
            await self._execute_tool(action)
        return self._compile_result()

# VIOLATION: Unbounded loop
async def run(self, goal: str) -> AgentResult:
    while True:  # DANGEROUS - no bound
        action = await self._plan(goal)
```

### 7. Agent Decisions Are Inspectable

Every agent decision must be recordable:

```python
@dataclass
class AgentStep:
    """Single step in agent execution - fully inspectable."""

    step_number: int
    timestamp: datetime

    # What the agent thought
    reasoning: str

    # What it decided to do
    action: str  # "tool_call", "complete", "ask_user"
    tool_name: str | None = None
    tool_args: dict[str, Any] | None = None

    # What happened
    tool_result: Frame | None = None
    error: str | None = None
```

### 8. v2 Uses v1, Not The Reverse

```
┌─────────────────────────────────────────────────────────┐
│                     v2 Agent Layer                       │
│   AgentExecutor, Skills, Planner                        │
│                          │                               │
│                          ▼                               │
│            ┌─────────────────────────┐                  │
│            │    Tool Invocation       │                  │
│            └─────────────────────────┘                  │
└─────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│                     v1 Pipeline Layer                    │
│   Processors, Frames, PipelineContext                   │
│                                                         │
│   PlaneProcessor, TranscriptionProcessor, etc.          │
└─────────────────────────────────────────────────────────┘
```

v1 processors must NEVER:
- Know they were called by an agent
- Behave differently based on caller
- Import from v2 layer

### 9. Memory Effects Are Explicit

```python
# CORRECT: Memory is a processor that outputs frames
class MemoryProcessor(Processor):
    """Adds context from memory - effects are visible in frames."""

    async def process(self, frame: QueryFrame, ctx: PipelineContext) -> EnrichedQueryFrame:
        relevant_memories = await self._search_memory(frame.query)

        # Memory effect is VISIBLE in the output frame
        return EnrichedQueryFrame(
            query=frame.query,
            memory_context=relevant_memories,  # Explicitly included
            source_frame_id=frame.id,
        )

# VIOLATION: Hidden memory injection
async def process(self, frame, ctx):
    ctx.memory = await self._search_memory(frame.query)  # WRONG - hidden
    return frame  # Memory effect invisible!
```

## Context Contract

PipelineContext is **execution utilities only**, not a data bus:

### Allowed in Context:
- Logger
- Metrics collector
- Correlation ID / trace ID
- Provider references (for DI)
- Execution metadata (timestamps)

### NOT Allowed in Context:
- Business data (use frames)
- Entity resolution results (use frames)
- Memory/RAG results (use frames)
- Cross-processor state (use frames)
- Mutable shared state

```python
@dataclass
class PipelineContext:
    """Execution utilities only."""

    # ✅ Allowed: execution infrastructure
    correlation_id: str
    logger: logging.Logger
    metrics: MetricsCollector
    providers: ProviderRegistry
    start_time: datetime

    # ❌ NOT ALLOWED (these belong in frames):
    # resolved_entities: dict  # WRONG
    # memory_results: list     # WRONG
    # shared_state: dict       # WRONG
```

## Verification Checklist

Before merging any PR, verify:

- [ ] Can I reconstruct execution by reading only the Frame stream?
- [ ] Are all data mutations visible in frames?
- [ ] Does Context contain only utilities?
- [ ] Do processors return without looping?
- [ ] Are agent iterations bounded and logged?
- [ ] Does v1 code have zero imports from v2?

## Consequences

### Positive

- **Debuggable**: Frame stream is a complete audit log
- **Testable**: Mock frames, not hidden state
- **Modular**: Processors are truly independent
- **Scalable**: No hidden coupling to break

### Negative

- **Verbosity**: More frames, more explicit data passing
- **Discipline**: Easy to "cheat" with context

### Mitigation

The discipline cost is intentional. When it's "hard" to hide state in context, developers are guided toward the correct pattern (frames).

## References

- [ADR-001: Pipeline Architecture](./ADR-001-pipeline-architecture.md)
- [ADR-003: Integration Pattern](./ADR-003-integration-pattern.md)
- [Pipecat Frame Design](https://github.com/pipecat-ai/pipecat)
