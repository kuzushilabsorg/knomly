# ADR-006: Architectural Invariants

## Status
**ACCEPTED** - January 2026

## Context

Knomly has reached architectural maturity with:
- v1 Pipeline Layer (execution substrate)
- v2 Agent Layer (agentic capabilities)
- Tools, Skills, and Factories

Feedback analysis identified that the system is at "maximum safe size for conceptual purity." Further growth must be deliberate and constrained.

This ADR establishes **frozen invariants** - architectural rules that must not change without a new ADR.

## Decision

### 1. Pipeline Layer Invariants (FROZEN)

The v1 pipeline layer is **conceptually frozen**. Only these changes are permitted:
- Observability enhancements
- Performance optimizations
- Ergonomic improvements (developer experience)

**Not permitted without new ADR:**
- New semantic primitives
- Changes to Frame flow semantics
- Modifications to routing behavior

### 2. Frame Invariants (FROZEN)

Frames are the **unit of meaning** and observability:

```
IF I cannot explain an execution by looking ONLY at the Frame stream,
THEN the design is broken.
```

**Frame rules:**
1. Every decision → emits a Frame
2. Every tool invocation → emits a Frame
3. Every state change → visible in Frame metadata
4. Frames are immutable after creation
5. Frame.id provides unique identification
6. Frame.source_frame_id provides lineage

### 3. Tool Interface Invariants (FROZEN)

The Tool protocol follows MCP alignment:

```python
class Tool(Protocol):
    name: str
    description: str
    input_schema: dict  # JSON Schema

    async def execute(self, arguments: dict) -> ToolResult
```

**Tool rules:**
1. Tools do NOT know they are called by an agent
2. Tools do exactly ONE real-world action
3. Tool errors are returned in ToolResult, not raised
4. Tools are stateless between calls
5. Tool credentials come from ToolContext (not construction)

### 4. Agent Layer Invariants

The v2 agent layer is a **client of v1**, not an extension:

1. Agents consume Frames, they don't replace them
2. Every agent decision emits a Frame
3. Agent loops are bounded (max_iterations)
4. Agent execution is observable via Frame stream
5. Agents don't modify pipeline semantics

### 5. Multi-Tenancy Invariants

Tools are built per-request, not per-deployment:

```
Context determines Execution.
```

**Rules:**
1. Tools receive credentials via ToolContext at runtime
2. Static tool construction is only for testing
3. User isolation is enforced at tool build time
4. Secrets never persist in memory beyond request scope

### 6. What Knomly Refuses To Do

To maintain conceptual purity, Knomly will NOT:

1. **Add execution semantics to Frames** - Frames are data, not executables
2. **Allow unbounded agent loops** - All loops have explicit limits
3. **Embed business logic in Tools** - Tools are wrappers, not domains
4. **Create hidden state** - All state is visible in Frames
5. **Add "magic" conveniences** - Explicit is better than implicit
6. **Optimize for demos** - Optimize for production resilience

## Consequences

### Positive
- Architectural stability enables confident extension
- New contributors have clear boundaries
- System remains explainable and auditable
- Testing surface is well-defined

### Negative
- New features require explicit justification
- Some "convenient" patterns are forbidden
- Growth is slower (by design)

## Compliance

To add new architectural elements:
1. Check if existing invariants apply
2. If they do, work within them
3. If not, propose a new ADR
4. Get explicit review before implementation

## Related ADRs
- ADR-004: v1/v2 Invariants
- ADR-005: Agentic Layer Design
- ADR-007: Multi-Tenancy Design
