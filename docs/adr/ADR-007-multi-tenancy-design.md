# ADR-007: Multi-Tenancy Design (ToolFactory Pattern)

## Status
**ACCEPTED** - January 2026

## Context

### The Problem: Static Tool Trap

The original v2 implementation instantiated tools at application startup:

```python
# builder.py (BEFORE - broken for multi-tenant)
tools = [PlaneCreateTaskTool(api_key="env_var")]  # Static
builder.add(AgentBridgeProcessor(tools=tools))
```

This created a **single-tenant agent**:
- All users shared one API key
- User A's request used User B's credentials
- Multi-tenant SaaS was impossible

### First Principles Violation

**"Context determines Execution."**

In the broken design, execution was determined by deployment, not by request context. The API key was "baked in" at startup.

## Decision

### The Solution: ToolFactory Pattern

Tools are now built per-request via a factory that receives user-specific context:

```python
# AFTER - multi-tenant safe
class ToolFactory(Protocol):
    def build_tools(self, context: ToolContext) -> Sequence[Tool]

@dataclass(frozen=True)
class ToolContext:
    user_id: str
    secrets: dict[str, str]  # User's API keys
    metadata: dict[str, Any]
```

### Architecture Change

```
BEFORE (Static):
┌─────────────────────────────────────────────┐
│ Application Startup                          │
│                                              │
│  tools = [PlaneCreateTaskTool(env_key)]     │
│  bridge = AgentBridgeProcessor(tools)        │
│                                              │
│ All requests use same tools                  │
└─────────────────────────────────────────────┘

AFTER (Dynamic):
┌─────────────────────────────────────────────┐
│ Application Startup                          │
│                                              │
│  factory = PlaneToolFactory(workspace="...") │
│  bridge = AgentBridgeProcessor(factory)      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ Per-Request (Runtime)                        │
│                                              │
│  context = extract_from_frame(frame)         │
│  tools = factory.build_tools(context)        │
│  executor = build_executor(tools)            │
│  result = executor.run(goal)                 │
│                                              │
│ Each request gets user-scoped tools          │
└─────────────────────────────────────────────┘
```

### Implementation Components

#### 1. ToolContext (tools/factory.py)

Carries user credentials extracted from Frame:

```python
@dataclass(frozen=True)
class ToolContext:
    user_id: str                    # Unique user identifier
    secrets: dict[str, str] = {}    # API keys, tokens
    metadata: dict[str, Any] = {}   # Additional context
```

#### 2. ToolFactory Protocol (tools/factory.py)

Interface for per-request tool creation:

```python
class ToolFactory(Protocol):
    def build_tools(self, context: ToolContext) -> Sequence[Tool]:
        ...
```

#### 3. Implementations

- **StaticToolFactory**: For testing/single-tenant (backwards compatible)
- **CompositeToolFactory**: Combines multiple factories
- **ConditionalToolFactory**: Role-based tool access
- **PlaneToolFactory**: Builds Plane tools with user's API key

#### 4. AgentBridgeProcessor Changes

Now accepts factory instead of static tools:

```python
class AgentBridgeProcessor(Processor):
    def __init__(
        self,
        *,
        tools: list[Tool] | None = None,        # Legacy
        tool_factory: ToolFactory | None = None, # Recommended
        ...
    ):
        if tools:
            self._tool_factory = StaticToolFactory(tools)
        elif tool_factory:
            self._tool_factory = tool_factory
```

Per-request processing:

```python
async def process(self, frame, ctx):
    # Extract context from Frame
    tool_context = extract_tool_context_from_frame(frame)

    # Build tools for THIS user
    tools = self._tool_factory.build_tools(tool_context)

    # Build executor for THIS request
    executor = self._build_executor(tools)

    # Execute
    return await executor.run(...)
```

### Context Extraction

User credentials are extracted from Frame metadata:

```python
def extract_tool_context_from_frame(frame) -> ToolContext:
    metadata = frame.metadata
    return ToolContext(
        user_id=metadata.get("user_id") or frame.sender_phone,
        secrets=metadata.get("secrets", {}),
        metadata={k: v for k, v in metadata.items()
                  if k not in ("user_id", "secrets")},
    )
```

### Security Considerations

1. **Secrets in Frame metadata is temporary**
   - Production: Use user_id to fetch from vault at runtime
   - See `extract_tool_context_with_vault()` helper

2. **Secrets don't persist beyond request**
   - ToolContext is frozen (immutable)
   - Tools are garbage collected after request

3. **User isolation enforced at build time**
   - Each request builds fresh tools
   - No shared state between users

## Usage Examples

### Single-Tenant (Testing)

```python
# Static tools - same API key for all requests
processor = AgentBridgeProcessor(
    tools=[PlaneCreateTaskTool(client, cache)],
    llm_provider=llm,
)
```

### Multi-Tenant (Production)

```python
# Dynamic tools - user's API key per request
processor = AgentBridgeProcessor(
    tool_factory=PlaneToolFactory(
        base_url="https://api.plane.so/api/v1",
        workspace_slug="my-workspace",
    ),
    llm_provider=llm,
)
```

### Role-Based Access

```python
# Different tools for different user roles
factory = CompositeToolFactory([
    StaticToolFactory([BaseQueryTool()]),  # Everyone
    ConditionalToolFactory(
        inner=StaticToolFactory([AdminTool()]),
        condition=lambda ctx: ctx.metadata.get("role") == "admin",
    ),
])
```

## Consequences

### Positive
- Multi-tenant SaaS is now possible
- User isolation is enforced
- Credentials are scoped to requests
- Backwards compatible (static tools still work)

### Negative
- Slight per-request overhead (tool construction)
- More complex wiring in builder
- Credential fetching adds latency (mitigated by caching)

### Mitigations
- **CachedPlaneToolFactory**: Caches clients per user_id
- **Lazy credential fetch**: Only fetch when needed
- **Connection pooling**: Share connections safely

## Related ADRs
- ADR-005: Agentic Layer Design
- ADR-006: Architectural Invariants

## Testing

```python
# Test multi-tenant isolation
def test_different_users_get_different_tools():
    factory = PlaneToolFactory(workspace_slug="test")

    ctx1 = ToolContext(user_id="user1", secrets={"plane_api_key": "key1"})
    ctx2 = ToolContext(user_id="user2", secrets={"plane_api_key": "key2"})

    tools1 = factory.build_tools(ctx1)
    tools2 = factory.build_tools(ctx2)

    # Different tool instances with different credentials
    assert tools1[0]._client._config.api_key == "key1"
    assert tools2[0]._client._config.api_key == "key2"
```
