# ADR-003: SaaS Integration Pattern

## Status

Accepted

## Context

Knomly needs to integrate with multiple SaaS platforms (Plane, Linear, Jira, Twenty CRM, etc.). We need a pattern that:

1. Allows swapping platforms without changing upstream pipeline code
2. Maintains type safety across the integration boundary
3. Supports the v1/v2 architecture (processors now, Skills/MCP later)
4. Follows Pipecat's design philosophy

## Decision

### Core Principle: Generic Frames, Platform-Specific Processors

Frames describe **WHAT** (the data/intent), not **WHERE** (the platform).
Processors handle **WHERE** (the platform) and **HOW** (the mapping).

```
┌─────────────────────────────────────────────────────────────────┐
│                      Generic Domain Layer                        │
│  TaskFrame, ContactFrame, EventFrame, DocumentFrame, etc.       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Platform-Specific Processors                  │
│  PlaneProcessor, LinearProcessor, JiraProcessor, etc.           │
│  (All accept generic frames, return generic results)            │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Platform Clients & APIs                       │
│  PlaneClient, LinearClient, JiraClient (REST/GraphQL)           │
└─────────────────────────────────────────────────────────────────┘
```

### Analogy with Pipecat

| Pipecat | Knomly |
|---------|--------|
| `AudioRawFrame` (generic) | `TaskFrame` (generic) |
| `DeepgramSTTService` | `PlaneProcessor` |
| `WhisperSTTService` | `LinearProcessor` |
| `TranscriptionFrame` (generic output) | `TaskResultFrame` (generic output) |

The same `AudioRawFrame` works with any STT service. Similarly, the same `TaskFrame` works with any task management platform.

## Architecture

### Layer 1: Generic Domain Frames

Located in `pipeline/frames/task.py`:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class TaskFrame(Frame):
    """Platform-agnostic task representation."""

    operation: TaskOperation = TaskOperation.CREATE
    name: str = ""
    description: str = ""
    priority: TaskPriority | str = TaskPriority.NONE
    status: TaskStatus | str = TaskStatus.TODO
    project: str = ""  # Name or key, not platform-specific ID
    # ... more fields

@dataclass(frozen=True, kw_only=True, slots=True)
class TaskResultFrame(Frame):
    """Platform-agnostic result."""

    success: bool = True
    task_id: str = ""
    platform: str = ""  # "plane", "linear", "jira"
    # ... more fields
```

### Layer 2: Platform-Specific Processors

Located in `pipeline/processors/integrations/<platform>.py`:

```python
class PlaneProcessor(Processor):
    """Maps generic TaskFrame to Plane API."""

    def __init__(
        self,
        client: PlaneClient,
        default_project_id: str = "",
        project_mapping: dict[str, str] | None = None,  # name → ID
    ):
        self.client = client
        self.project_mapping = project_mapping or {}

    async def process(self, frame: Frame, ctx: PipelineContext):
        if isinstance(frame, TaskFrame):
            # Map generic priority → Plane priority
            plane_priority = PlaneMappings.priority_to_plane(frame.priority)

            # Map project name → Plane project ID
            project_id = self._resolve_project(frame.project)

            # Call Plane API
            work_item = await self.client.create_work_item(
                project_id=project_id,
                name=frame.name,
                priority=plane_priority,
            )

            # Return GENERIC result
            return TaskResultFrame(
                success=True,
                task_id=work_item.id,
                platform="plane",
                source_frame_id=frame.id,
            )
```

### Layer 3: Platform Clients

Located in `integrations/<platform>/client.py`:

```python
class PlaneClient(IntegrationClient):
    """Low-level Plane API client."""

    @property
    def name(self) -> str:
        return "plane"

    async def create_work_item(
        self,
        project_id: str,
        name: str,
        *,
        priority: str | None = None,
    ) -> WorkItem:
        response = await self._request(
            "POST",
            f"/api/v1/workspaces/{self.workspace_slug}/projects/{project_id}/work-items/",
            json={"name": name, "priority": priority},
        )
        return WorkItem.model_validate(response.json())
```

## Adding a New Integration

### Step 1: Define Platform Schemas

```python
# integrations/linear/schemas.py
class LinearIssue(BaseModel):
    id: str
    title: str
    priority: int  # Linear uses 0-4
    # ...
```

### Step 2: Implement Platform Client

```python
# integrations/linear/client.py
class LinearClient(IntegrationClient):
    """Linear GraphQL API client."""

    @property
    def name(self) -> str:
        return "linear"

    async def create_issue(self, team_id: str, title: str, **kwargs) -> LinearIssue:
        query = """
        mutation IssueCreate($input: IssueCreateInput!) {
            issueCreate(input: $input) { issue { id title } }
        }
        """
        # ...
```

### Step 3: Implement Processor with Mappings

```python
# pipeline/processors/integrations/linear.py
class LinearMappings:
    """Map generic types to Linear-specific values."""

    PRIORITY_MAP = {
        TaskPriority.URGENT: 1,
        TaskPriority.HIGH: 2,
        TaskPriority.MEDIUM: 3,
        TaskPriority.LOW: 4,
        TaskPriority.NONE: 0,
    }

class LinearProcessor(Processor):
    """Maps generic TaskFrame to Linear API."""

    async def process(self, frame: Frame, ctx: PipelineContext):
        if isinstance(frame, TaskFrame):
            linear_priority = LinearMappings.PRIORITY_MAP.get(
                frame.normalized_priority, 0
            )
            issue = await self.client.create_issue(
                team_id=self._resolve_team(frame.project),
                title=frame.name,
                priority=linear_priority,
            )
            return TaskResultFrame(
                success=True,
                task_id=issue.id,
                platform="linear",
            )
```

### Step 4: Export in __init__.py

```python
# pipeline/processors/integrations/__init__.py
from .linear import LinearProcessor

__all__ = ["PlaneProcessor", "LinearProcessor", "TaskCreatorProcessor"]
```

## Benefits

### 1. Processor Swappability

```python
# Same TaskFrame, different destination
task = TaskFrame(name="Fix bug", priority=TaskPriority.HIGH)

# Option A: Plane
pipeline = PipelineBuilder().add(PlaneProcessor(plane_client)).build()

# Option B: Linear
pipeline = PipelineBuilder().add(LinearProcessor(linear_client)).build()

# Option C: Multi-platform (fan-out)
pipeline = PipelineBuilder().add(MultiTaskProcessor([
    PlaneProcessor(plane_client),
    LinearProcessor(linear_client),
])).build()
```

### 2. Clean Testing

```python
def test_pipeline_with_mock_processor():
    """Test pipeline logic without hitting any API."""

    class MockTaskProcessor(Processor):
        async def process(self, frame, ctx):
            if isinstance(frame, TaskFrame):
                return TaskResultFrame(success=True, platform="mock")
            return frame

    pipeline = PipelineBuilder()
        .add(TaskCreatorProcessor())
        .add(MockTaskProcessor())  # No real API calls
        .build()
```

### 3. Future-Proof for v2

When v2 Skills layer is added:

```python
# Skills wrap processors for agent use
class TaskManagementSkill(Skill):
    """Agent-facing skill that uses v1 processors."""

    def __init__(self, processor: Processor):
        self.processor = processor  # PlaneProcessor, LinearProcessor, etc.

    async def execute(self, task: TaskFrame, ctx: SkillContext):
        result = await self.processor.process(task, ctx.pipeline_context)
        return SkillResult.from_frame(result)
```

## Domain Frame Types

| Domain | Generic Frame | Platform Processors |
|--------|--------------|---------------------|
| Task Management | `TaskFrame`, `TaskQueryFrame` | `PlaneProcessor`, `LinearProcessor`, `JiraProcessor` |
| CRM | `ContactFrame`, `DealFrame` | `TwentyProcessor`, `HubSpotProcessor`, `SalesforceProcessor` |
| Calendar | `EventFrame` | `GoogleCalendarProcessor`, `OutlookProcessor` |
| Documents | `DocumentFrame` | `NotionProcessor`, `GoogleDocsProcessor` |

## Consequences

### Positive

- **Portability**: Switch platforms without rewriting pipelines
- **Testability**: Mock processors without API dependencies
- **Consistency**: Uniform interface across all integrations
- **Extensibility**: Add new platforms by implementing one processor

### Negative

- **Mapping Overhead**: Each platform needs explicit mapping code
- **Lowest Common Denominator**: Some platform-specific features may not map to generic frames
- **Indirection**: Extra layer between business logic and API calls

### Mitigation

For platform-specific features, use `platform_config` dict:

```python
task = TaskFrame(
    name="Fix bug",
    platform_config={
        "plane": {"cycle_id": "cycle-123"},  # Plane-specific
        "linear": {"team_id": "team-456"},   # Linear-specific
    },
)
```

Processors can access their platform-specific config if available.

## References

- [Pipecat Services Architecture](https://github.com/pipecat-ai/pipecat)
- [ADR-001: Pipeline Architecture](./ADR-001-pipeline-architecture.md)
- [Strategic Roadmap v2](../roadmap/STRATEGIC_ROADMAP_V2.md)
