# Knomly

[![PyPI version](https://badge.fury.io/py/knomly.svg)](https://badge.fury.io/py/knomly)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://github.com/kuzushi-labs/knomly/actions/workflows/test.yml/badge.svg)](https://github.com/kuzushi-labs/knomly/actions/workflows/test.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Knomly** is a Pythonic pipeline framework for building composable, event-driven workflows with first-class SaaS integrations.

Inspired by [Pipecat's](https://github.com/pipecat-ai/pipecat) elegant architecture, Knomly generalizes the pattern beyond conversational AI to **any domain** — ETL pipelines, webhook handlers, batch processing, agent workflows, and yes, voice applications too.

## The Vision

```
Today:    Webhook → Transcribe (Gemini) → Notify (Zulip) → Confirm
Tomorrow: Webhook → Transcribe (Claude) → Notify (Slack + Zulip) → Create Task (Plane) → Confirm
                              ↑                    ↑                        ↑
                          one-line swap      one-line add              one-line add
```

**Swap any component. Add any integration. Zero coupling.**

## Philosophy

### Everything is a Frame

Frames are immutable data containers that flow through pipelines. A frame can hold anything — audio, text, JSON, database records, API responses, events.

```python
frame = Frame.create("webhook", data={"event": "push", "repo": "knomly"})
frame = Frame.create("audio", data={"bytes": audio_data, "mime": "audio/ogg"})
frame = Frame.create("task", data={"title": "Fix bug", "project": "Mobile"})
```

### Processors are Pure Transformations

A processor takes a frame, transforms it, returns a frame. That's it.

```python
class MyProcessor(Processor):
    async def process(self, frame: Frame, ctx: Context) -> Frame | None:
        # Transform → Return
        return frame.derive(data={**frame.data, "processed": True})
```

### Pipelines are Composable

Chain processors like LEGO blocks. Route conditionally. Fan out in parallel.

```python
pipeline = Pipeline([
    ValidateInput(),
    Conditional(
        when=lambda f: f.data.get("type") == "audio",
        then=Pipeline([Transcribe(), ExtractIntent()]),
        else=Pipeline([ParseText()]),
    ),
    CreateTask(provider="plane"),
    Notify(provider="zulip"),
])
```

### Integrations are First-Class

Every SaaS product is a potential integration. Swap providers without changing pipeline logic.

```python
# Version 1: Gemini + Zulip
pipeline = Pipeline([
    Transcribe(provider="gemini"),
    Notify(provider="zulip"),
])

# Version 2: Claude + Slack (one-line changes)
pipeline = Pipeline([
    Transcribe(provider="anthropic"),  # ← swapped
    Notify(provider="slack"),          # ← swapped
])
```

## Installation

```bash
pip install knomly
```

With optional dependencies:

```bash
# Full installation
pip install knomly[full]

# Specific integrations
pip install knomly[llm-openai,llm-anthropic,chat-zulip,transport-twilio]

# Development
pip install knomly[dev]
```

## Use Cases

Knomly is **domain-agnostic**. Here are examples across different domains:

### Voice-to-Chat (Conversational AI)

```python
pipeline = Pipeline([
    DownloadAudio(),
    Transcribe(provider="gemini"),
    ExtractIntent(provider="openai"),
    Conditional(
        when=lambda f: f.data["intent"] == "create_task",
        then=CreateTask(provider="plane"),
    ),
    PostMessage(provider="zulip"),
    SendConfirmation(provider="twilio"),
])
```

### Webhook Event Processing

```python
pipeline = Pipeline([
    ParseWebhook(schema=GitHubPushEvent),
    Conditional(
        when=lambda f: f.data["ref"] == "refs/heads/main",
        then=Pipeline([
            TriggerBuild(provider="github_actions"),
            Notify(provider="slack", channel="#deployments"),
        ]),
    ),
])
```

### ETL Pipeline

```python
pipeline = Pipeline([
    FetchFromAPI(provider="salesforce", query="SELECT * FROM Lead"),
    Transform(enrich_leads),
    ValidateSchema(schema=LeadSchema),
    LoadToDatabase(provider="postgres", table="leads"),
    Notify(provider="slack", message="ETL complete: {count} records"),
])
```

### Batch Processing

```python
pipeline = Pipeline([
    ListFiles(provider="s3", bucket="uploads", pattern="*.csv"),
    FanOut(
        processor=Pipeline([
            DownloadFile(),
            ParseCSV(),
            ValidateRows(),
            InsertToDatabase(),
        ]),
        strategy="parallel",
        max_concurrency=10,
    ),
    AggregateResults(),
    SendReport(provider="email"),
])
```

### Agent Workflow (Tool Calling)

```python
pipeline = Pipeline([
    ParseUserRequest(),
    AgentBridge(
        llm=OpenAILLM(),
        tools=[
            PlaneCreateTask(),
            ZulipSendMessage(),
            SlackPostMessage(),
        ],
        max_iterations=5,
    ),
    FormatResponse(),
])
```

## Quick Start

```python
import asyncio
from knomly import Pipeline, Processor, Frame, PipelineContext

class Uppercase(Processor):
    @property
    def name(self) -> str:
        return "uppercase"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
        text = frame.data.get("text", "")
        return frame.derive(data={"text": text.upper()})

class AddTimestamp(Processor):
    @property
    def name(self) -> str:
        return "timestamp"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame:
        from datetime import datetime
        return frame.derive(data={**frame.data, "processed_at": datetime.now().isoformat()})

# Build pipeline
pipeline = Pipeline([Uppercase(), AddTimestamp()])

# Execute
async def main():
    frame = Frame.create("text", data={"text": "hello world"})
    result = await pipeline.execute(frame)
    print(result.frames[-1].data)
    # {'text': 'HELLO WORLD', 'processed_at': '2024-01-15T10:30:00'}

asyncio.run(main())
```

## Core Concepts

### Frames

Frames are immutable, typed containers for any data:

```python
from knomly.pipeline.frames import Frame

# Generic frame
frame = Frame.create("event", data={"type": "user.created", "user_id": 123})

# Derive new frame (immutable)
new_frame = frame.derive(data={**frame.data, "processed": True})

# With metadata
frame = Frame.create("task", data=task_data, metadata={"tenant_id": "acme"})
```

### Processors

Processors are the atomic units of transformation:

```python
from knomly import Processor

class ValidateProcessor(Processor):
    @property
    def name(self) -> str:
        return "validate"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not frame.data.get("required_field"):
            return None  # Filter out invalid frames
        return frame
```

### Routing

Control flow with routing primitives:

```python
from knomly.pipeline.routing import Conditional, Switch, FanOut, TypeRouter

# Conditional routing
Conditional(
    when=lambda f: f.data.get("priority") == "high",
    then=UrgentHandler(),
    else=NormalHandler(),
)

# Switch on value
Switch(
    key=lambda f: f.data.get("type"),
    cases={
        "create": CreateHandler(),
        "update": UpdateHandler(),
        "delete": DeleteHandler(),
    },
    default=UnknownHandler(),
)

# Parallel fan-out
FanOut(
    processors=[NotifySlack(), NotifyEmail(), NotifyWebhook()],
    strategy="parallel",
)

# Route by frame type
TypeRouter({
    "audio": AudioPipeline(),
    "text": TextPipeline(),
    "image": ImagePipeline(),
})
```

### Integrations

SaaS integrations follow a consistent pattern:

```python
from knomly.integrations.plane import PlaneClient, PlaneConfig

# Initialize integration
plane = PlaneClient(PlaneConfig(
    api_key="...",
    workspace="my-workspace",
))

# Use directly
task = await plane.create_issue(
    project="Mobile App",
    title="Fix login bug",
    priority="high",
)

# Or use as processor in pipeline
pipeline = Pipeline([
    ParseRequest(),
    CreateTask(provider="plane"),  # Uses registered integration
    Notify(provider="zulip"),
])
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              INPUT LAYER                                     │
│  Webhooks │ Message Queues │ Scheduled Jobs │ API Calls │ File Watchers    │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ Frames
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                            PIPELINE LAYER                                    │
│                                                                              │
│   ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐  │
│   │Processor│───▶│Processor│───▶│ Router  │───▶│Processor│───▶│Processor│  │
│   └─────────┘    └─────────┘    └────┬────┘    └─────────┘    └─────────┘  │
│                                      │                                       │
│                         ┌────────────┼────────────┐                         │
│                         ▼            ▼            ▼                         │
│                    ┌────────┐  ┌────────┐  ┌────────┐                       │
│                    │Pipeline│  │Pipeline│  │Pipeline│                       │
│                    └────────┘  └────────┘  └────────┘                       │
│                                                                              │
└──────────────────────────────────┬──────────────────────────────────────────┘
                                   │ Frames
┌──────────────────────────────────▼──────────────────────────────────────────┐
│                          INTEGRATION LAYER                                   │
│                                                                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐          │
│  │  Plane   │ │  Zulip   │ │  Slack   │ │  OpenAI  │ │  Stripe  │  ...     │
│  │(Projects)│ │ (Chat)   │ │ (Chat)   │ │  (LLM)   │ │(Payments)│          │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Integrations

Knomly provides a growing library of SaaS integrations:

### AI/ML Services
| Provider | Type | Package |
|----------|------|---------|
| OpenAI | LLM, STT, TTS | `knomly[llm-openai]` |
| Anthropic | LLM | `knomly[llm-anthropic]` |
| Google Gemini | LLM, STT | `knomly[llm-gemini,stt-gemini]` |
| Deepgram | STT | `knomly[stt-deepgram]` |

### Communication
| Provider | Type | Package |
|----------|------|---------|
| Zulip | Team Chat | `knomly[chat-zulip]` |
| Twilio | SMS, WhatsApp, Voice | `knomly[transport-twilio]` |
| *(coming)* Slack | Team Chat | `knomly[chat-slack]` |

### Project Management
| Provider | Type | Package |
|----------|------|---------|
| Plane | Issues, Projects | built-in |
| *(coming)* Linear | Issues, Projects | `knomly[pm-linear]` |
| *(coming)* Jira | Issues, Projects | `knomly[pm-jira]` |

### Adding Your Own Integration

```python
from knomly.integrations.base import IntegrationClient, IntegrationConfig

class MyServiceConfig(IntegrationConfig):
    api_key: str
    base_url: str = "https://api.myservice.com"

class MyServiceClient(IntegrationClient):
    def __init__(self, config: MyServiceConfig):
        super().__init__(config)

    @property
    def name(self) -> str:
        return "myservice"

    def _get_auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.config.api_key}"}

    async def do_something(self, data: dict) -> dict:
        response = await self._request("POST", "/api/action", json=data)
        return response.json()
```

## Configuration

### Environment Variables

```bash
# Core
KNOMLY_ENVIRONMENT=production
KNOMLY_DEBUG=false

# AI Providers
KNOMLY_OPENAI_API_KEY=sk-...
KNOMLY_ANTHROPIC_API_KEY=sk-ant-...
KNOMLY_GEMINI_API_KEY=...

# Integrations
KNOMLY_PLANE_API_KEY=...
KNOMLY_ZULIP_SITE=https://chat.example.com
KNOMLY_ZULIP_BOT_EMAIL=bot@example.com
KNOMLY_ZULIP_API_KEY=...

# Transport
KNOMLY_TWILIO_ACCOUNT_SID=...
KNOMLY_TWILIO_AUTH_TOKEN=...

# Database (for config storage)
KNOMLY_MONGODB_URL=mongodb://localhost:27017
```

### Configuration-Driven Pipelines

For multi-tenant SaaS applications, define pipelines in configuration:

```yaml
# pipelines/customer-support.yaml
name: customer-support
processors:
  - type: webhook_input
    provider: twilio
  - type: transcribe
    provider: gemini
  - type: classify_intent
    provider: openai
  - type: conditional
    when: "intent == 'create_task'"
    then:
      - type: create_task
        provider: plane
        config:
          project: "{{tenant.default_project}}"
    else:
      - type: respond
        message: "I couldn't understand that request."
  - type: notify
    provider: zulip
    config:
      stream: "{{tenant.notification_stream}}"
```

```python
from knomly.runtime import PipelineResolver

resolver = PipelineResolver(loader=FileDefinitionLoader("pipelines/"))
pipeline = await resolver.resolve_for_user(user_id="tenant-123")
result = await pipeline.execute(initial_frame)
```

## Built-in Resilience

### Retry with Backoff

```python
from knomly.pipeline.retry import with_retry, RetryPolicy, ExponentialBackoff

@with_retry(RetryPolicy(
    max_retries=3,
    backoff=ExponentialBackoff(base=1.0, max_delay=30.0),
    retryable_exceptions=[ConnectionError, TimeoutError],
))
async def unreliable_operation():
    ...
```

### Circuit Breaker

```python
from knomly.pipeline.retry import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,
    recovery_timeout=30.0,
)

async with breaker:
    await external_service_call()
```

### Rate Limiting

```python
from knomly.pipeline.ratelimit import RateLimiter, TokenBucket

limiter = RateLimiter(TokenBucket(rate=10, capacity=100))

async with limiter:
    await api_call()
```

## Observability

### Structured Logging

```python
from knomly.pipeline.observability import PipelineLogger

logger = PipelineLogger(service="my-service")
logger.info("Processing frame", frame_id=frame.id, frame_type=frame.frame_type)
```

### Metrics

```python
from knomly.pipeline.observability import get_metrics

metrics = get_metrics()
metrics.increment("frames_processed", tags={"type": frame.frame_type})
metrics.histogram("processing_time", duration_ms, tags={"processor": processor.name})
```

### Distributed Tracing

```python
from knomly.pipeline.observability import get_tracer

tracer = get_tracer()
with tracer.span("process_frame") as span:
    span.set_attribute("frame.type", frame.frame_type)
    result = await processor.process(frame, ctx)
```

## Development

```bash
# Clone
git clone https://github.com/kuzushi-labs/knomly.git
cd knomly

# Install
pip install -e ".[dev,docs]"

# Test
pytest

# Lint
ruff check . && ruff format .

# Type check
mypy knomly

# Docs
mkdocs serve
```

## Project Structure

```
knomly/
├── knomly/
│   ├── pipeline/           # Core pipeline framework
│   │   ├── frames/         # Frame types
│   │   ├── processor.py    # Base processor
│   │   ├── executor.py     # Pipeline execution
│   │   ├── routing.py      # Routing primitives
│   │   ├── retry.py        # Resilience patterns
│   │   └── observability.py
│   ├── integrations/       # SaaS integrations
│   │   ├── base.py         # Integration base classes
│   │   ├── plane/          # Plane.so integration
│   │   └── ...
│   ├── providers/          # AI service providers
│   │   ├── llm/            # LLM providers
│   │   ├── stt/            # Speech-to-text
│   │   └── chat/           # Chat providers
│   ├── agent/              # Agent layer (tool calling)
│   ├── runtime/            # Dynamic configuration
│   └── adapters/           # External adapters
├── tests/
├── examples/
└── docs/
```

## Comparison

| Feature | Knomly | Pipecat | Langchain | Prefect |
|---------|--------|---------|-----------|---------|
| General-purpose pipelines | ✅ | ❌ (voice) | ❌ (LLM) | ✅ |
| Real-time capable | ✅ | ✅ | ⚠️ | ❌ |
| SaaS integrations | ✅ | ⚠️ (AI only) | ⚠️ (AI only) | ⚠️ |
| Typed Python API | ✅ | ✅ | ⚠️ | ✅ |
| Configuration-driven | ✅ | ❌ | ⚠️ | ✅ |
| Agent/tool support | ✅ | ✅ | ✅ | ❌ |
| Built-in resilience | ✅ | ⚠️ | ⚠️ | ✅ |

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Adding a new integration?** That's the best way to contribute. Every SaaS product is a potential integration.

```bash
# Create integration structure
mkdir -p knomly/integrations/myservice
touch knomly/integrations/myservice/{__init__,client,frames,processors,schemas}.py
```

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

- [Pipecat](https://github.com/pipecat-ai/pipecat) — The elegant architecture we generalized
- [FastAPI](https://fastapi.tiangolo.com/) — Inspiration for developer experience
- [Pydantic](https://docs.pydantic.dev/) — The foundation for type safety

---

**Knomly** — *The missing pipeline framework for the SaaS age.*
