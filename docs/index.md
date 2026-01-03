# Knomly Documentation

**Knomly** is a Pythonic pipeline framework for building composable, event-driven workflows with first-class SaaS integrations.

Inspired by [Pipecat's](https://github.com/pipecat-ai/pipecat) elegant architecture, Knomly generalizes the pattern beyond conversational AI to **any domain**.

## The Vision

```
Swap any component. Add any integration. Zero coupling.
```

```
Today:    Webhook → Transcribe (Gemini) → Notify (Zulip) → Confirm
Tomorrow: Webhook → Transcribe (Claude) → Notify (Slack) → Create Task (Plane) → Confirm
```

## Philosophy

### Everything is a Frame

Frames are immutable data containers that flow through pipelines. A frame can hold anything — audio, text, JSON, database records, API responses, events.

### Processors are Pure Transformations

A processor takes a frame, transforms it, returns a frame. No hidden state, no side effects in the pipeline itself.

### Pipelines are Composable

Chain processors like LEGO blocks. Route conditionally. Fan out in parallel. Nest pipelines within pipelines.

### Integrations are First-Class

Every SaaS product is a potential integration. Swap providers without changing pipeline logic.

## Use Cases

Knomly is **domain-agnostic**:

- **Conversational AI** — Voice-to-chat, chatbots, agent workflows
- **Event Processing** — Webhook handlers, event-driven architectures
- **ETL Pipelines** — Extract, transform, load workflows
- **Batch Processing** — File processing, data migrations
- **Agent Workflows** — LLM tool calling, ReAct patterns

## Quick Start

```bash
pip install knomly
```

```python
from knomly import Pipeline, Processor, Frame

class Uppercase(Processor):
    @property
    def name(self) -> str:
        return "uppercase"

    async def process(self, frame, ctx):
        text = frame.data.get("text", "")
        return frame.derive(data={"text": text.upper()})

pipeline = Pipeline([Uppercase()])
result = await pipeline.execute(Frame.create("text", data={"text": "hello"}))
print(result.frames[-1].data)  # {"text": "HELLO"}
```

## Documentation Sections

### [Getting Started](getting-started/installation.md)
Installation, first pipeline, and basic concepts.

### [Core Concepts](concepts/frames.md)
Deep dive into Frames, Processors, Pipelines, and Context.

### [Architecture Decisions](adr/index.md)
ADRs documenting key design decisions.

### [API Reference](api/pipeline.md)
Complete API documentation.

### [Guides](guides/custom-processor.md)
Step-by-step tutorials for common use cases.

### [Recipes](recipes/voice-to-chat.md)
Production-ready patterns and recipes.

## Design Principles

1. **Composition over Inheritance** — Build complex behavior from simple parts
2. **Explicit over Implicit** — Clear data flow, no hidden magic
3. **Type Safety** — Full type hints for IDE support and correctness
4. **Domain Agnostic** — Framework doesn't assume your use case
5. **Provider Agnostic** — Swap integrations without changing logic
6. **Configuration-Driven** — Define pipelines in code or config

## Comparison

| Feature | Knomly | Pipecat | Langchain | Prefect |
|---------|--------|---------|-----------|---------|
| General-purpose | ✅ | ❌ (voice) | ❌ (LLM) | ✅ |
| Real-time capable | ✅ | ✅ | ⚠️ | ❌ |
| SaaS integrations | ✅ | ⚠️ | ⚠️ | ⚠️ |
| Typed Python API | ✅ | ✅ | ⚠️ | ✅ |
| Agent support | ✅ | ✅ | ✅ | ❌ |

## License

MIT License. See [LICENSE](https://github.com/kuzushi-labs/knomly/blob/main/LICENSE).

---

**Knomly** — *The missing pipeline framework for the SaaS age.*
