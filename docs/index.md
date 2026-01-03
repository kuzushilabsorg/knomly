# Knomly Documentation

**Knomly** is a Pipecat-inspired pipeline framework for building AI-powered voice and messaging applications.

## Features

- **Modular Pipeline Architecture**: Build complex workflows from simple, composable processors
- **Type-Safe Frames**: Immutable data containers with full type hints
- **Flexible Routing**: Conditional, Switch, TypeRouter, Guard, and FanOut patterns
- **Transport Abstraction**: Support any messaging platform (WhatsApp, Telegram, Slack, etc.)
- **Provider System**: Pluggable STT, LLM, and Chat integrations
- **Built-in Resilience**: Retry policies, circuit breakers, and rate limiting
- **Observability**: Structured logging, metrics, and distributed tracing

## Quick Start

```bash
pip install knomly
```

```python
from knomly import Pipeline, PipelineBuilder, Processor

class MyProcessor(Processor):
    @property
    def name(self) -> str:
        return "my_processor"

    async def process(self, frame, ctx):
        # Transform the frame
        return frame.derive(data=processed_data)

# Build and execute
pipeline = PipelineBuilder().add(MyProcessor()).build()
result = await pipeline.execute(initial_frame)
```

## Documentation Sections

### [Getting Started](getting-started/installation.md)
Installation, first pipeline, and basic concepts.

### [Core Concepts](concepts/frames.md)
Deep dive into Frames, Processors, Pipelines, and Context.

### [Guides](guides/custom-processor.md)
Step-by-step tutorials for common use cases.

### [API Reference](api/pipeline.md)
Complete API documentation.

### [Recipes](recipes/voice-to-chat.md)
Production-ready patterns and recipes.

## Philosophy

Knomly follows these design principles:

1. **Composition over Inheritance**: Build complex behavior from simple parts
2. **Explicit over Implicit**: Clear data flow, no hidden magic
3. **Type Safety**: Full type hints for IDE support and correctness
4. **Transport Agnostic**: Framework doesn't care how messages arrive or leave
5. **Provider Agnostic**: Swap AI services without changing pipeline logic

## License

BSD-2-Clause License. See [LICENSE](https://github.com/kuzushi-labs/knomly/blob/main/LICENSE).
