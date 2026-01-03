# Quickstart

This guide walks you through building your first Knomly pipeline in 5 minutes.

## Prerequisites

- Python 3.10+
- `pip` or `uv` package manager

## Installation

```bash
pip install knomly
```

Or with uv:
```bash
uv add knomly
```

## Your First Pipeline

Let's build a simple pipeline that transforms text.

### Step 1: Create a Frame

Frames are immutable data containers that flow through your pipeline.

```python
from knomly.pipeline.frames.base import Frame

# Create a simple text frame
frame = Frame.create("text", data={"content": "Hello, World!"})
```

### Step 2: Create a Processor

Processors transform frames. Each processor has a single responsibility.

```python
from knomly.pipeline.processors.base import Processor, ProcessorResult

class UppercaseProcessor(Processor):
    """Converts text content to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        content = frame.data.get("content", "")
        new_data = {"content": content.upper()}
        return ProcessorResult.ok(frame.derive(data=new_data))
```

### Step 3: Build and Execute

```python
import asyncio
from knomly.pipeline import PipelineBuilder

async def main():
    # Build the pipeline
    pipeline = (
        PipelineBuilder()
        .add(UppercaseProcessor())
        .build()
    )

    # Create initial frame
    frame = Frame.create("text", data={"content": "Hello, World!"})

    # Execute
    result = await pipeline.execute(frame)

    print(result.frames[-1].data["content"])
    # Output: HELLO, WORLD!

asyncio.run(main())
```

## Adding Multiple Processors

Chain processors for complex transformations:

```python
class ReverseProcessor(Processor):
    """Reverses text content."""

    @property
    def name(self) -> str:
        return "reverse"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        content = frame.data.get("content", "")
        new_data = {"content": content[::-1]}
        return ProcessorResult.ok(frame.derive(data=new_data))

# Chain processors
pipeline = (
    PipelineBuilder()
    .add(UppercaseProcessor())
    .add(ReverseProcessor())
    .build()
)

# Input: "Hello" -> uppercase -> "HELLO" -> reverse -> "OLLEH"
```

## Using Providers

Knomly includes provider abstractions for AI services.

### LLM Provider

```python
from knomly.providers.llm.openai import OpenAILLMProvider
from knomly.providers.llm import LLMConfig, Message

# Initialize provider
llm = OpenAILLMProvider(api_key="your-api-key")

# Generate completion
response = await llm.complete(
    messages=[
        Message.system("You are a helpful assistant."),
        Message.user("What is 2 + 2?"),
    ],
    config=LLMConfig(temperature=0.7),
)

print(response.content)
```

### STT Provider

```python
from knomly.providers.stt.gemini import GeminiSTTProvider

stt = GeminiSTTProvider(api_key="your-gemini-key")

# Transcribe audio
result = await stt.transcribe(audio_bytes, mime_type="audio/ogg")
print(result.text)
```

## Routing Frames

Use routing processors for conditional logic:

```python
from knomly.pipeline.routing import ConditionalRouter

# Route based on frame type
router = ConditionalRouter(
    condition=lambda f: f.frame_type == "audio",
    true_processor=AudioProcessor(),
    false_processor=TextProcessor(),
)
```

## Environment Variables

Configure Knomly using environment variables:

```bash
# LLM Providers
KNOMLY_OPENAI_API_KEY=sk-...
KNOMLY_ANTHROPIC_API_KEY=sk-ant-...

# STT/TTS Providers
KNOMLY_GEMINI_API_KEY=...
KNOMLY_DEEPGRAM_API_KEY=...

# Integrations
KNOMLY_PLANE_API_KEY=...
KNOMLY_ZULIP_API_KEY=...
```

## Next Steps

- [Core Concepts: Frames](../concepts/frames.md) - Understand the data model
- [Core Concepts: Processors](../concepts/processors.md) - Build custom processors
- [API Reference](../api/pipeline.md) - Complete API documentation
- [Recipes](../recipes/voice-to-chat.md) - Production patterns
