# Building Custom Processors

This guide walks you through creating custom processors for your Knomly pipelines.

## Overview

Processors are the core building blocks of Knomly pipelines. Each processor:

- Has a single responsibility
- Receives a Frame and context
- Returns a ProcessorResult

## Basic Processor

```python
from knomly.pipeline.processors.base import Processor, ProcessorResult
from knomly.pipeline.frames.base import Frame

class MyProcessor(Processor):
    """A simple processor that adds metadata to frames."""

    @property
    def name(self) -> str:
        return "my_processor"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        # Add metadata to the frame
        new_frame = frame.with_metadata({"processed_by": self.name})
        return ProcessorResult.ok(new_frame)
```

## Processor with Configuration

```python
from dataclasses import dataclass

@dataclass
class TransformConfig:
    uppercase: bool = False
    strip_whitespace: bool = True

class TransformProcessor(Processor):
    def __init__(self, config: TransformConfig):
        self._config = config

    @property
    def name(self) -> str:
        return "transform"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        content = frame.data.get("content", "")

        if self._config.strip_whitespace:
            content = content.strip()

        if self._config.uppercase:
            content = content.upper()

        return ProcessorResult.ok(frame.derive(data={"content": content}))
```

## Skipping Frames

Use `ProcessorResult.skip()` when a processor shouldn't handle a frame:

```python
class AudioOnlyProcessor(Processor):
    @property
    def name(self) -> str:
        return "audio_only"

    async def can_process(self, frame: Frame) -> bool:
        return frame.frame_type == "audio"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        if frame.frame_type != "audio":
            return ProcessorResult.skip("Not an audio frame")

        # Process audio...
        return ProcessorResult.ok(processed_frame)
```

## Error Handling

Use `ProcessorResult.error()` for recoverable errors:

```python
class ValidatingProcessor(Processor):
    @property
    def name(self) -> str:
        return "validator"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        try:
            validated = self._validate(frame)
            return ProcessorResult.ok(validated)
        except ValidationError as e:
            return ProcessorResult.error(f"Validation failed: {e}")
```

## Using Context

Access pipeline context for shared state:

```python
class ContextAwareProcessor(Processor):
    @property
    def name(self) -> str:
        return "context_aware"

    async def process(self, frame: Frame, ctx) -> ProcessorResult:
        # Access user ID from context
        user_id = ctx.user_id

        # Access metadata
        locale = ctx.metadata.get("locale", "en")

        # Use providers
        llm = ctx.providers.get_llm()
        response = await llm.complete(...)

        return ProcessorResult.ok(frame.derive(data={"response": response}))
```

## Processor Chaining

Processors execute in order. Each processor receives the output of the previous:

```python
pipeline = (
    PipelineBuilder()
    .add(ValidateProcessor())      # Validates input
    .add(EnrichProcessor())        # Adds context
    .add(TransformProcessor())     # Transforms data
    .add(OutputProcessor())        # Sends output
    .build()
)
```

## Testing Processors

```python
import pytest
from knomly.pipeline.frames.base import Frame
from knomly.pipeline.context import PipelineContext

@pytest.mark.asyncio
async def test_my_processor():
    # Create processor
    processor = MyProcessor()

    # Create test frame
    frame = Frame.create("test", data={"content": "hello"})

    # Create mock context
    ctx = PipelineContext(session_id="test", user_id="test-user")

    # Process
    result = await processor.process(frame, ctx)

    # Assert
    assert result.success
    assert "processed_by" in result.frame.metadata
```

## Best Practices

1. **Single Responsibility**: Each processor should do one thing well
2. **Immutability**: Never modify the input frame; always return a new frame
3. **Type Safety**: Use type hints for all parameters and returns
4. **Error Handling**: Use ProcessorResult.error() for recoverable errors
5. **Logging**: Use structured logging with processor name prefix
6. **Testing**: Test processors in isolation with mock context
