# Processors

Processors are **single-responsibility transformers** that process frames and produce new frames.

## Processor Protocol

Every processor must implement:

```python
from abc import ABC, abstractmethod
from knomly import Processor, PipelineContext
from knomly.pipeline.frames import Frame

class Processor(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this processor."""
        ...

    @abstractmethod
    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | Sequence[Frame] | None:
        """
        Process a frame.

        Args:
            frame: Input frame to process
            ctx: Pipeline context with providers and state

        Returns:
            - Single Frame: Transformed data
            - Sequence[Frame]: Fan-out to multiple frames
            - None: Filter out (no output)
        """
        ...
```

## Creating a Processor

```python
from knomly import Processor, PipelineContext
from knomly.pipeline.frames import Frame

class UppercaseProcessor(Processor):
    """Converts text to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | None:
        # Type check input
        if not hasattr(frame, "text"):
            return frame  # Pass through unchanged

        # Transform and return new frame
        return frame.derive(
            text=frame.text.upper(),
            source_frame_id=frame.id,
        )
```

## Return Types

### Single Frame (Transform)

Most common - transform input to output:

```python
async def process(self, frame, ctx):
    return TransformedFrame(data=process(frame.data))
```

### Multiple Frames (Fan-out)

Split one input into multiple outputs:

```python
async def process(self, frame, ctx):
    return [
        ChunkFrame(chunk=chunk, index=i)
        for i, chunk in enumerate(frame.data.split())
    ]
```

### None (Filter)

Drop the frame from the pipeline:

```python
async def process(self, frame, ctx):
    if not frame.is_valid:
        return None  # Frame is dropped
    return frame
```

## Using Pipeline Context

The context provides access to providers and state:

```python
class TranscriptionProcessor(Processor):
    @property
    def name(self) -> str:
        return "transcription"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, AudioInputFrame):
            return frame

        # Get STT provider from context
        stt = ctx.providers.get_stt()

        # Transcribe audio
        result = await stt.transcribe(
            audio_bytes=frame.audio_data,
            mime_type=frame.mime_type,
        )

        return TranscriptionFrame(
            original_text=result.text,
            detected_language=result.language,
            source_frame_id=frame.id,
        )
```

## Passthrough Processor

A built-in processor that passes frames unchanged:

```python
from knomly import PassthroughProcessor

pipeline = (
    PipelineBuilder()
    .add(PassthroughProcessor())  # Does nothing
    .add(MyProcessor())
    .build()
)
```

## Processor Lifecycle

Processors can have initialization and cleanup hooks:

```python
class DatabaseProcessor(Processor):
    def __init__(self, connection_string: str):
        self._conn_string = connection_string
        self._db = None

    @property
    def name(self) -> str:
        return "database"

    async def initialize(self, ctx: PipelineContext) -> None:
        """Called before pipeline execution."""
        self._db = await connect(self._conn_string)

    async def cleanup(self) -> None:
        """Called after pipeline execution."""
        if self._db:
            await self._db.close()

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        # Use self._db
        ...
```

## Error Handling

Exceptions in processors are caught and converted to ErrorFrames:

```python
class RiskyProcessor(Processor):
    async def process(self, frame, ctx):
        if something_wrong:
            raise ValueError("Something went wrong")
        return frame

# In pipeline execution:
# ValueError -> ErrorFrame with traceback
```

To handle errors gracefully:

```python
class SafeProcessor(Processor):
    async def process(self, frame, ctx):
        try:
            return await self._risky_operation(frame)
        except SpecificError as e:
            # Return error frame or handle differently
            return ErrorFrame.from_exception(
                exc=e,
                processor_name=self.name,
                source_frame=frame,
            )
```

## Resilient Processors

Wrap processors with retry and circuit breaker:

```python
from knomly import ResilientProcessor, RETRY_WITH_BACKOFF, CircuitBreaker

resilient = ResilientProcessor(
    processor=UnreliableAPIProcessor(),
    retry_policy=RETRY_WITH_BACKOFF,
    circuit_breaker=CircuitBreaker(
        failure_threshold=5,
        recovery_timeout=30.0,
    ),
)
```

## Best Practices

1. **Single Responsibility**: One processor, one job
2. **Type Check Input**: Validate frame type at start
3. **Return New Frames**: Never mutate input frames
4. **Use Context**: Access providers via context, not globals
5. **Handle Errors**: Catch expected errors, let unexpected propagate
6. **Keep Stateless**: Store state in context, not processor
