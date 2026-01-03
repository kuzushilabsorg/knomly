# Frames

Frames are **immutable data containers** that flow through the pipeline. They carry data between processors and maintain lineage for debugging.

## Core Properties

Every frame has these properties:

```python
@dataclass(frozen=True, kw_only=True, slots=True)
class Frame:
    id: UUID                    # Unique identifier (auto-generated)
    created_at: datetime        # Creation timestamp
    metadata: dict[str, Any]    # Arbitrary metadata
    source_frame_id: UUID | None # Parent frame ID (for lineage)
```

## Built-in Frame Types

### Input Frames

```python
from knomly.pipeline.frames import AudioInputFrame, TextInputFrame

# Audio from messaging platform
audio = AudioInputFrame(
    media_url="https://...",
    mime_type="audio/ogg",
    sender_phone="+1234567890",
    channel_id="twilio",
)

# Text message
text = TextInputFrame(
    text="Hello, world!",
    sender_phone="+1234567890",
)
```

### Processing Frames

```python
from knomly.pipeline.frames import TranscriptionFrame, ExtractionFrame

# After STT processing
transcription = TranscriptionFrame(
    original_text="Hola mundo",
    english_text="Hello world",
    detected_language="es",
    confidence=0.95,
)

# After LLM extraction
extraction = ExtractionFrame(
    today_items=("Task 1", "Task 2"),
    blockers=("Waiting on API"),
    summary="Working on features",
)
```

### Action Frames

```python
from knomly.pipeline.frames import ZulipMessageFrame, ConfirmationFrame

# Posted to chat
zulip = ZulipMessageFrame(
    stream="standup",
    topic="daily-updates",
    content="**Standup**\n- Task 1",
    message_id=12345,
    success=True,
)

# Confirmation sent
confirmation = ConfirmationFrame(
    recipient_phone="+1234567890",
    message="Your standup has been posted!",
    success=True,
)
```

### Error Frames

```python
from knomly.pipeline.frames import ErrorFrame

error = ErrorFrame.from_exception(
    exc=ValueError("Invalid input"),
    processor_name="transcription",
    source_frame=audio,
)
```

## Creating Custom Frames

```python
from dataclasses import dataclass
from knomly.pipeline.frames import Frame

@dataclass(frozen=True, kw_only=True, slots=True)
class MyCustomFrame(Frame):
    """Custom frame for my use case."""

    data: str = ""
    score: float = 0.0
    tags: tuple[str, ...] = ()

# Usage
frame = MyCustomFrame(
    data="processed result",
    score=0.95,
    tags=("important", "verified"),
)
```

## Frame Immutability

Frames are frozen dataclasses - they cannot be modified after creation:

```python
frame = TextInputFrame(text="hello")
frame.text = "world"  # ERROR: FrozenInstanceError
```

## Deriving New Frames

Use `derive()` to create a new frame with some fields changed:

```python
original = AudioInputFrame(
    media_url="https://...",
    sender_phone="+1234567890",
)

# Create derived frame with audio bytes
with_audio = original.derive(
    audio_data=downloaded_bytes,
    source_frame_id=original.id,  # Track lineage
)
```

## Frame Lineage

Every frame can track its parent via `source_frame_id`:

```python
audio = AudioInputFrame(media_url="...")
transcription = TranscriptionFrame(
    text="Hello",
    source_frame_id=audio.id,  # Links to parent
)
extraction = ExtractionFrame(
    today_items=("Task",),
    source_frame_id=transcription.id,  # Links to parent
)

# Build lineage chain: audio -> transcription -> extraction
```

## Frame Serialization

Convert frames to dictionaries for logging or storage:

```python
frame = AudioInputFrame(
    media_url="https://example.com/audio.ogg",
    sender_phone="+1234567890",
)

data = frame.to_dict()
# {
#     'id': 'abc123...',
#     'frame_type': 'AudioInputFrame',
#     'created_at': '2024-01-01T12:00:00Z',
#     'media_url': 'https://example.com/audio.ogg',
#     'sender_phone': '+1234567890',
#     ...
# }
```

## Best Practices

1. **Keep frames immutable**: Never mutate, always derive
2. **Track lineage**: Set `source_frame_id` when creating derived frames
3. **Use tuples for collections**: `tuple[str, ...]` not `list[str]`
4. **Add metadata**: Use the `metadata` dict for debugging info
5. **Be specific**: Create domain-specific frame types
