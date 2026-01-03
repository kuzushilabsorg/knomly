"""
Simple Pipeline Example

This example demonstrates the basic pipeline pattern:
1. Create custom processors
2. Build a pipeline
3. Execute with frames

Run: python -m examples.01-simple-pipeline.main
"""

import asyncio
from dataclasses import dataclass

from knomly import PipelineBuilder, PipelineContext, Processor
from knomly.pipeline.frames import Frame

# =============================================================================
# Custom Frames
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class TextFrame(Frame):
    """Simple text frame."""

    text: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class UppercaseFrame(Frame):
    """Frame with uppercase text."""

    text: str = ""
    original: str = ""


@dataclass(frozen=True, kw_only=True, slots=True)
class CountFrame(Frame):
    """Frame with word count."""

    text: str = ""
    word_count: int = 0


# =============================================================================
# Custom Processors
# =============================================================================


class UppercaseProcessor(Processor):
    """Converts text to uppercase."""

    @property
    def name(self) -> str:
        return "uppercase"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, TextFrame):
            return frame

        return UppercaseFrame(
            text=frame.text.upper(),
            original=frame.text,
            source_frame_id=frame.id,
        )


class WordCountProcessor(Processor):
    """Counts words in text."""

    @property
    def name(self) -> str:
        return "word_count"

    async def process(self, frame: Frame, ctx: PipelineContext) -> Frame | None:
        if not isinstance(frame, UppercaseFrame):
            return frame

        words = frame.text.split()
        return CountFrame(
            text=frame.text,
            word_count=len(words),
            source_frame_id=frame.id,
        )


# =============================================================================
# Main
# =============================================================================


async def main():
    # Build pipeline
    pipeline = PipelineBuilder().add(UppercaseProcessor()).add(WordCountProcessor()).build()

    print(f"Pipeline: {pipeline}")
    print(f"Processors: {pipeline.processor_names}")
    print()

    # Execute with a text frame
    initial_frame = TextFrame(text="Hello world from Knomly pipeline!")

    result = await pipeline.execute(initial_frame)

    print(f"Success: {result.success}")
    print(f"Duration: {result.duration_ms:.2f}ms")
    print()

    # Get output
    output = result.get_frame(CountFrame)
    if output:
        print(f"Original text: {initial_frame.text}")
        print(f"Uppercase: {output.text}")
        print(f"Word count: {output.word_count}")


if __name__ == "__main__":
    asyncio.run(main())
