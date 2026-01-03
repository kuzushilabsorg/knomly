"""
Processor abstraction for Knomly pipeline.

Processors are modular workers that transform frames.
Adapted from Pipecat patterns for HTTP request/response context.

See ADR-001 for design decisions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .context import PipelineContext
    from .frames import Frame

logger = logging.getLogger(__name__)


# Type alias for processor return value
ProcessorResult = "Frame | Sequence[Frame] | None"


class Processor(ABC):
    """
    Base class for all processors in the Knomly pipeline.

    Processors are single-responsibility transformers that:
    - Receive a frame and context
    - Return transformed frame(s), or None to stop

    Design Principles (from ADR-001):
    - Pure functions: process(Frame, Context) → Frame | list[Frame] | None
    - Explicit returns (not generators/iterators)
    - Errors become ErrorFrames (pipeline catches exceptions)

    Lifecycle Methods:
    - initialize(): Called once when pipeline starts (optional)
    - process(): The main transformation logic (required)
    - cleanup(): Called when pipeline completes (optional)

    Pipeline Integration:
    - process_frame(): Async generator wrapper for pipeline compatibility
      Wraps process() results and yields them for the pipeline executor.

    Subclasses must implement:
    - name: Unique processor identifier
    - process(): The transformation logic
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this processor, used in logging and metrics."""
        ...

    async def initialize(self, context: PipelineContext) -> None:
        """
        Initialize processor with pipeline context.

        Called once when pipeline starts. Override for:
        - Establishing connections
        - Loading configuration
        - Initializing clients

        Default implementation does nothing.
        """
        pass

    async def cleanup(self) -> None:
        """
        Cleanup processor resources.

        Called when pipeline completes or errors. Override for:
        - Closing connections
        - Flushing buffers
        - Releasing resources

        Default implementation does nothing.
        """
        pass

    @abstractmethod
    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> ProcessorResult:
        """
        Transform input frame.

        Args:
            frame: Input frame to process
            ctx: Pipeline context with providers and state

        Returns:
            Frame: Continue pipeline with single frame
            Sequence[Frame]: Fan-out to multiple frames
            None: Stop pipeline (frame consumed)

        Raises:
            Exception: Pipeline wraps in ErrorFrame
        """
        ...

    async def process_frame(
        self,
        frame: Frame,
        context: PipelineContext,
    ) -> AsyncIterator[Frame]:
        """
        Async generator wrapper for pipeline compatibility.

        Calls process() and yields results for the pipeline executor.
        Handles error wrapping for consistent pipeline behavior.

        Args:
            frame: Input frame to process
            context: Pipeline context with providers and state

        Yields:
            Frame: Output frames from process()
        """
        from .frames import ErrorFrame

        try:
            result = await self.process(frame, context)

            if result is None:
                # None = frame consumed, stop propagation
                return
            elif isinstance(result, Sequence) and not isinstance(result, str | bytes):
                # Sequence of frames - yield each
                for output_frame in result:
                    yield output_frame
            else:
                # Single frame
                yield result

        except Exception as e:
            logger.error(
                f"Processor '{self.name}' error processing {frame.frame_type}: {e}",
                exc_info=True,
            )
            yield ErrorFrame.from_exception(
                exc=e,
                processor_name=self.name,
                source_frame=frame,
            )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class PassthroughProcessor(Processor):
    """
    A no-op processor that passes frames through unchanged.

    Useful for testing and as a placeholder.
    """

    @property
    def name(self) -> str:
        return "passthrough"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> ProcessorResult:
        return frame
