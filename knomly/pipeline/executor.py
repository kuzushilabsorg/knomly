"""
Pipeline Executor for Knomly.

The Pipeline executes a sequence of processors on input frames.
Adapted from Pipecat patterns for HTTP request/response context.

See ADR-001 for design decisions.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Sequence
from uuid import uuid4

from .context import PipelineContext, PipelineResult
from .frames import ErrorFrame, Frame
from .routing import PipelineExit

if TYPE_CHECKING:
    from .processor import Processor

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline orchestrates sequential frame processing.

    Execution Model (from ADR-001):
    - Frames flow sequentially through processors
    - Each processor can return 0, 1, or N frames
    - Exceptions become ErrorFrames
    - ErrorFrames continue through pipeline (can be handled)

    Example:
        pipeline = Pipeline([
            AudioDownloadProcessor(...),
            TranscriptionProcessor(...),
            ExtractionProcessor(...),
            ZulipProcessor(...),
            ConfirmationProcessor(...),
        ])

        result = await pipeline.execute(
            initial_frame=AudioInputFrame(...),
            ctx=PipelineContext(...),
        )
    """

    def __init__(self, processors: list["Processor"]):
        """
        Initialize pipeline with ordered list of processors.

        Args:
            processors: List of processors in execution order
        """
        if not processors:
            raise ValueError("Pipeline must have at least one processor")
        self.processors = processors

    @property
    def processor_names(self) -> list[str]:
        """Get names of all processors in order."""
        return [p.name for p in self.processors]

    async def execute(
        self,
        initial_frame: Frame,
        ctx: PipelineContext | None = None,
    ) -> PipelineResult:
        """
        Execute the pipeline on an initial frame.

        Args:
            initial_frame: The starting frame
            ctx: Pipeline context (created if not provided)

        Returns:
            PipelineResult with output frames and execution details
        """
        if ctx is None:
            ctx = PipelineContext()

        logger.info(
            f"Pipeline starting: execution_id={str(ctx.execution_id)[:8]}..., "
            f"processors={self.processor_names}"
        )

        result = PipelineResult(context=ctx)
        frames: list[Frame] = [initial_frame]

        for processor in self.processors:
            if not frames:
                logger.warning(f"No frames after processor '{processor.name}', stopping")
                break

            next_frames: list[Frame] = []
            start_time = time.perf_counter()

            for frame in frames:
                # Log frame entering processor
                ctx.record_frame(frame.to_dict(), processor.name)

                try:
                    output = await processor.process(frame, ctx)
                    next_frames.extend(self._normalize_output(output))
                except PipelineExit as exit_signal:
                    # Guard or other control flow triggered early exit
                    logger.info(
                        f"Pipeline exit triggered by '{processor.name}': "
                        f"returning {exit_signal.exit_frame.frame_type}"
                    )
                    result.output_frames = [exit_signal.exit_frame]
                    result.success = True
                    return result
                except Exception as e:
                    logger.error(
                        f"Processor '{processor.name}' error: {e}",
                        exc_info=True,
                    )
                    error_frame = ErrorFrame.from_exception(
                        exc=e,
                        processor_name=processor.name,
                        source_frame=frame,
                    )
                    next_frames.append(error_frame)

            # Record processor timing
            duration_ms = (time.perf_counter() - start_time) * 1000
            ctx.record_timing(processor.name, duration_ms)

            logger.debug(
                f"Processor '{processor.name}': "
                f"in={len(frames)}, out={len(next_frames)}, time={duration_ms:.1f}ms"
            )

            frames = next_frames

        # Collect results
        result.output_frames = frames

        # Check for errors
        error_frames = [f for f in frames if isinstance(f, ErrorFrame)]
        if error_frames:
            fatal_errors = [f for f in error_frames if f.is_fatal]
            if fatal_errors:
                result.success = False
                result.error = fatal_errors[0].error_message
            else:
                # Non-fatal errors: mark as partial success
                result.success = len(error_frames) < len(frames)
                if not result.success:
                    result.error = error_frames[0].error_message

        logger.info(
            f"Pipeline complete: execution_id={str(ctx.execution_id)[:8]}..., "
            f"success={result.success}, duration={ctx.elapsed_ms:.1f}ms, "
            f"output_frames={len(result.output_frames)}"
        )

        return result

    def _normalize_output(
        self,
        output: "Frame | Sequence[Frame] | None",
    ) -> list[Frame]:
        """Normalize processor output to list of frames."""
        if output is None:
            return []
        if isinstance(output, Frame):
            return [output]
        return list(output)

    def __repr__(self) -> str:
        return f"Pipeline(processors={self.processor_names})"


class PipelineBuilder:
    """
    Builder for constructing pipelines with fluent API.

    Example:
        pipeline = (
            PipelineBuilder()
            .add(AudioDownloadProcessor(...))
            .add(TranscriptionProcessor(...))
            .build()
        )
    """

    def __init__(self) -> None:
        self._processors: list["Processor"] = []

    def add(self, processor: "Processor") -> "PipelineBuilder":
        """Add a processor to the pipeline."""
        self._processors.append(processor)
        return self

    def add_if(self, condition: bool, processor: "Processor") -> "PipelineBuilder":
        """Conditionally add a processor."""
        if condition:
            self._processors.append(processor)
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline."""
        return Pipeline(self._processors)
