"""
Pipeline Orchestrator for Knomly.

The Pipeline executes a sequence of processors on input frames,
managing flow control, error handling, and result collection.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, List, Optional

from .context import PipelineContext, PipelineResult
from .frames import EndFrame, ErrorFrame, Frame, StartFrame

if TYPE_CHECKING:
    from .processor import Processor

logger = logging.getLogger(__name__)


class Pipeline:
    """
    Pipeline orchestrates frame flow through a sequence of processors.

    Execution Model:
    - Frames flow sequentially through processors
    - Each processor can yield 0..N frames (fan-out supported)
    - ErrorFrames flow through but can be handled by error-aware processors
    - Pipeline completes when all frames have been processed

    Lifecycle:
    1. __init__: Configure pipeline with processors
    2. execute(): Run pipeline on initial frame(s)
       a. Initialize all processors
       b. Emit StartFrame
       c. Process frames through each processor
       d. Collect output frames
       e. Emit EndFrame
       f. Cleanup all processors
    3. Return PipelineResult with outputs and timing

    Example:
        pipeline = Pipeline([
            AudioDownloadProcessor(),
            TranscriptionProcessor(),
            StandupExtractionProcessor(),
            ZulipProcessor(),
            ConfirmationProcessor(),
        ])

        result = await pipeline.execute(
            initial_frame=AudioInputFrame(source_url="..."),
            context=PipelineContext(sender_phone="+1234567890"),
        )
    """

    def __init__(self, processors: List["Processor"]):
        """
        Initialize pipeline with ordered list of processors.

        Args:
            processors: List of processors in execution order
        """
        self.processors = processors
        self._initialized = False
        self._initialized_processors: List["Processor"] = []

    @property
    def processor_names(self) -> List[str]:
        """Get names of all processors in order."""
        return [p.name for p in self.processors]

    async def _initialize_processors(self, context: PipelineContext) -> None:
        """
        Initialize all processors with context.

        Tracks which processors were successfully initialized so cleanup
        can handle partial initialization failures gracefully.
        """
        self._initialized_processors = []
        for processor in self.processors:
            try:
                await processor.initialize(context)
                self._initialized_processors.append(processor)
            except Exception as e:
                logger.error(
                    f"Failed to initialize processor '{processor.name}': {e}"
                )
                # Re-raise after logging - cleanup will handle partial init
                raise
        self._initialized = True
        logger.info(f"Pipeline initialized with {len(self.processors)} processors")

    async def _cleanup_processors(self) -> None:
        """
        Cleanup all initialized processors.

        Only cleans up processors that were successfully initialized,
        handles partial initialization failures gracefully.
        """
        # Only cleanup processors that were actually initialized
        for processor in self._initialized_processors:
            try:
                await processor.cleanup()
            except Exception as e:
                logger.error(f"Error cleaning up processor {processor.name}: {e}")

        self._initialized_processors = []
        self._initialized = False

    async def _process_frame_through_pipeline(
        self,
        frame: Frame,
        context: PipelineContext,
    ) -> List[Frame]:
        """
        Process a single frame through all processors.

        Returns list of output frames from the final processor.
        """
        current_frames = [frame]

        for processor in self.processors:
            next_frames = []
            processor_start = time.perf_counter()

            for current_frame in current_frames:
                # Record frame in audit trail
                context.record_frame(current_frame.to_dict(), processor.name)

                # Process frame through this processor
                async for output_frame in processor.process_frame(current_frame, context):
                    next_frames.append(output_frame)

            # Record processor timing
            processor_duration = (time.perf_counter() - processor_start) * 1000
            context.record_timing(processor.name, processor_duration)

            logger.debug(
                f"Processor '{processor.name}' produced {len(next_frames)} frames "
                f"in {processor_duration:.1f}ms"
            )

            # Move to next processor with output frames
            current_frames = next_frames

            # Short-circuit if no frames to process
            if not current_frames:
                logger.warning(f"No frames after processor '{processor.name}', stopping pipeline")
                break

        return current_frames

    async def execute(
        self,
        initial_frame: Frame,
        context: Optional[PipelineContext] = None,
    ) -> PipelineResult:
        """
        Execute the pipeline on an initial frame.

        Args:
            initial_frame: The starting frame (typically AudioInputFrame)
            context: Pipeline context (created if not provided)

        Returns:
            PipelineResult with output frames and execution details
        """
        # Create context if not provided
        if context is None:
            context = PipelineContext()

        logger.info(
            f"Pipeline starting execution {context.execution_id[:8]}... "
            f"with {len(self.processors)} processors"
        )

        result = PipelineResult(context=context)

        try:
            # Initialize all processors
            await self._initialize_processors(context)

            # Emit start frame
            start_frame = StartFrame(pipeline_id=context.execution_id)
            context.record_frame(start_frame.to_dict(), "pipeline")

            # Process the initial frame through all processors
            output_frames = await self._process_frame_through_pipeline(
                initial_frame, context
            )

            # Collect results
            result.output_frames = output_frames

            # Check for errors in output
            error_frames = [f for f in output_frames if isinstance(f, ErrorFrame)]
            if error_frames:
                # Pipeline completed but with errors
                result.success = False
                result.error = error_frames[0].error_message
                result.error_frame = error_frames[0]
                logger.warning(
                    f"Pipeline {context.execution_id[:8]}... completed with "
                    f"{len(error_frames)} error(s)"
                )
            else:
                logger.info(
                    f"Pipeline {context.execution_id[:8]}... completed successfully "
                    f"in {context.elapsed_ms:.1f}ms with {len(output_frames)} output frames"
                )

            # Emit end frame
            end_frame = EndFrame(
                pipeline_id=context.execution_id,
                source_frame_id=initial_frame.id,
            )
            context.record_frame(end_frame.to_dict(), "pipeline")

        except Exception as e:
            # Unexpected pipeline error
            logger.error(
                f"Pipeline {context.execution_id[:8]}... failed: {e}",
                exc_info=True,
            )
            result.success = False
            result.error = str(e)
            result.error_frame = ErrorFrame.from_exception(
                exception=e,
                processor_name="pipeline",
                original_frame=initial_frame,
            )

        finally:
            # Always cleanup
            await self._cleanup_processors()

        return result

    def __repr__(self) -> str:
        return f"Pipeline(processors={self.processor_names})"


class PipelineBuilder:
    """
    Builder for constructing pipelines with fluent API.

    Example:
        pipeline = (
            PipelineBuilder()
            .add(AudioDownloadProcessor())
            .add(TranscriptionProcessor())
            .add(StandupExtractionProcessor())
            .add(ZulipProcessor())
            .add(ConfirmationProcessor())
            .build()
        )
    """

    def __init__(self):
        self._processors: List["Processor"] = []

    def add(self, processor: "Processor") -> "PipelineBuilder":
        """Add a processor to the pipeline."""
        self._processors.append(processor)
        return self

    def add_if(
        self,
        condition: bool,
        processor: "Processor",
    ) -> "PipelineBuilder":
        """Conditionally add a processor."""
        if condition:
            self._processors.append(processor)
        return self

    def build(self) -> Pipeline:
        """Build and return the pipeline."""
        if not self._processors:
            raise ValueError("Pipeline must have at least one processor")
        return Pipeline(self._processors)
