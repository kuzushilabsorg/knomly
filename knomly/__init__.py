"""
Knomly - A Pipecat-inspired pipeline framework for AI-powered voice and messaging applications.

Knomly provides a modular, extensible framework for building voice-first AI applications
with features like:

- **Pipeline Architecture**: Compose processors into data processing pipelines
- **Frame-Based Data Flow**: Type-safe data containers flowing through processors
- **Provider Abstraction**: Swappable STT, LLM, and Chat providers
- **Transport Layer**: Platform-agnostic messaging (Twilio, Telegram, etc.)
- **Agent Layer**: ReAct-style agent execution with tool calling
- **Multi-Tenancy**: Per-user configuration and credential management

Quick Start:
    >>> from knomly.pipeline import Pipeline, PipelineBuilder
    >>> from knomly.pipeline.frames import AudioInputFrame
    >>> from knomly.pipeline.processors import TranscriptionProcessor
    >>>
    >>> pipeline = (
    ...     PipelineBuilder()
    ...     .add(TranscriptionProcessor())
    ...     .build()
    ... )
    >>> result = await pipeline.execute(frame)

For more information, see:
- Documentation: https://github.com/kuzushi-labs/knomly
- Examples: https://github.com/kuzushi-labs/knomly/tree/main/examples
"""

__version__ = "0.1.0"
__author__ = "Kuzushi Labs"
__license__ = "MIT"

# Core exports for convenient imports
from knomly.pipeline import Pipeline, PipelineBuilder, PipelineContext, PipelineResult
from knomly.pipeline.processor import Processor
from knomly.pipeline.frames import Frame

__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__license__",
    # Core pipeline
    "Pipeline",
    "PipelineBuilder",
    "PipelineContext",
    "PipelineResult",
    "Processor",
    "Frame",
]
