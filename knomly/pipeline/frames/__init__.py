"""
Knomly Pipeline Frames

Frames are immutable data containers that flow through the pipeline.
Each frame type represents a stage of data transformation.

See ADR-001 for design decisions.
"""

from .base import EndFrame, ErrorFrame, ErrorType, Frame, StartFrame
from .input import AudioInputFrame, TextInputFrame
from .processing import ExtractionFrame, TranscriptionFrame
from .action import ConfirmationFrame, UserResponseFrame, ZulipMessageFrame
from .task import (
    TaskActionFrame,
    TaskData,
    TaskFrame,
    TaskOperation,
    TaskPriority,
    TaskQueryFrame,
    TaskResultFrame,
    TaskStatus,
)

__all__ = [
    # Base
    "Frame",
    "ErrorFrame",
    "ErrorType",
    # Control
    "StartFrame",
    "EndFrame",
    # Input
    "AudioInputFrame",
    "TextInputFrame",
    # Processing
    "TranscriptionFrame",
    "ExtractionFrame",
    # Action
    "UserResponseFrame",
    "ZulipMessageFrame",
    "ConfirmationFrame",
    # Task Management (generic)
    "TaskFrame",
    "TaskQueryFrame",
    "TaskResultFrame",
    "TaskActionFrame",
    "TaskData",
    "TaskPriority",
    "TaskStatus",
    "TaskOperation",
]
