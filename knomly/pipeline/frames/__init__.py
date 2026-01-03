"""
Knomly Pipeline Frames

Frames are immutable data containers that flow through the pipeline.
Each frame type represents a stage of data transformation.

See ADR-001 for design decisions.
"""

from .action import ConfirmationFrame, UserResponseFrame, ZulipMessageFrame
from .base import EndFrame, ErrorFrame, ErrorType, Frame, StartFrame
from .input import AudioInputFrame, TextInputFrame
from .processing import ExtractionFrame, TranscriptionFrame
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
    # Input
    "AudioInputFrame",
    "ConfirmationFrame",
    "EndFrame",
    "ErrorFrame",
    "ErrorType",
    "ExtractionFrame",
    # Base
    "Frame",
    # Control
    "StartFrame",
    "TaskActionFrame",
    "TaskData",
    # Task Management (generic)
    "TaskFrame",
    "TaskOperation",
    "TaskPriority",
    "TaskQueryFrame",
    "TaskResultFrame",
    "TaskStatus",
    "TextInputFrame",
    # Processing
    "TranscriptionFrame",
    # Action
    "UserResponseFrame",
    "ZulipMessageFrame",
]
