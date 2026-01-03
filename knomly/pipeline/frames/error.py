"""
Error frames for Knomly pipeline.

These frames represent errors that occurred during pipeline processing.
They allow graceful error handling and recovery.
"""

from __future__ import annotations

import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from .base import Frame


class ErrorType(str, Enum):
    """Classification of error types for handling decisions."""

    # Recoverable errors - can retry or skip
    NETWORK_ERROR = "network_error"
    TIMEOUT_ERROR = "timeout_error"
    RATE_LIMIT_ERROR = "rate_limit_error"
    TEMPORARY_ERROR = "temporary_error"

    # Non-recoverable errors - must fail gracefully
    VALIDATION_ERROR = "validation_error"
    AUTHENTICATION_ERROR = "authentication_error"
    CONFIGURATION_ERROR = "configuration_error"
    PROVIDER_ERROR = "provider_error"

    # System errors
    INTERNAL_ERROR = "internal_error"
    UNKNOWN_ERROR = "unknown_error"


@dataclass(frozen=True)
class ErrorFrame(Frame):
    """
    Frame representing an error that occurred during processing.

    ErrorFrames can flow through the pipeline like regular frames,
    allowing error-aware processors to handle them appropriately.

    Attributes:
        error_type: Classification of the error
        error_message: Human-readable error description
        error_code: Optional error code from provider/system
        recoverable: Whether the error is potentially recoverable
        processor_name: Name of the processor that generated the error
        original_frame_type: Type of frame that caused the error
        exception_type: Python exception class name
        stack_trace: Stack trace for debugging (optional)
        sender_phone: Phone number for user notification
    """

    error_type: ErrorType = ErrorType.UNKNOWN_ERROR
    error_message: str = "An unknown error occurred"
    error_code: str | None = None
    recoverable: bool = False
    processor_name: str = ""
    original_frame_type: str = ""
    exception_type: str | None = None
    stack_trace: str | None = None
    sender_phone: str | None = None
    retry_count: int = 0
    max_retries: int = 3

    _id: uuid.UUID = field(default_factory=uuid.uuid4, repr=False)
    _created_at: datetime = field(default_factory=datetime.utcnow, repr=False)
    _metadata: dict[str, Any] = field(default_factory=dict, repr=False)
    _source_frame_id: uuid.UUID | None = field(default=None, repr=False)

    @property
    def id(self) -> uuid.UUID:
        return self._id

    @property
    def created_at(self) -> datetime:
        return self._created_at

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata.copy()

    @property
    def source_frame_id(self) -> uuid.UUID | None:
        return self._source_frame_id

    @property
    def frame_type(self) -> str:
        return "ErrorFrame"

    @property
    def can_retry(self) -> bool:
        """Check if error can be retried."""
        return self.recoverable and self.retry_count < self.max_retries

    @property
    def is_user_notifiable(self) -> bool:
        """Check if user should be notified about this error."""
        # Always notify for non-recoverable errors or when retries exhausted
        return not self.recoverable or self.retry_count >= self.max_retries

    def format_user_message(self) -> str:
        """Format a user-friendly error message."""
        if self.error_type == ErrorType.NETWORK_ERROR:
            return "There was a network issue. Please try again in a moment."
        elif self.error_type == ErrorType.RATE_LIMIT_ERROR:
            return "Too many requests. Please wait a few minutes and try again."
        elif self.error_type == ErrorType.VALIDATION_ERROR:
            return f"Could not process your message: {self.error_message}"
        elif self.error_type == ErrorType.AUTHENTICATION_ERROR:
            return "There's an issue with service authentication. We're looking into it."
        elif self.error_type == ErrorType.CONFIGURATION_ERROR:
            return "There's a configuration issue. Please contact support."
        else:
            return "Sorry, something went wrong. Please try again later."

    def with_retry(self) -> ErrorFrame:
        """Create a new error frame with incremented retry count."""
        return ErrorFrame(
            error_type=self.error_type,
            error_message=self.error_message,
            error_code=self.error_code,
            recoverable=self.recoverable,
            processor_name=self.processor_name,
            original_frame_type=self.original_frame_type,
            exception_type=self.exception_type,
            stack_trace=self.stack_trace,
            sender_phone=self.sender_phone,
            retry_count=self.retry_count + 1,
            max_retries=self.max_retries,
            _metadata=self._metadata,
            _source_frame_id=self._source_frame_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": str(self._id),
            "frame_type": self.frame_type,
            "created_at": self._created_at.isoformat(),
            "source_frame_id": str(self._source_frame_id) if self._source_frame_id else None,
            "metadata": self._metadata,
            "error_type": self.error_type.value,
            "error_message": self.error_message,
            "error_code": self.error_code,
            "recoverable": self.recoverable,
            "processor_name": self.processor_name,
            "original_frame_type": self.original_frame_type,
            "exception_type": self.exception_type,
            "sender_phone": self.sender_phone,
            "retry_count": self.retry_count,
            "can_retry": self.can_retry,
        }

    @classmethod
    def from_exception(
        cls,
        exception: Exception,
        processor_name: str,
        original_frame: Frame | None = None,
        error_type: ErrorType | None = None,
        sender_phone: str | None = None,
        include_stack_trace: bool = True,
    ) -> ErrorFrame:
        """
        Factory method to create ErrorFrame from an exception.

        Args:
            exception: The caught exception
            processor_name: Name of the processor that caught the error
            original_frame: The frame being processed when error occurred
            error_type: Override error type classification
            sender_phone: Phone number for notification
            include_stack_trace: Whether to include full stack trace
        """
        # Auto-classify error type if not provided
        if error_type is None:
            exc_name = type(exception).__name__.lower()
            if "timeout" in exc_name or "timedout" in exc_name:
                error_type = ErrorType.TIMEOUT_ERROR
            elif "connection" in exc_name or "network" in exc_name:
                error_type = ErrorType.NETWORK_ERROR
            elif "ratelimit" in exc_name or "429" in str(exception):
                error_type = ErrorType.RATE_LIMIT_ERROR
            elif "auth" in exc_name or "401" in str(exception) or "403" in str(exception):
                error_type = ErrorType.AUTHENTICATION_ERROR
            elif "validation" in exc_name or "invalid" in exc_name:
                error_type = ErrorType.VALIDATION_ERROR
            else:
                error_type = ErrorType.INTERNAL_ERROR

        # Determine recoverability
        recoverable = error_type in {
            ErrorType.NETWORK_ERROR,
            ErrorType.TIMEOUT_ERROR,
            ErrorType.RATE_LIMIT_ERROR,
            ErrorType.TEMPORARY_ERROR,
        }

        return cls(
            error_type=error_type,
            error_message=str(exception),
            recoverable=recoverable,
            processor_name=processor_name,
            original_frame_type=original_frame.frame_type if original_frame else "",
            exception_type=type(exception).__name__,
            stack_trace=traceback.format_exc() if include_stack_trace else None,
            sender_phone=sender_phone
            or (getattr(original_frame, "sender_phone", None) if original_frame else None),
            _source_frame_id=original_frame.id if original_frame else None,
        )
