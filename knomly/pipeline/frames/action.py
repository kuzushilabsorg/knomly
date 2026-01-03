"""
Action frames for Knomly pipeline.

These frames represent completed actions or results from external systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import Frame

# =============================================================================
# User Response Frame (Generic)
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class UserResponseFrame(Frame):
    """
    Generic frame for sending a response message back to the user.

    This is transport-agnostic - the downstream processor (e.g., ConfirmationProcessor)
    determines how to deliver the message based on the original channel.

    Use this instead of ZulipMessageFrame when:
    - You need to respond to the user but haven't interacted with Zulip/Slack
    - The response is about the pipeline itself (errors, unknown intent, etc.)
    - You want to decouple the response from any specific integration

    Attributes:
        message: The message content to send to the user
        sender_phone: Phone number of the original sender (for routing)
        success: Whether the operation that triggered this response succeeded
        error: Error description if success=False
    """

    message: str = ""
    sender_phone: str = ""
    success: bool = True
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update(
            {
                "message": self.message[:100] + "..." if len(self.message) > 100 else self.message,
                "sender_phone": self.sender_phone,
                "success": self.success,
                "error": self.error,
            }
        )
        return base


# =============================================================================
# Zulip Message Frame
# =============================================================================


@dataclass(frozen=True, kw_only=True, slots=True)
class ZulipMessageFrame(Frame):
    """
    Frame representing a Zulip message post result.

    Attributes:
        stream: Zulip stream where message was posted
        topic: Zulip topic where message was posted
        content: Content of the posted message
        message_id: Zulip message ID (after successful post)
        success: Whether the post succeeded
        error: Error message if post failed
        sender_phone: Phone number of the original sender
    """

    stream: str = ""
    topic: str = ""
    content: str = ""
    message_id: int | None = None
    success: bool = False
    error: str | None = None
    sender_phone: str = ""

    def format_confirmation(self) -> str:
        """Format a confirmation message for the user."""
        if self.success:
            return f"Your standup has been posted to #{self.stream} > {self.topic}"
        return f"Failed to post standup: {self.error or 'Unknown error'}"

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update(
            {
                "stream": self.stream,
                "topic": self.topic,
                "content_length": len(self.content),
                "message_id": self.message_id,
                "success": self.success,
                "error": self.error,
                "sender_phone": self.sender_phone,
            }
        )
        return base


@dataclass(frozen=True, kw_only=True, slots=True)
class ConfirmationFrame(Frame):
    """
    Frame representing a WhatsApp confirmation message result.

    Attributes:
        recipient_phone: Phone number confirmation was sent to
        message: Confirmation message content
        message_sid: Twilio message SID (after successful send)
        success: Whether the send succeeded
        error: Error message if send failed
    """

    recipient_phone: str = ""
    message: str = ""
    message_sid: str | None = None
    success: bool = False
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        base = Frame.to_dict(self)
        base.update(
            {
                "recipient_phone": self.recipient_phone,
                "message": self.message[:100] + "..." if len(self.message) > 100 else self.message,
                "message_sid": self.message_sid,
                "success": self.success,
                "error": self.error,
            }
        )
        return base
