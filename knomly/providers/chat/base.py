"""
Chat Provider Protocol for Knomly.

Defines the interface for chat/messaging providers (Zulip, Slack, etc.).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable


class MessageType(str, Enum):
    """Type of chat message."""

    STREAM = "stream"  # Zulip stream message
    PRIVATE = "private"  # Direct message
    CHANNEL = "channel"  # Slack channel


@dataclass
class ChatMessage:
    """
    A chat message to send.

    Attributes:
        content: Message content (markdown supported)
        stream: Target stream/channel name
        topic: Message topic (Zulip-specific)
        message_type: Type of message (stream, private)
        recipients: List of user IDs for private messages
    """

    content: str
    stream: str = ""
    topic: str = ""
    message_type: MessageType = MessageType.STREAM
    recipients: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "stream": self.stream,
            "topic": self.topic,
            "message_type": self.message_type.value,
            "recipients": self.recipients,
        }


@dataclass
class MessageResult:
    """
    Result of sending a chat message.

    Attributes:
        success: Whether the message was sent
        message_id: Provider's message ID
        stream: Stream where message was posted
        topic: Topic where message was posted
        error: Error message if failed
        timestamp: When message was sent
    """

    success: bool
    message_id: Optional[int] = None
    stream: str = ""
    topic: str = ""
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "message_id": self.message_id,
            "stream": self.stream,
            "topic": self.topic,
            "error": self.error,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class StreamInfo:
    """
    Information about a chat stream/channel.

    Attributes:
        stream_id: Unique stream identifier
        name: Stream name
        description: Stream description
        topics: List of topics in the stream
    """

    stream_id: int
    name: str
    description: str = ""
    topics: List[str] = field(default_factory=list)


@runtime_checkable
class ChatProvider(Protocol):
    """
    Protocol for Chat providers.

    Implementations must provide:
    - send_message(): Send a message to stream/channel
    - name: Provider identifier

    Supported providers:
    - Zulip (REST API)
    - Slack (future)
    """

    @property
    def name(self) -> str:
        """Provider name for logging and configuration."""
        ...

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        """
        Send a message to a stream/channel.

        Args:
            stream: Target stream/channel name
            topic: Message topic (Zulip-specific)
            content: Message content (markdown)

        Returns:
            MessageResult with send status
        """
        ...


class BaseChatProvider(ABC):
    """
    Base class for Chat provider implementations.

    Provides common functionality and enforces interface.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name."""
        pass

    @abstractmethod
    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        """Send a message to a stream."""
        pass

    async def send_private_message(
        self,
        recipients: List[str],
        content: str,
    ) -> MessageResult:
        """
        Send a private/direct message.

        Default implementation raises NotImplementedError.
        Override in providers that support DMs.
        """
        raise NotImplementedError(
            f"{self.name} does not support private messages"
        )

    async def list_streams(self) -> List[StreamInfo]:
        """
        List available streams/channels.

        Default implementation raises NotImplementedError.
        Override in providers that support stream listing.
        """
        raise NotImplementedError(
            f"{self.name} does not support stream listing"
        )

    async def get_stream_topics(self, stream: str) -> List[str]:
        """
        Get topics in a stream.

        Default implementation raises NotImplementedError.
        Override in providers that support topic listing.
        """
        raise NotImplementedError(
            f"{self.name} does not support topic listing"
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
