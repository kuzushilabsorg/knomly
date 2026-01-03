"""
Transport Adapter Protocol for Knomly.

Defines the interface for bidirectional message transport adapters.
Adapters normalize incoming webhook requests and send outgoing messages.

This abstraction enables the framework to support multiple messaging
platforms (WhatsApp/Twilio, Telegram, Slack, etc.) without changing
core pipeline logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from starlette.requests import Request

    from ..frames import AudioInputFrame


@dataclass(frozen=True, slots=True)
class SendResult:
    """
    Result of sending a message via transport.

    Attributes:
        success: Whether the message was sent successfully
        message_id: Platform-specific message identifier (if available)
        error: Error message if sending failed
    """

    success: bool
    message_id: str | None = None
    error: str | None = None


@runtime_checkable
class TransportAdapter(Protocol):
    """
    Bidirectional transport adapter for message channels.

    Implementations handle:
    - Ingress: Normalizing webhook requests to AudioInputFrame
    - Egress: Sending messages back to users

    The Transport Pattern decouples the pipeline from specific
    messaging platforms, enabling easy addition of new channels
    (Telegram, Slack, Signal, etc.) without modifying core logic.

    Example implementations:
    - TwilioTransport (WhatsApp)
    - TelegramTransport
    - SlackTransport

    Example usage:
        # Register transport at app startup
        transport = TwilioTransport(
            account_sid="AC...",
            auth_token="...",
            from_number="whatsapp:+14155238886",
        )
        register_transport(transport)

        # In webhook handler
        transport = get_transport("twilio")
        frame = await transport.normalize_request(request, form_data)

        # In confirmation processor
        transport = get_transport(ctx.channel_id)
        result = await transport.send_message(recipient, message)
    """

    @property
    def channel_id(self) -> str:
        """
        Unique identifier for this transport channel.

        This is used to look up the transport in the registry
        and to tag frames/context for routing.

        Examples: "twilio", "telegram", "slack"
        """
        ...

    async def normalize_request(
        self,
        request: Request,
        form_data: dict[str, Any] | None = None,
    ) -> AudioInputFrame:
        """
        Convert incoming webhook request to AudioInputFrame.

        This method normalizes platform-specific webhook payloads
        into a standard AudioInputFrame that can flow through
        the pipeline.

        Args:
            request: The incoming HTTP request
            form_data: Pre-parsed form data (optional, for performance)

        Returns:
            AudioInputFrame with normalized data and channel_id set

        Raises:
            ValueError: If request is missing required fields
        """
        ...

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> SendResult:
        """
        Send a message to a recipient.

        This method handles the platform-specific details of
        message delivery.

        Args:
            recipient: Recipient identifier (phone number, user ID, etc.)
            message: Message content to send

        Returns:
            SendResult with success status and message ID
        """
        ...
