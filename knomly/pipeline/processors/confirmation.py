"""
Confirmation Processor for Knomly.

Sends confirmation messages back to users via the appropriate transport.
Transport-agnostic - uses the transport registry to determine how to send.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..processor import Processor

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import Frame

logger = logging.getLogger(__name__)


class ConfirmationProcessor(Processor):
    """
    Sends confirmation messages back to users.

    Input: ZulipMessageFrame, UserResponseFrame, or ErrorFrame
    Output: ConfirmationFrame with send result

    This processor is transport-agnostic. It uses the channel_id from
    PipelineContext to look up the appropriate transport adapter and
    send the message via that channel.

    Frame handling:
    - ZulipMessageFrame: Sends "Your standup has been posted..." or error
    - UserResponseFrame: Sends the message content directly (generic response)
    - ErrorFrame: Sends user-friendly error message
    - Other frames: Passed through unchanged

    Transport selection:
    - Uses ctx.channel_id to determine which transport to use
    - Falls back to direct Twilio if credentials provided (legacy mode)
    - Raises error if no transport configured

    Example:
        # Modern usage (transport registry)
        processor = ConfirmationProcessor()

        # Legacy usage (direct credentials - deprecated)
        processor = ConfirmationProcessor(
            twilio_account_sid="...",
            twilio_auth_token="...",
            twilio_from_number="...",
        )
    """

    def __init__(
        self,
        twilio_account_sid: str | None = None,
        twilio_auth_token: str | None = None,
        twilio_from_number: str | None = None,
    ):
        """
        Initialize confirmation processor.

        Args:
            twilio_account_sid: (Deprecated) Twilio account SID for legacy mode
            twilio_auth_token: (Deprecated) Twilio auth token for legacy mode
            twilio_from_number: (Deprecated) Twilio WhatsApp number for legacy mode

        Note:
            Direct credentials are deprecated. Use the transport registry instead:

            # At app startup
            register_transport(TwilioTransport(...))

            # In pipeline builder
            builder.add(ConfirmationProcessor())  # No credentials needed
        """
        # Legacy mode credentials (deprecated)
        self._legacy_account_sid = twilio_account_sid
        self._legacy_auth_token = twilio_auth_token
        self._legacy_from_number = twilio_from_number

    @property
    def name(self) -> str:
        return "confirmation"

    async def process(
        self,
        frame: Frame,
        ctx: PipelineContext,
    ) -> Frame | None:
        from ..frames import ConfirmationFrame, ErrorFrame, UserResponseFrame, ZulipMessageFrame

        # Determine message based on frame type
        message: str
        recipient: str

        if isinstance(frame, ZulipMessageFrame):
            # Zulip-specific: format as "Your standup has been posted..."
            message = frame.format_confirmation()
            recipient = frame.sender_phone
        elif isinstance(frame, UserResponseFrame):
            # Generic response: use message content directly
            message = frame.message
            recipient = frame.sender_phone
        elif isinstance(frame, ErrorFrame):
            # Error: format user-friendly message
            message = frame.format_user_message()
            recipient = frame.sender_phone or ""
        else:
            # Unknown frame type: pass through unchanged
            return frame

        if not recipient:
            logger.warning("No recipient phone for confirmation")
            return ConfirmationFrame(
                recipient_phone="",
                message=message,
                success=False,
                error="No recipient phone number",
                source_frame_id=frame.id,
            )

        logger.info(f"Sending confirmation to {recipient}")

        # Send via transport (modern) or legacy mode
        result = await self._send_message(ctx, recipient, message)

        if result.success:
            logger.info(f"Confirmation sent, sid={result.message_id}")
        else:
            logger.error(f"Confirmation failed: {result.error}")

        return ConfirmationFrame(
            recipient_phone=recipient,
            message=message,
            message_sid=result.message_id,
            success=result.success,
            error=result.error,
            source_frame_id=frame.id,
        )

    async def _send_message(
        self,
        ctx: PipelineContext,
        recipient: str,
        message: str,
    ) -> SendResult:
        """
        Send message via appropriate transport.

        Uses transport registry if channel_id is set, otherwise
        falls back to legacy direct Twilio mode.
        """
        from ..transports import SendResult, TransportNotFoundError, get_transport

        # Modern mode: use transport registry
        if ctx.channel_id:
            try:
                transport = get_transport(ctx.channel_id)
                return await transport.send_message(recipient, message)
            except TransportNotFoundError as e:
                logger.error(f"Transport not found: {e}")
                return SendResult(success=False, error=str(e))

        # Legacy mode: use direct credentials
        if self._has_legacy_credentials():
            return await self._send_legacy_whatsapp(recipient, message)

        # No transport configured
        logger.error("No transport configured: set channel_id or provide credentials")
        return SendResult(
            success=False,
            error="No transport configured",
        )

    def _has_legacy_credentials(self) -> bool:
        """Check if legacy credentials are configured."""
        return bool(
            self._legacy_account_sid and self._legacy_auth_token and self._legacy_from_number
        )

    async def _send_legacy_whatsapp(
        self,
        to_phone: str,
        message: str,
    ) -> SendResult:
        """
        Send WhatsApp message via Twilio (legacy mode).

        Deprecated: Use transport registry instead.
        """
        import asyncio

        from ..transports import SendResult

        try:
            from twilio.rest import Client

            # Format phone numbers for WhatsApp
            to_number = (
                to_phone
                if to_phone.startswith("whatsapp:")
                else f"whatsapp:+{to_phone.lstrip('+')}"
            )
            from_number = (
                self._legacy_from_number
                if self._legacy_from_number.startswith("whatsapp:")
                else f"whatsapp:{self._legacy_from_number}"
            )

            # Twilio client is sync, run in executor
            client = Client(self._legacy_account_sid, self._legacy_auth_token)

            def send():
                return client.messages.create(
                    body=message,
                    from_=from_number,
                    to=to_number,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, send)

            return SendResult(success=True, message_id=result.sid)

        except ImportError:
            return SendResult(success=False, error="twilio package not installed")
        except Exception as e:
            logger.error(f"Twilio send error: {e}", exc_info=True)
            return SendResult(success=False, error=str(e))


# Type hint for import
if TYPE_CHECKING:
    from ..transports import SendResult
