"""
Twilio Transport Adapter for Knomly.

Handles WhatsApp messaging via Twilio's API.
Implements the TransportAdapter protocol for bidirectional communication.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from .protocol import SendResult

if TYPE_CHECKING:
    from starlette.requests import Request

    from ..frames import AudioInputFrame

logger = logging.getLogger(__name__)


class TwilioTransport:
    """
    Twilio transport adapter for WhatsApp messaging.

    Handles:
    - Ingress: Parsing Twilio webhook form data into AudioInputFrame
    - Egress: Sending WhatsApp messages via Twilio API

    Example:
        transport = TwilioTransport(
            account_sid="AC...",
            auth_token="...",
            from_number="whatsapp:+14155238886",
        )

        # Register for use in pipeline
        register_transport(transport)

        # Normalize incoming webhook
        frame = await transport.normalize_request(request, form_data)

        # Send outgoing message
        result = await transport.send_message("+1234567890", "Hello!")
    """

    def __init__(
        self,
        account_sid: str,
        auth_token: str,
        from_number: str,
    ):
        """
        Initialize Twilio transport.

        Args:
            account_sid: Twilio account SID
            auth_token: Twilio auth token
            from_number: WhatsApp number (e.g., "whatsapp:+14155238886")
        """
        self._account_sid = account_sid
        self._auth_token = auth_token
        self._from_number = from_number

    @property
    def channel_id(self) -> str:
        """Unique identifier for Twilio/WhatsApp channel."""
        return "twilio"

    @property
    def account_sid(self) -> str:
        """Get account SID (for authenticated media downloads)."""
        return self._account_sid

    @property
    def auth_token(self) -> str:
        """Get auth token (for authenticated media downloads)."""
        return self._auth_token

    async def normalize_request(
        self,
        request: "Request",
        form_data: dict[str, Any] | None = None,
    ) -> "AudioInputFrame":
        """
        Convert Twilio webhook to AudioInputFrame.

        Twilio sends form-encoded data with:
        - From: Sender phone (e.g., "whatsapp:+1234567890")
        - MediaUrl0: URL to the audio file
        - MediaContentType0: MIME type
        - Body: Optional text content
        - ProfileName: WhatsApp profile name

        Args:
            request: The incoming HTTP request
            form_data: Pre-parsed form data (optional)

        Returns:
            AudioInputFrame with normalized data

        Raises:
            ValueError: If request is missing required fields
        """
        from ..frames import AudioInputFrame

        if form_data is None:
            form_data = dict(await request.form())

        # Extract sender phone (strip "whatsapp:" prefix if present)
        sender_raw = str(form_data.get("From", ""))
        sender_phone = self._normalize_phone(sender_raw)

        # Extract profile name
        profile_name = str(form_data.get("ProfileName", "")).strip()

        # Extract media URL
        media_url = str(form_data.get("MediaUrl0", "")).strip()
        if not media_url:
            raise ValueError("No media URL in Twilio webhook")

        # Extract MIME type
        mime_type = str(form_data.get("MediaContentType0", "audio/ogg")).strip()

        return AudioInputFrame(
            media_url=media_url,
            mime_type=mime_type,
            sender_phone=sender_phone,
            profile_name=profile_name,
            channel_id=self.channel_id,
            metadata={
                "body": str(form_data.get("Body", "")),
                "message_sid": str(form_data.get("MessageSid", "")),
            },
        )

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> SendResult:
        """
        Send WhatsApp message via Twilio.

        Args:
            recipient: Phone number (with or without whatsapp: prefix)
            message: Message content to send

        Returns:
            SendResult with success status and message SID
        """
        try:
            from twilio.rest import Client

            # Format phone numbers for WhatsApp
            to_number = self._format_whatsapp_number(recipient)
            from_number = self._format_whatsapp_number(self._from_number)

            # Twilio client is sync, run in executor
            client = Client(self._account_sid, self._auth_token)

            def send():
                return client.messages.create(
                    body=message,
                    from_=from_number,
                    to=to_number,
                )

            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(None, send)

            logger.info(f"Twilio message sent: sid={result.sid}")
            return SendResult(success=True, message_id=result.sid)

        except ImportError:
            return SendResult(
                success=False,
                error="twilio package not installed",
            )
        except Exception as e:
            logger.error(f"Twilio send error: {e}", exc_info=True)
            return SendResult(success=False, error=str(e))

    def _normalize_phone(self, phone: str) -> str:
        """
        Normalize phone number to digits only.

        Strips "whatsapp:" prefix and non-digit characters.
        """
        # Strip whatsapp: prefix
        phone = phone.replace("whatsapp:", "")
        # Extract digits only
        digits = "".join(c for c in phone if c.isdigit())
        # Ensure country code (default to India if 10 digits)
        if len(digits) == 10:
            digits = "91" + digits
        return digits

    def _format_whatsapp_number(self, phone: str) -> str:
        """
        Format phone number for WhatsApp API.

        Ensures proper whatsapp:+{number} format.
        """
        if phone.startswith("whatsapp:"):
            return phone
        # Ensure + prefix for international format
        clean = phone.lstrip("+")
        return f"whatsapp:+{clean}"


def create_twilio_transport(
    account_sid: str,
    auth_token: str,
    from_number: str,
) -> TwilioTransport:
    """
    Factory function for creating TwilioTransport.

    Convenience function for configuration-driven setup.

    Args:
        account_sid: Twilio account SID
        auth_token: Twilio auth token
        from_number: WhatsApp number

    Returns:
        Configured TwilioTransport instance
    """
    return TwilioTransport(
        account_sid=account_sid,
        auth_token=auth_token,
        from_number=from_number,
    )
