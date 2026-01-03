"""
Knomly Transport Layer.

Bidirectional transport adapters for message channels.

The Transport Pattern decouples the pipeline from specific messaging
platforms, enabling easy addition of new channels (Telegram, Slack,
Signal, etc.) without modifying core logic.

Core Components:
- TransportAdapter: Protocol defining the adapter interface
- SendResult: Result of sending a message
- TransportRegistry: Global registry for adapter lookup

Built-in Transports:
- TwilioTransport: WhatsApp via Twilio

Usage:
    # At app startup
    from knomly.pipeline.transports import (
        TwilioTransport,
        register_transport,
    )

    transport = TwilioTransport(
        account_sid=settings.twilio_account_sid,
        auth_token=settings.twilio_auth_token,
        from_number=settings.twilio_whatsapp_number,
    )
    register_transport(transport)

    # In webhook handler
    from knomly.pipeline.transports import get_transport

    transport = get_transport("twilio")
    frame = await transport.normalize_request(request, form_data)

    # In processors
    transport = get_transport(ctx.channel_id)
    result = await transport.send_message(recipient, message)

Adding New Transports:
    1. Create a class implementing TransportAdapter protocol
    2. Implement channel_id, normalize_request(), send_message()
    3. Register at app startup

    class TelegramTransport:
        @property
        def channel_id(self) -> str:
            return "telegram"

        async def normalize_request(self, request, form_data=None):
            # Parse Telegram webhook format
            ...

        async def send_message(self, recipient, message):
            # Send via Telegram Bot API
            ...
"""

from .protocol import SendResult, TransportAdapter
from .registry import (
    TransportNotFoundError,
    TransportRegistry,
    get_transport,
    get_transport_registry,
    register_transport,
    reset_transport_registry,
)
from .twilio import TwilioTransport, create_twilio_transport

__all__ = [
    "SendResult",
    # Protocol
    "TransportAdapter",
    "TransportNotFoundError",
    # Registry
    "TransportRegistry",
    # Implementations
    "TwilioTransport",
    "create_twilio_transport",
    "get_transport",
    "get_transport_registry",
    "register_transport",
    "reset_transport_registry",
]
