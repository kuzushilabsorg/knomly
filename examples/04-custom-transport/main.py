"""
Custom Transport Example

This example demonstrates how to create a custom transport adapter:
1. Implement TransportAdapter protocol
2. Register with transport registry
3. Use in confirmation processor

Run: python -m examples.04-custom-transport.main
"""

import asyncio
from typing import Any

from knomly import (
    SendResult,
    TransportAdapter,
    get_transport,
    register_transport,
    reset_transport_registry,
)
from knomly.pipeline.frames import AudioInputFrame

# =============================================================================
# Custom Transport: Console Transport
# =============================================================================


class ConsoleTransport:
    """
    A simple transport that prints messages to the console.

    Useful for testing and development without external services.
    """

    def __init__(self, prefix: str = "[CONSOLE]"):
        self._prefix = prefix
        self._messages: list[dict[str, Any]] = []

    @property
    def channel_id(self) -> str:
        return "console"

    async def normalize_request(
        self,
        request: Any,
        form_data: dict[str, Any] | None = None,
    ) -> AudioInputFrame:
        """
        Normalize a console 'request' to AudioInputFrame.

        In a real transport, this would parse webhook data.
        For console, we just create a frame from the form_data.
        """
        if form_data is None:
            form_data = {}

        return AudioInputFrame(
            media_url=form_data.get("media_url", ""),
            sender_phone=form_data.get("sender", "console-user"),
            profile_name=form_data.get("name", "Console User"),
            channel_id=self.channel_id,
        )

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> SendResult:
        """
        'Send' a message by printing to console.
        """
        print(f"\n{self._prefix} Message to {recipient}:")
        print(f"{self._prefix} {message}")
        print()

        # Store for later inspection
        self._messages.append(
            {
                "recipient": recipient,
                "message": message,
            }
        )

        return SendResult(
            success=True,
            message_id=f"console-{len(self._messages)}",
        )

    @property
    def sent_messages(self) -> list[dict[str, Any]]:
        """Get all sent messages (for testing)."""
        return self._messages.copy()


# =============================================================================
# Custom Transport: Webhook Transport
# =============================================================================


class WebhookTransport:
    """
    A transport that sends messages via HTTP webhook.

    Useful for integrating with custom services.
    """

    def __init__(self, webhook_url: str, auth_token: str | None = None):
        self._webhook_url = webhook_url
        self._auth_token = auth_token

    @property
    def channel_id(self) -> str:
        return "webhook"

    async def normalize_request(
        self,
        request: Any,
        form_data: dict[str, Any] | None = None,
    ) -> AudioInputFrame:
        """Parse incoming webhook data."""
        if form_data is None:
            form_data = {}

        # Assume webhook sends JSON with these fields
        return AudioInputFrame(
            media_url=form_data.get("audio_url", ""),
            sender_phone=form_data.get("user_id", ""),
            profile_name=form_data.get("user_name", ""),
            channel_id=self.channel_id,
        )

    async def send_message(
        self,
        recipient: str,
        message: str,
    ) -> SendResult:
        """Send message via webhook."""
        import httpx

        headers = {}
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self._webhook_url,
                    json={
                        "recipient": recipient,
                        "message": message,
                    },
                    headers=headers,
                    timeout=10.0,
                )
                response.raise_for_status()

                data = response.json()
                return SendResult(
                    success=True,
                    message_id=data.get("id", "unknown"),
                )

        except Exception as e:
            return SendResult(
                success=False,
                error=str(e),
            )


# =============================================================================
# Verify Protocol Compliance
# =============================================================================


def verify_transport(transport: Any) -> bool:
    """Verify that a transport implements the protocol correctly."""
    # Check it's a TransportAdapter
    if not isinstance(transport, TransportAdapter):
        print(f"ERROR: {type(transport)} is not a TransportAdapter")
        return False

    # Check required properties/methods
    required = ["channel_id", "normalize_request", "send_message"]
    for attr in required:
        if not hasattr(transport, attr):
            print(f"ERROR: Missing {attr}")
            return False

    print(f"OK: {type(transport).__name__} implements TransportAdapter")
    return True


# =============================================================================
# Main
# =============================================================================


async def main():
    # Reset registry for clean state
    reset_transport_registry()

    print("Custom Transport Example")
    print("=" * 50)
    print()

    # Create and verify transports
    console = ConsoleTransport(prefix="[MSG]")
    webhook = WebhookTransport(
        webhook_url="https://example.com/webhook",
        auth_token="secret-token",
    )

    print("Verifying protocol compliance:")
    verify_transport(console)
    verify_transport(webhook)
    print()

    # Register transports
    register_transport(console)
    register_transport(webhook)
    print("Registered transports: console, webhook")
    print()

    # Use the console transport
    transport = get_transport("console")
    print(f"Retrieved transport: {transport.channel_id}")
    print()

    # Simulate sending messages
    print("Sending test messages via console transport:")

    result1 = await transport.send_message(
        recipient="+1234567890",
        message="Hello from Knomly!",
    )
    print(f"Result: success={result1.success}, id={result1.message_id}")

    result2 = await transport.send_message(
        recipient="+0987654321",
        message="Your standup has been posted to #general",
    )
    print(f"Result: success={result2.success}, id={result2.message_id}")

    # Check sent messages
    print()
    print(f"Total messages sent: {len(console.sent_messages)}")

    # Normalize a request
    print()
    print("Normalizing a mock request:")
    frame = await transport.normalize_request(
        request=None,
        form_data={
            "sender": "+1234567890",
            "name": "John Doe",
            "media_url": "https://example.com/audio.ogg",
        },
    )
    print(f"Created frame: {frame.frame_type}")
    print(f"  - sender_phone: {frame.sender_phone}")
    print(f"  - profile_name: {frame.profile_name}")
    print(f"  - channel_id: {frame.channel_id}")


if __name__ == "__main__":
    asyncio.run(main())
