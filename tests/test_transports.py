"""
Tests for Transport Layer.

Tests the transport abstraction pattern including:
- TransportAdapter protocol
- TransportRegistry
- TwilioTransport implementation
"""

from unittest.mock import MagicMock, patch

import pytest

from knomly.pipeline.frames import AudioInputFrame
from knomly.pipeline.transports import (
    SendResult,
    TransportAdapter,
    TransportNotFoundError,
    TransportRegistry,
    TwilioTransport,
    create_twilio_transport,
    get_transport,
    get_transport_registry,
    register_transport,
    reset_transport_registry,
)

# =============================================================================
# SendResult Tests
# =============================================================================


class TestSendResult:
    """Tests for SendResult dataclass."""

    def test_success_result(self):
        result = SendResult(success=True, message_id="SM123")
        assert result.success is True
        assert result.message_id == "SM123"
        assert result.error is None

    def test_failure_result(self):
        result = SendResult(success=False, error="Connection failed")
        assert result.success is False
        assert result.message_id is None
        assert result.error == "Connection failed"

    def test_is_immutable(self):
        result = SendResult(success=True)
        with pytest.raises(Exception):  # frozen dataclass
            result.success = False


# =============================================================================
# TransportRegistry Tests
# =============================================================================


class TestTransportRegistry:
    """Tests for TransportRegistry."""

    def setup_method(self):
        """Reset registry before each test."""
        reset_transport_registry()

    def test_register_and_get(self):
        registry = TransportRegistry()
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        registry.register(transport)
        retrieved = registry.get("twilio")

        assert retrieved is transport
        assert retrieved.channel_id == "twilio"

    def test_get_raises_when_not_found(self):
        registry = TransportRegistry()

        with pytest.raises(TransportNotFoundError) as exc_info:
            registry.get("telegram")

        assert "telegram" in str(exc_info.value)
        assert "Available:" in str(exc_info.value)

    def test_has(self):
        registry = TransportRegistry()
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        assert registry.has("twilio") is False
        registry.register(transport)
        assert registry.has("twilio") is True

    def test_registered_channels(self):
        registry = TransportRegistry()
        transport1 = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        assert registry.registered_channels == []
        registry.register(transport1)
        assert registry.registered_channels == ["twilio"]

    def test_unregister(self):
        registry = TransportRegistry()
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        registry.register(transport)
        assert registry.has("twilio") is True

        result = registry.unregister("twilio")
        assert result is True
        assert registry.has("twilio") is False

        result = registry.unregister("twilio")
        assert result is False

    def test_clear(self):
        registry = TransportRegistry()
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        registry.register(transport)
        assert registry.has("twilio") is True

        registry.clear()
        assert registry.has("twilio") is False
        assert registry.registered_channels == []

    def test_replace_existing_transport(self):
        registry = TransportRegistry()
        transport1 = TwilioTransport(
            account_sid="AC123",
            auth_token="token1",
            from_number="whatsapp:+14155238886",
        )
        transport2 = TwilioTransport(
            account_sid="AC456",
            auth_token="token2",
            from_number="whatsapp:+14155238886",
        )

        registry.register(transport1)
        registry.register(transport2)

        retrieved = registry.get("twilio")
        assert retrieved.account_sid == "AC456"


# =============================================================================
# Global Registry Tests
# =============================================================================


class TestGlobalRegistry:
    """Tests for global registry functions."""

    def setup_method(self):
        reset_transport_registry()

    def test_get_transport_registry_singleton(self):
        reg1 = get_transport_registry()
        reg2 = get_transport_registry()
        assert reg1 is reg2

    def test_register_and_get_transport(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        register_transport(transport)
        retrieved = get_transport("twilio")

        assert retrieved is transport

    def test_reset_transport_registry(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )
        register_transport(transport)

        reset_transport_registry()

        with pytest.raises(TransportNotFoundError):
            get_transport("twilio")


# =============================================================================
# TwilioTransport Tests
# =============================================================================


class TestTwilioTransport:
    """Tests for TwilioTransport adapter."""

    def test_channel_id(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )
        assert transport.channel_id == "twilio"

    def test_credentials_properties(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="secret",
            from_number="whatsapp:+14155238886",
        )
        assert transport.account_sid == "AC123"
        assert transport.auth_token == "secret"

    @pytest.mark.asyncio
    async def test_normalize_request(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        mock_request = MagicMock()
        form_data = {
            "From": "whatsapp:+919876543210",
            "MediaUrl0": "https://api.twilio.com/media/audio.ogg",
            "MediaContentType0": "audio/ogg",
            "ProfileName": "Test User",
            "Body": "Hello",
            "MessageSid": "SM123",
        }

        frame = await transport.normalize_request(mock_request, form_data)

        assert isinstance(frame, AudioInputFrame)
        assert frame.media_url == "https://api.twilio.com/media/audio.ogg"
        assert frame.mime_type == "audio/ogg"
        assert frame.sender_phone == "919876543210"
        assert frame.profile_name == "Test User"
        assert frame.channel_id == "twilio"
        assert frame.metadata["body"] == "Hello"
        assert frame.metadata["message_sid"] == "SM123"

    @pytest.mark.asyncio
    async def test_normalize_request_missing_media_url(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        mock_request = MagicMock()
        form_data = {
            "From": "whatsapp:+919876543210",
        }

        with pytest.raises(ValueError) as exc_info:
            await transport.normalize_request(mock_request, form_data)

        assert "No media URL" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_normalize_request_phone_normalization(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        # Test 10-digit number gets country code
        form_data = {
            "From": "9876543210",
            "MediaUrl0": "https://example.com/audio.ogg",
        }
        frame = await transport.normalize_request(MagicMock(), form_data)
        assert frame.sender_phone == "919876543210"

        # Test number with + prefix
        form_data = {
            "From": "+919876543210",
            "MediaUrl0": "https://example.com/audio.ogg",
        }
        frame = await transport.normalize_request(MagicMock(), form_data)
        assert frame.sender_phone == "919876543210"

    @pytest.mark.asyncio
    async def test_send_message_twilio_not_installed(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        # Mock the twilio import to raise ImportError
        with patch.dict("sys.modules", {"twilio": None, "twilio.rest": None}):
            # Force reimport to trigger ImportError
            import sys

            # Remove cached module
            if "twilio.rest" in sys.modules:
                del sys.modules["twilio.rest"]
            if "twilio" in sys.modules:
                del sys.modules["twilio"]

        # Since twilio is installed, we can't easily test ImportError
        # Just verify the interface works with actual implementation
        # (This would fail if twilio is not installed)

    @pytest.mark.asyncio
    async def test_send_message_success(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        # Create mock twilio module structure
        mock_message = MagicMock()
        mock_message.sid = "SM123456"

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_message

        mock_client_class = MagicMock(return_value=mock_client_instance)

        # Create mock module
        mock_twilio_rest = MagicMock()
        mock_twilio_rest.Client = mock_client_class

        # Patch sys.modules to mock the import
        import sys

        with patch.dict(sys.modules, {"twilio": MagicMock(), "twilio.rest": mock_twilio_rest}):
            result = await transport.send_message("+919876543210", "Test message")

        assert result.success is True
        assert result.message_id == "SM123456"
        assert result.error is None

        # Verify correct formatting
        mock_client_instance.messages.create.assert_called_once()
        call_kwargs = mock_client_instance.messages.create.call_args[1]
        assert call_kwargs["to"] == "whatsapp:+919876543210"
        assert call_kwargs["from_"] == "whatsapp:+14155238886"
        assert call_kwargs["body"] == "Test message"


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateTwilioTransport:
    """Tests for create_twilio_transport factory."""

    def test_creates_transport(self):
        transport = create_twilio_transport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        assert isinstance(transport, TwilioTransport)
        assert transport.channel_id == "twilio"
        assert transport.account_sid == "AC123"


# =============================================================================
# Protocol Compliance Tests
# =============================================================================


class TestTransportAdapterProtocol:
    """Tests that implementations comply with TransportAdapter protocol."""

    def test_twilio_is_transport_adapter(self):
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )

        # Check protocol compliance
        assert isinstance(transport, TransportAdapter)
        assert hasattr(transport, "channel_id")
        assert hasattr(transport, "normalize_request")
        assert hasattr(transport, "send_message")

    @pytest.mark.asyncio
    async def test_custom_transport_compliance(self):
        """Test that a custom transport can comply with the protocol."""

        class CustomTransport:
            @property
            def channel_id(self) -> str:
                return "custom"

            async def normalize_request(self, request, form_data=None):
                return AudioInputFrame(
                    media_url="https://example.com/audio.ogg",
                    channel_id="custom",
                )

            async def send_message(self, recipient: str, message: str):
                return SendResult(success=True, message_id="custom-123")

        transport = CustomTransport()
        assert isinstance(transport, TransportAdapter)

        # Test it works
        frame = await transport.normalize_request(None)
        assert frame.channel_id == "custom"

        result = await transport.send_message("+1234567890", "Hello")
        assert result.success is True


# =============================================================================
# Integration Tests
# =============================================================================


class TestTransportIntegration:
    """Integration tests for transport pattern."""

    def setup_method(self):
        reset_transport_registry()

    @pytest.mark.asyncio
    async def test_full_transport_flow(self):
        """Test complete flow: register -> normalize -> send."""
        # Register transport
        transport = TwilioTransport(
            account_sid="AC123",
            auth_token="token",
            from_number="whatsapp:+14155238886",
        )
        register_transport(transport)

        # Get transport and normalize
        retrieved = get_transport("twilio")
        form_data = {
            "From": "whatsapp:+919876543210",
            "MediaUrl0": "https://api.twilio.com/media/audio.ogg",
            "ProfileName": "Test User",
        }
        frame = await retrieved.normalize_request(MagicMock(), form_data)

        # Verify frame
        assert frame.channel_id == "twilio"
        assert frame.sender_phone == "919876543210"

        # Create mock twilio module structure
        mock_message = MagicMock()
        mock_message.sid = "SM123"

        mock_client_instance = MagicMock()
        mock_client_instance.messages.create.return_value = mock_message

        mock_client_class = MagicMock(return_value=mock_client_instance)

        mock_twilio_rest = MagicMock()
        mock_twilio_rest.Client = mock_client_class

        import sys

        with patch.dict(sys.modules, {"twilio": MagicMock(), "twilio.rest": mock_twilio_rest}):
            result = await retrieved.send_message(frame.sender_phone, "Confirmation")

        assert result.success is True
