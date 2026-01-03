"""
Transport Registry for Knomly.

Global registry for managing transport adapters.
Provides thread-safe singleton pattern for adapter lookup.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import TransportAdapter

logger = logging.getLogger(__name__)


class TransportNotFoundError(Exception):
    """
    Raised when a transport adapter is not found in the registry.

    This typically indicates a configuration error - the channel_id
    in the frame/context doesn't match any registered transport.
    """

    pass


class TransportRegistry:
    """
    Registry for transport adapters.

    Thread-safe singleton pattern for managing transport adapters.
    Transports are registered at application startup and looked up
    during pipeline execution.

    Example:
        # At app startup
        registry = get_transport_registry()
        registry.register(TwilioTransport(...))
        registry.register(TelegramTransport(...))

        # In processors
        transport = registry.get("twilio")
        await transport.send_message("+1234567890", "Hello!")
    """

    def __init__(self) -> None:
        self._adapters: dict[str, "TransportAdapter"] = {}

    def register(self, adapter: "TransportAdapter") -> None:
        """
        Register a transport adapter.

        Args:
            adapter: Transport adapter to register

        Note:
            If an adapter with the same channel_id is already registered,
            it will be replaced (useful for testing).
        """
        channel_id = adapter.channel_id
        if channel_id in self._adapters:
            logger.warning(f"Replacing existing transport adapter: {channel_id}")
        self._adapters[channel_id] = adapter
        logger.info(f"Registered transport adapter: {channel_id}")

    def get(self, channel_id: str) -> "TransportAdapter":
        """
        Get a transport adapter by channel ID.

        Args:
            channel_id: The channel identifier

        Returns:
            The registered transport adapter

        Raises:
            TransportNotFoundError: If no adapter is registered for the channel
        """
        adapter = self._adapters.get(channel_id)
        if adapter is None:
            available = ", ".join(self._adapters.keys()) or "(none)"
            raise TransportNotFoundError(
                f"No transport adapter registered for channel: {channel_id}. "
                f"Available: {available}"
            )
        return adapter

    def has(self, channel_id: str) -> bool:
        """Check if a transport adapter is registered."""
        return channel_id in self._adapters

    @property
    def registered_channels(self) -> list[str]:
        """Get list of registered channel IDs."""
        return list(self._adapters.keys())

    def unregister(self, channel_id: str) -> bool:
        """
        Unregister a transport adapter.

        Args:
            channel_id: The channel identifier to unregister

        Returns:
            True if adapter was removed, False if not found
        """
        if channel_id in self._adapters:
            del self._adapters[channel_id]
            logger.info(f"Unregistered transport adapter: {channel_id}")
            return True
        return False

    def clear(self) -> None:
        """Clear all registered adapters (for testing)."""
        self._adapters.clear()
        logger.debug("Cleared all transport adapters")


# Global registry instance
_registry: TransportRegistry | None = None


def get_transport_registry() -> TransportRegistry:
    """
    Get the global transport registry.

    Creates the registry on first access (lazy initialization).
    """
    global _registry
    if _registry is None:
        _registry = TransportRegistry()
    return _registry


def get_transport(channel_id: str) -> "TransportAdapter":
    """
    Get a transport adapter from the global registry.

    Convenience function for common case.

    Args:
        channel_id: The channel identifier

    Returns:
        The registered transport adapter

    Raises:
        TransportNotFoundError: If no adapter is registered
    """
    return get_transport_registry().get(channel_id)


def register_transport(adapter: "TransportAdapter") -> None:
    """
    Register a transport adapter in the global registry.

    Convenience function for common case.

    Args:
        adapter: Transport adapter to register
    """
    get_transport_registry().register(adapter)


def reset_transport_registry() -> None:
    """
    Reset the global transport registry (for testing).

    Creates a fresh registry instance.
    """
    global _registry
    if _registry is not None:
        _registry.clear()
    _registry = None
