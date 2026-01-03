"""
Zulip Chat Provider for Knomly.

Uses Zulip's REST API for posting messages to streams.
"""

from __future__ import annotations

import base64
import logging

import httpx

from .base import BaseChatProvider, MessageResult, StreamInfo

logger = logging.getLogger(__name__)


class ZulipChatProvider(BaseChatProvider):
    """
    Zulip-based Chat provider.

    Uses Zulip's REST API for:
    - Posting messages to streams
    - Sending private messages
    - Listing streams and topics

    Authentication:
    - Bot email + API key (Basic Auth)
    - Generate API key at: https://your-zulip-server/#settings/account-and-privacy

    API Reference:
    - https://zulip.com/api/send-message
    """

    def __init__(
        self,
        site: str,
        bot_email: str,
        api_key: str,
    ):
        """
        Initialize Zulip Chat provider.

        Args:
            site: Zulip server URL (e.g., "https://chat.example.com")
            bot_email: Bot email address
            api_key: Bot API key
        """
        self._site = site.rstrip("/")
        self._bot_email = bot_email
        self._api_key = api_key
        self._client: httpx.AsyncClient | None = None

    @property
    def name(self) -> str:
        return "zulip"

    def _get_auth_header(self) -> str:
        """Generate Basic auth header."""
        credentials = f"{self._bot_email}:{self._api_key}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._site,
                headers={
                    "Authorization": self._get_auth_header(),
                },
                timeout=30.0,
            )
        return self._client

    async def send_message(
        self,
        stream: str,
        topic: str,
        content: str,
    ) -> MessageResult:
        """
        Send a message to a Zulip stream.

        Args:
            stream: Target stream name (e.g., "standup")
            topic: Message topic (e.g., "arunank-updates")
            content: Message content (markdown supported)

        Returns:
            MessageResult with send status
        """
        try:
            client = await self._get_client()

            # Zulip uses form-encoded data
            data = {
                "type": "stream",
                "to": stream,
                "topic": topic,
                "content": content,
            }

            response = await client.post(
                "/api/v1/messages",
                data=data,
            )

            result = response.json()

            if response.status_code == 200 and result.get("result") == "success":
                return MessageResult(
                    success=True,
                    message_id=result.get("id"),
                    stream=stream,
                    topic=topic,
                )
            else:
                error_msg = result.get("msg", "Unknown error")
                logger.error(f"Zulip send failed: {error_msg}")
                return MessageResult(
                    success=False,
                    stream=stream,
                    topic=topic,
                    error=error_msg,
                )

        except httpx.TimeoutException:
            logger.error("Zulip request timed out")
            return MessageResult(
                success=False,
                stream=stream,
                topic=topic,
                error="Request timed out",
            )
        except Exception as e:
            logger.error(f"Zulip send error: {e}", exc_info=True)
            return MessageResult(
                success=False,
                stream=stream,
                topic=topic,
                error=str(e),
            )

    async def send_private_message(
        self,
        recipients: list[str],
        content: str,
    ) -> MessageResult:
        """
        Send a private message to one or more users.

        Args:
            recipients: List of user email addresses
            content: Message content

        Returns:
            MessageResult with send status
        """
        try:
            client = await self._get_client()

            # Zulip uses comma-separated emails for private messages
            data = {
                "type": "private",
                "to": ",".join(recipients),
                "content": content,
            }

            response = await client.post(
                "/api/v1/messages",
                data=data,
            )

            result = response.json()

            if response.status_code == 200 and result.get("result") == "success":
                return MessageResult(
                    success=True,
                    message_id=result.get("id"),
                )
            else:
                error_msg = result.get("msg", "Unknown error")
                error_code = result.get("code", "unknown")
                logger.error(
                    f"Zulip private message failed: {error_msg} "
                    f"(code={error_code}, status={response.status_code})"
                )
                return MessageResult(
                    success=False,
                    error=error_msg,
                )

        except Exception as e:
            logger.error(f"Zulip private message error: {e}", exc_info=True)
            return MessageResult(
                success=False,
                error=str(e),
            )

    async def list_streams(self) -> list[StreamInfo]:
        """
        List all subscribed streams.

        Returns:
            List of StreamInfo objects
        """
        try:
            client = await self._get_client()

            response = await client.get("/api/v1/streams")
            result = response.json()

            if result.get("result") != "success":
                logger.error(f"Zulip list streams failed: {result.get('msg')}")
                return []

            streams = []
            for stream_data in result.get("streams", []):
                streams.append(
                    StreamInfo(
                        stream_id=stream_data.get("stream_id", 0),
                        name=stream_data.get("name", ""),
                        description=stream_data.get("description", ""),
                    )
                )

            return streams

        except Exception as e:
            logger.error(f"Zulip list streams error: {e}", exc_info=True)
            return []

    async def get_stream_topics(self, stream: str) -> list[str]:
        """
        Get topics in a stream.

        Args:
            stream: Stream name

        Returns:
            List of topic names
        """
        try:
            client = await self._get_client()

            # First get stream ID
            response = await client.get("/api/v1/streams")
            result = response.json()

            if result.get("result") != "success":
                logger.error(
                    f"Zulip get streams failed: {result.get('msg')} "
                    f"(code={result.get('code', 'unknown')})"
                )
                return []

            stream_id = None
            for s in result.get("streams", []):
                if s.get("name") == stream:
                    stream_id = s.get("stream_id")
                    break

            if stream_id is None:
                logger.error(f"Stream '{stream}' not found in available streams")
                return []

            # Get topics for stream
            response = await client.get(f"/api/v1/users/me/{stream_id}/topics")
            result = response.json()

            if result.get("result") != "success":
                logger.error(
                    f"Zulip get topics failed for stream '{stream}': {result.get('msg')} "
                    f"(code={result.get('code', 'unknown')})"
                )
                return []

            return [t.get("name", "") for t in result.get("topics", [])]

        except Exception as e:
            logger.error(f"Zulip get topics error: {e}", exc_info=True)
            return []

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
