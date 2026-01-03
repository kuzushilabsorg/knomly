"""
Zulip Processor for Knomly.

Posts standup messages to Zulip.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ..processor import Processor

if TYPE_CHECKING:
    from ..context import PipelineContext
    from ..frames import Frame

logger = logging.getLogger(__name__)


class ZulipProcessor(Processor):
    """
    Posts standup messages to Zulip.

    Input: ExtractionFrame
    Output: ZulipMessageFrame with post result

    Uses Chat provider from ctx.providers.
    """

    def __init__(self, provider_name: str | None = None):
        """
        Args:
            provider_name: Specific Chat provider to use (or default)
        """
        self._provider_name = provider_name

    @property
    def name(self) -> str:
        return "zulip"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | None":
        from ..frames import ExtractionFrame, ZulipMessageFrame

        if not isinstance(frame, ExtractionFrame):
            return frame

        if not frame.has_items:
            logger.warning("ExtractionFrame has no items to post")
            return ZulipMessageFrame(
                stream=frame.zulip_stream,
                topic=frame.zulip_topic,
                content="",
                success=False,
                error="No standup items to post",
                sender_phone=frame.sender_phone,
                source_frame_id=frame.id,
            )

        if ctx.providers is None:
            raise RuntimeError("No providers configured in context")

        chat = ctx.providers.get_chat(self._provider_name)

        # Format message
        content = frame.format_zulip_message()

        logger.info(
            f"Posting to {frame.zulip_stream} > {frame.zulip_topic} "
            f"({len(content)} chars)"
        )

        result = await chat.send_message(
            stream=frame.zulip_stream,
            topic=frame.zulip_topic,
            content=content,
        )

        if result.success:
            logger.info(f"Posted to Zulip, message_id={result.message_id}")
        else:
            logger.error(f"Zulip post failed: {result.error}")

        return ZulipMessageFrame(
            stream=frame.zulip_stream,
            topic=frame.zulip_topic,
            content=content,
            message_id=result.message_id,
            success=result.success,
            error=result.error,
            sender_phone=frame.sender_phone,
            source_frame_id=frame.id,
        )
