"""
Twilio Webhook Handler for Knomly.

Receives WhatsApp voice notes and triggers standup pipeline.

Uses the Transport Pattern for platform-agnostic message handling:
- TwilioTransport normalizes incoming webhook requests
- ConfirmationProcessor sends responses via transport registry

Architecture (v3 Dynamic Configuration):
    The webhook now supports two modes:
    1. Static Mode (legacy): Uses hardcoded PipelineFactory
    2. Dynamic Mode (v3): Uses PipelineResolver for per-user config

    The mode is determined by KNOMLY_USE_RESOLVER environment variable.
    Default is hybrid: static processors + dynamic tools.

Security:
    Webhook signature validation is REQUIRED in production.
    Set KNOMLY_TWILIO_AUTH_TOKEN to enable validation.
    Set KNOMLY_SKIP_TWILIO_VALIDATION=true to disable (development only).
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request

from knomly.pipeline import PipelineContext
from knomly.pipeline.transports import get_transport

if TYPE_CHECKING:
    from knomly.pipeline.frames import AudioInputFrame

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/webhook", tags=["webhooks"])

# Feature flag for v3 resolver
USE_RESOLVER = os.getenv("KNOMLY_USE_RESOLVER", "true").lower() == "true"

# Security: Twilio signature validation
TWILIO_AUTH_TOKEN = os.getenv("KNOMLY_TWILIO_AUTH_TOKEN", "")
SKIP_TWILIO_VALIDATION = os.getenv("KNOMLY_SKIP_TWILIO_VALIDATION", "false").lower() == "true"


async def _validate_twilio_signature(request: Request, form_data: dict[str, Any]) -> bool:
    """
    Validate Twilio webhook signature to prevent forged requests.

    Security: This is CRITICAL for production. Without validation,
    attackers can forge webhook requests and potentially trigger
    unauthorized actions.

    Args:
        request: FastAPI request object
        form_data: Form data from the request

    Returns:
        True if valid (or validation skipped), False if invalid

    Raises:
        HTTPException: If validation fails and not in skip mode
    """
    # Skip validation in development mode
    if SKIP_TWILIO_VALIDATION:
        if os.getenv("KNOMLY_ENVIRONMENT", "development") == "production":
            logger.warning(
                "[SECURITY] Twilio signature validation is DISABLED in production! "
                "Set KNOMLY_SKIP_TWILIO_VALIDATION=false"
            )
        return True

    # Require auth token for validation
    if not TWILIO_AUTH_TOKEN:
        logger.warning(
            "[SECURITY] KNOMLY_TWILIO_AUTH_TOKEN not set. Webhook signature validation disabled."
        )
        return True

    # Get signature from header
    signature = request.headers.get("X-Twilio-Signature", "")
    if not signature:
        logger.warning("[SECURITY] Missing X-Twilio-Signature header")
        raise HTTPException(status_code=401, detail="Missing Twilio signature")

    try:
        # Import Twilio validator
        from twilio.request_validator import RequestValidator

        validator = RequestValidator(TWILIO_AUTH_TOKEN)

        # Build the full URL that Twilio used to sign the request
        # Must match exactly what Twilio sent
        url = str(request.url)

        # Handle potential proxy headers
        forwarded_proto = request.headers.get("X-Forwarded-Proto")
        forwarded_host = request.headers.get("X-Forwarded-Host")

        if forwarded_proto and forwarded_host:
            # Reconstruct URL with forwarded headers
            url = f"{forwarded_proto}://{forwarded_host}{request.url.path}"
            if request.url.query:
                url = f"{url}?{request.url.query}"

        # Validate the signature
        # Form data must be passed as dict with string values
        params = {k: str(v) for k, v in form_data.items()}
        is_valid = validator.validate(url, params, signature)

        if not is_valid:
            logger.warning(f"[SECURITY] Invalid Twilio signature for URL: {url[:100]}")
            raise HTTPException(status_code=401, detail="Invalid Twilio signature")

        logger.debug("[SECURITY] Twilio signature validated successfully")
        return True

    except ImportError:
        logger.warning("[SECURITY] twilio package not installed. Install with: pip install twilio")
        return True
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[SECURITY] Signature validation error: {e}")
        # Fail closed: reject on validation error
        raise HTTPException(status_code=401, detail="Signature validation failed")


def _normalize_phone(phone: str) -> str:
    """Normalize phone number to digits only."""
    digits = "".join(c for c in phone if c.isdigit())
    # Ensure country code
    if len(digits) == 10:
        digits = "91" + digits  # Default to India
    return digits


async def _process_standup_pipeline(
    initial_frame: AudioInputFrame,
    channel_id: str,
    webhook_data: dict[str, Any],
) -> None:
    """
    Background task to process standup pipeline.

    This runs asynchronously after webhook response.

    Architecture (v3):
        When USE_RESOLVER is enabled:
        1. Lookup user_id from phone number
        2. Load tools dynamically via PipelineResolver
        3. Inject tools into pipeline context
        4. Execute pipeline with dynamic tools

    Args:
        initial_frame: The normalized AudioInputFrame
        channel_id: Transport channel identifier
        webhook_data: Raw webhook data for debugging
    """
    try:
        # Import here to avoid circular imports at module load
        from knomly.app.dependencies import (
            get_config_service,
            get_pipeline,
            get_providers,
            resolve_tools_for_user,
        )

        # Get dependencies
        pipeline = get_pipeline()
        providers = get_providers()
        config_service = get_config_service()

        # Lookup user configuration
        user_config = await config_service.get_user_by_phone(initial_frame.sender_phone)

        # Determine user_id
        user_id = user_config.user_id if user_config else f"phone:{initial_frame.sender_phone}"

        # =================================================================
        # v3: Dynamic Tool Loading via PipelineResolver
        # =================================================================
        dynamic_tools: list = []

        if USE_RESOLVER:
            try:
                dynamic_tools = await resolve_tools_for_user(user_id)
                if dynamic_tools:
                    logger.info(
                        f"[v3] Loaded {len(dynamic_tools)} dynamic tools for user={user_id}: "
                        f"{[t.name for t in dynamic_tools]}"
                    )
            except Exception as e:
                logger.warning(f"[v3] Failed to load dynamic tools: {e}")
                # Continue with static pipeline

        # Create pipeline context with channel_id for transport lookup
        context = PipelineContext(
            sender_phone=initial_frame.sender_phone,
            message_type="audio",
            channel_id=channel_id,  # Enable transport-based confirmation
            providers=providers,
            config=config_service,
        )

        # Populate user context if found
        if user_config:
            context.user_id = user_config.user_id
            context.user_name = user_config.user_name
            context.zulip_stream = user_config.zulip_stream
            context.zulip_topic = user_config.zulip_topic

        # Inject dynamic tools into context metadata
        if dynamic_tools:
            context.metadata["dynamic_tools"] = dynamic_tools
            context.metadata["resolver_enabled"] = True

        # Execute pipeline
        logger.info(
            f"Starting pipeline for {initial_frame.sender_phone} "
            f"(resolver={'enabled' if USE_RESOLVER else 'disabled'}, "
            f"tools={len(dynamic_tools)})"
        )
        result = await pipeline.execute(initial_frame, context)

        if result.success:
            logger.info(
                f"Pipeline completed successfully: {str(result.execution_id)[:8]}... "
                f"in {result.duration_ms:.1f}ms"
            )
        else:
            logger.error(f"Pipeline failed: {str(result.execution_id)[:8]}... - {result.error}")

        # Log audit record
        if config_service:
            from knomly.config.schemas import PipelineAuditLog

            audit = PipelineAuditLog(
                execution_id=str(result.execution_id),
                started_at=context.started_at,
                duration_ms=result.duration_ms,
                sender_phone=initial_frame.sender_phone,
                message_type="audio",
                user_id=context.user_id,
                user_name=context.user_name,
                success=result.success,
                error=result.error,
                processor_timings=context.processor_timings,
                frame_count=len(context.frame_log),
                output_frame_types=[f.frame_type for f in result.output_frames],
            )
            await config_service.log_pipeline_execution(audit)

    except Exception as e:
        logger.error(f"Pipeline execution error: {e}", exc_info=True)


@router.post(
    "/twilio",
    summary="Receive Twilio WhatsApp webhook",
    responses={
        200: {"description": "Message received and queued"},
        400: {"description": "Invalid webhook payload"},
    },
)
async def receive_twilio_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Receive and process incoming WhatsApp messages via Twilio.

    Handles voice notes by:
    1. Validating Twilio signature (security)
    2. Using TwilioTransport to normalize the webhook payload
    3. Validating message type (audio only for standup)
    4. Queuing background pipeline execution
    5. Returning immediate acknowledgment

    Non-audio messages receive a help response.
    """
    try:
        form = await request.form()
        form_data = dict(form)

        # Security: Validate Twilio signature
        await _validate_twilio_signature(request, form_data)

        # Extract sender info
        raw_from = str(form.get("From") or "")
        sender_phone = _normalize_phone(raw_from)

        if not sender_phone:
            logger.warning("Twilio webhook missing sender phone")
            return {"status": "failed", "message": "missing sender"}

        # Extract message details
        message_type = str(form.get("MessageType") or "").strip()
        num_media = int(str(form.get("NumMedia") or "0") or 0)
        body = str(form.get("Body") or "").strip()
        profile_name = str(form.get("ProfileName") or "").strip()

        logger.info(
            f"Twilio webhook: from={sender_phone}, type={message_type}, "
            f"num_media={num_media}, has_body={bool(body)}"
        )

        # Collect webhook data for context
        webhook_data = {
            "from": raw_from,
            "sender_phone": sender_phone,
            "message_type": message_type,
            "num_media": num_media,
            "body": body,
            "profile_name": profile_name,
            "message_sid": str(form.get("MessageSid") or ""),
        }

        # Handle audio messages (standup voice notes)
        if message_type == "audio" and num_media > 0:
            # Get Twilio transport for normalization
            try:
                transport = get_transport("twilio")
            except Exception:
                # Fallback: transport not registered yet (legacy mode)
                logger.warning("Twilio transport not registered, using legacy normalization")
                return await _handle_audio_legacy(
                    form_data, sender_phone, profile_name, webhook_data, background_tasks
                )

            # Use transport to normalize request
            try:
                initial_frame = await transport.normalize_request(request, form_data)
                logger.info(f"Processing audio message: {initial_frame.media_url[:50]}...")

                # Queue background processing
                background_tasks.add_task(
                    _process_standup_pipeline,
                    initial_frame=initial_frame,
                    channel_id=transport.channel_id,
                    webhook_data=webhook_data,
                )

                return {"status": "success", "message": "audio processing queued"}

            except ValueError as e:
                logger.warning(f"Invalid audio webhook: {e}")
                return {"status": "failed", "message": str(e)}

        # Handle text messages (provide help)
        elif message_type == "text" and body:
            logger.info(f"Text message received: {body[:50]}...")
            # TODO: Could trigger text-based standup or help response
            return {"status": "success", "message": "text received"}

        # Unsupported message types
        else:
            logger.info(f"Unsupported message type: {message_type}")
            return {"status": "success", "message": "unsupported type"}

    except Exception as e:
        logger.error(f"Twilio webhook error: {e}", exc_info=True)
        return {"status": "failed", "message": "internal error"}


async def _handle_audio_legacy(
    form_data: dict[str, Any],
    sender_phone: str,
    profile_name: str,
    webhook_data: dict[str, Any],
    background_tasks: BackgroundTasks,
) -> dict[str, str]:
    """
    Legacy audio handling without transport abstraction.

    This is a fallback for when TwilioTransport is not registered.
    Deprecated: Register TwilioTransport at app startup instead.
    """
    from knomly.pipeline.frames import AudioInputFrame

    media_url = str(form_data.get("MediaUrl0") or "").strip()
    media_type = str(form_data.get("MediaContentType0") or "audio/ogg").strip()

    if not media_url:
        logger.warning("Audio message missing media URL")
        return {"status": "failed", "message": "missing media url"}

    logger.info(f"Processing audio message (legacy): {media_url[:50]}...")

    initial_frame = AudioInputFrame(
        media_url=media_url,
        mime_type=media_type,
        sender_phone=sender_phone,
        profile_name=profile_name,
        channel_id="",  # Legacy mode: no channel_id
    )

    # Queue background processing
    background_tasks.add_task(
        _process_standup_pipeline,
        initial_frame=initial_frame,
        channel_id="",  # Legacy mode
        webhook_data=webhook_data,
    )

    return {"status": "success", "message": "audio processing queued"}


@router.post(
    "/twilio/status",
    summary="Twilio delivery status callback",
    responses={
        200: {"description": "Status processed"},
    },
)
async def receive_twilio_status(request: Request) -> dict[str, str]:
    """
    Handle Twilio message delivery status updates.

    Currently just logs status for monitoring.
    Could be extended to track delivery confirmation.
    """
    try:
        form = await request.form()

        message_sid = str(form.get("MessageSid") or form.get("SmsSid") or "")
        status = str(form.get("MessageStatus") or "").lower()
        to_number = str(form.get("To") or "")

        logger.info(f"Twilio status: sid={message_sid}, status={status}, to={to_number}")

        return {"status": "success", "message": "status processed"}

    except Exception as e:
        logger.error(f"Twilio status error: {e}", exc_info=True)
        return {"status": "failed", "message": "error processing status"}


# Type hints
if __name__ != "__main__":
    pass
