"""
Configuration Schemas for Knomly.

Pydantic models for configuration data stored in MongoDB.

Security:
    Sensitive fields use SecretStr to prevent accidental logging
    of credentials. Access the value with `.get_secret_value()`.
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel, Field, SecretStr


def _utc_now() -> datetime:
    """Get current UTC time with timezone awareness."""
    return datetime.now(UTC)


class PromptConfig(BaseModel):
    """
    LLM prompt configuration.

    Stored in MongoDB 'prompts' collection.
    """

    name: str = Field(..., description="Unique prompt identifier")
    system_prompt: str = Field(..., description="System prompt content")
    user_template: str | None = Field(None, description="User message template with {placeholders}")
    model: str | None = Field(None, description="Override model for this prompt")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1)
    description: str = Field("", description="Human-readable description")
    version: int = Field(1, description="Prompt version for A/B testing")
    active: bool = Field(True, description="Whether prompt is active")
    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    class Config:
        extra = "allow"


class UserConfig(BaseModel):
    """
    User configuration.

    Stored in MongoDB 'users' collection.
    Maps phone numbers to user settings and Zulip channels.
    """

    phone: str = Field(..., description="Phone number (E.164 format)")
    user_id: str = Field(..., description="Unique user identifier")
    user_name: str = Field(..., description="Display name")
    email: str | None = Field(None, description="Email address")

    # Zulip settings
    zulip_stream: str = Field("standup", description="Default Zulip stream")
    zulip_topic: str = Field(..., description="Zulip topic for this user")
    zulip_email: str | None = Field(None, description="Zulip user email")

    # Preferences
    language_preference: str = Field("en", description="Preferred language code")
    timezone: str = Field("Asia/Kolkata", description="User timezone")
    active: bool = Field(True, description="Whether user is active")

    created_at: datetime = Field(default_factory=_utc_now)
    updated_at: datetime = Field(default_factory=_utc_now)

    class Config:
        extra = "allow"


class PipelineAuditLog(BaseModel):
    """
    Audit log for pipeline execution.

    Stored in MongoDB 'pipeline_audit' collection.
    """

    execution_id: str = Field(..., description="Unique execution identifier")
    started_at: datetime = Field(default_factory=_utc_now)
    completed_at: datetime | None = None
    duration_ms: float = Field(0.0)

    # Request context
    sender_phone: str | None = None
    message_type: str = "audio"
    user_id: str | None = None
    user_name: str | None = None

    # Execution results
    success: bool = False
    error: str | None = None
    processor_timings: dict[str, float] = Field(default_factory=dict)
    frame_count: int = 0

    # Output summary
    output_frame_types: list[str] = Field(default_factory=list)
    zulip_message_id: int | None = None
    confirmation_sent: bool = False

    class Config:
        extra = "allow"


class AppSettings(BaseModel):
    """
    Application settings model.

    Used for type-safe settings access.

    Security:
        API keys and tokens use SecretStr to prevent accidental logging.
        Access secret values with: settings.api_key.get_secret_value()
    """

    # Service identity
    service_name: str = "knomly"
    environment: str = "development"
    debug: bool = False

    # MongoDB
    mongodb_url: SecretStr = Field(..., description="MongoDB connection URL")
    mongodb_database: str = "knomly"

    # Provider API keys (SecretStr prevents accidental logging)
    gemini_api_key: SecretStr = Field(default=SecretStr(""), description="Google Gemini API key")
    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None

    # Zulip
    zulip_site: str = Field(default="", description="Zulip server URL")
    zulip_bot_email: str = Field(default="", description="Zulip bot email")
    zulip_api_key: SecretStr = Field(default=SecretStr(""), description="Zulip bot API key")

    # Twilio
    twilio_account_sid: str = Field(default="", description="Twilio account SID")
    twilio_auth_token: SecretStr = Field(default=SecretStr(""), description="Twilio auth token")
    twilio_whatsapp_number: str = Field(default="", description="Twilio WhatsApp number")

    # Provider selection
    default_stt_provider: str = "gemini"
    default_llm_provider: str = "openai"
    default_chat_provider: str = "zulip"

    class Config:
        env_prefix = "KNOMLY_"
        case_sensitive = False
