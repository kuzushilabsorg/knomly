"""
Configuration Schemas for Knomly.

Pydantic models for configuration data stored in MongoDB.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PromptConfig(BaseModel):
    """
    LLM prompt configuration.

    Stored in MongoDB 'prompts' collection.
    """

    name: str = Field(..., description="Unique prompt identifier")
    system_prompt: str = Field(..., description="System prompt content")
    user_template: Optional[str] = Field(None, description="User message template with {placeholders}")
    model: Optional[str] = Field(None, description="Override model for this prompt")
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: int = Field(1024, ge=1)
    description: str = Field("", description="Human-readable description")
    version: int = Field(1, description="Prompt version for A/B testing")
    active: bool = Field(True, description="Whether prompt is active")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

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
    email: Optional[str] = Field(None, description="Email address")

    # Zulip settings
    zulip_stream: str = Field("standup", description="Default Zulip stream")
    zulip_topic: str = Field(..., description="Zulip topic for this user")
    zulip_email: Optional[str] = Field(None, description="Zulip user email")

    # Preferences
    language_preference: str = Field("en", description="Preferred language code")
    timezone: str = Field("Asia/Kolkata", description="User timezone")
    active: bool = Field(True, description="Whether user is active")

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        extra = "allow"


class PipelineAuditLog(BaseModel):
    """
    Audit log for pipeline execution.

    Stored in MongoDB 'pipeline_audit' collection.
    """

    execution_id: str = Field(..., description="Unique execution identifier")
    started_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None
    duration_ms: float = Field(0.0)

    # Request context
    sender_phone: Optional[str] = None
    message_type: str = "audio"
    user_id: Optional[str] = None
    user_name: Optional[str] = None

    # Execution results
    success: bool = False
    error: Optional[str] = None
    processor_timings: Dict[str, float] = Field(default_factory=dict)
    frame_count: int = 0

    # Output summary
    output_frame_types: List[str] = Field(default_factory=list)
    zulip_message_id: Optional[int] = None
    confirmation_sent: bool = False

    class Config:
        extra = "allow"


class AppSettings(BaseModel):
    """
    Application settings model.

    Used for type-safe settings access.
    """

    # Service identity
    service_name: str = "knomly"
    environment: str = "development"
    debug: bool = False

    # MongoDB
    mongodb_url: str = Field(..., description="MongoDB connection URL")
    mongodb_database: str = "knomly"

    # Provider API keys
    gemini_api_key: str = Field(..., description="Google Gemini API key")
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None

    # Zulip
    zulip_site: str = Field(..., description="Zulip server URL")
    zulip_bot_email: str = Field(..., description="Zulip bot email")
    zulip_api_key: str = Field(..., description="Zulip bot API key")

    # Twilio
    twilio_account_sid: str = Field(..., description="Twilio account SID")
    twilio_auth_token: str = Field(..., description="Twilio auth token")
    twilio_whatsapp_number: str = Field(..., description="Twilio WhatsApp number")

    # Provider selection
    default_stt_provider: str = "gemini"
    default_llm_provider: str = "openai"
    default_chat_provider: str = "zulip"

    class Config:
        env_prefix = "KNOMLY_"
        case_sensitive = False
