"""
Pipeline Definition Schema.

JSON-serializable schema for complete pipeline configuration that can be
stored in databases (MongoDB), cached (Redis), and loaded at runtime.

Design Principle:
    A PipelinePacket is the "executable configuration" - it contains
    everything needed to initialize a pipeline for a specific session.

Comparison to ChitChat ExecutablePipelinePacket:
    We follow the same pattern:
    - SessionContext: Who/what/where
    - AgentContext: System prompt, behavior
    - PipelineProviderConfig: STT/LLM/TTS providers
    - Tool definitions (added for agent capabilities)

Flow:
    1. Admin configures Pipeline in MongoDB (via API)
    2. On session start, resolve Pipeline to PipelinePacket
    3. Store PipelinePacket in Redis (for agent to retrieve)
    4. Agent reads from Redis and initializes providers/tools

Usage:
    # Create packet for a session
    packet = PipelinePacket.create_for_session(
        session_id="sess-123",
        user_id="user-456",
        system_prompt="You are a helpful assistant...",
        stt=ProviderDefinition.stt("deepgram", model="nova-2"),
        llm=ProviderDefinition.llm("openai", model="gpt-4o"),
        tools=[
            ToolDefinition(name="create_task", ...),
            ToolDefinition(name="search_tasks", ...),
        ],
    )

    # Store in Redis
    await redis.set(packet.to_redis_key(), packet.model_dump_json())

    # Retrieve and use
    packet = PipelinePacket.model_validate_json(await redis.get(key))
    stt = factory.create_service(packet.providers.stt, secrets)
    tools = await adapter.build_tools(packet.tools, context)
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from .provider import ProviderDefinition
from .tool import ToolDefinition


class SessionContext(BaseModel):
    """
    Context for the current session.

    Identifies who is using the system and where.

    Attributes:
        session_id: Unique session identifier
        user_id: User/tenant identifier
        room_name: LiveKit room name (for voice)
        locale: Language/locale code
        metadata: Additional context
    """

    session_id: str = Field(..., description="Unique session ID")
    user_id: str = Field(..., description="User/tenant ID")
    room_name: str = Field(default="", description="LiveKit room name")
    locale: str = Field(default="en", description="Language/locale code")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional session metadata",
    )

    class Config:
        populate_by_name = True


class AgentContext(BaseModel):
    """
    Agent behavior configuration.

    Defines how the agent should behave in this session.

    Attributes:
        system_prompt: Base system prompt for LLM
        welcome_message: Message to send on connect
        voice_id: TTS voice ID (for voice sessions)
        persona_name: Agent persona name
        persona_traits: Personality traits
    """

    system_prompt: str = Field(default="", description="System prompt for LLM")
    welcome_message: str = Field(default="", description="Welcome message")
    voice_id: str | None = Field(default=None, description="TTS voice ID")
    persona_name: str = Field(default="Assistant", description="Agent name")
    persona_traits: list[str] = Field(
        default_factory=list,
        description="Personality traits",
    )

    # Optional behavioral controls
    max_response_tokens: int = Field(default=1024, description="Max LLM output")
    response_style: str = Field(default="concise", description="Response style")

    class Config:
        populate_by_name = True


class PipelineProviderConfig(BaseModel):
    """
    Provider configuration for the pipeline.

    Contains ProviderDefinition for each service type.
    Services not needed can be None.

    Attributes:
        stt: Speech-to-text provider
        llm: Language model provider
        tts: Text-to-speech provider
        chat: Chat/messaging provider (e.g., Zulip)
        vad: Voice activity detection
    """

    stt: ProviderDefinition | None = Field(default=None, description="STT provider")
    llm: ProviderDefinition | None = Field(default=None, description="LLM provider")
    tts: ProviderDefinition | None = Field(default=None, description="TTS provider")
    chat: ProviderDefinition | None = Field(default=None, description="Chat provider")
    vad: ProviderDefinition | None = Field(default=None, description="VAD provider")

    class Config:
        populate_by_name = True

    @classmethod
    def with_defaults(
        cls,
        stt_provider: str = "deepgram",
        llm_provider: str = "openai",
        tts_provider: str | None = None,
        locale: str = "en",
    ) -> PipelineProviderConfig:
        """Create with default provider settings."""
        return cls(
            stt=ProviderDefinition.stt(
                stt_provider,
                model="nova-2" if stt_provider == "deepgram" else None,
                language=locale,
            ),
            llm=ProviderDefinition.llm(llm_provider, model="gpt-4o"),
            tts=(ProviderDefinition.tts(tts_provider) if tts_provider else None),
        )


class PipelineParams(BaseModel):
    """
    Pipeline execution parameters.

    Controls behavior of the pipeline execution.

    Attributes:
        allow_interruptions: Allow user to interrupt agent
        enable_metrics: Track performance metrics
        enable_memory: Use conversation memory
        max_agent_iterations: Limit agent loop iterations
        timeout_seconds: Maximum execution time
    """

    allow_interruptions: bool = Field(default=True)
    enable_metrics: bool = Field(default=True)
    enable_memory: bool = Field(default=True)
    enable_persistence: bool = Field(default=True)
    max_agent_iterations: int = Field(default=5)
    timeout_seconds: float = Field(default=60.0)

    class Config:
        populate_by_name = True


class PipelinePacket(BaseModel):
    """
    Complete pipeline configuration packet.

    This is the main "executable configuration" that contains everything
    needed to initialize a pipeline for a specific session.

    Designed to be:
    - JSON-serializable for storage in Redis/MongoDB
    - Self-contained (no external dependencies)
    - Versioned for schema evolution

    Usage:
        # Create for session
        packet = PipelinePacket.create_for_session(
            session_id="sess-123",
            user_id="user-456",
            system_prompt="You are...",
        )

        # Store in Redis
        await redis.set(packet.to_redis_key(), packet.model_dump_json())

        # Load and use
        packet = PipelinePacket.model_validate_json(json_str)
    """

    # Session identification
    session: SessionContext = Field(..., description="Session context")

    # Agent behavior
    agent: AgentContext = Field(
        default_factory=AgentContext,
        description="Agent configuration",
    )

    # Provider configuration
    providers: PipelineProviderConfig = Field(
        default_factory=PipelineProviderConfig,
        description="Provider definitions",
    )

    # Execution parameters
    params: PipelineParams = Field(
        default_factory=PipelineParams,
        description="Pipeline parameters",
    )

    # Tool definitions (for agentic pipelines)
    tools: list[ToolDefinition] = Field(
        default_factory=list,
        description="Available tool definitions",
    )

    # Pipeline identity (from admin config)
    pipeline_id: str | None = Field(default=None, description="Pipeline ID from DB")
    pipeline_code: str | None = Field(default=None, description="Pipeline code name")

    # Timestamps
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Creation timestamp",
    )

    # Schema version
    version: int = Field(default=1, description="Schema version")

    class Config:
        populate_by_name = True

    def to_redis_key(self) -> str:
        """Generate Redis key for this packet."""
        return f"pipeline:{self.session.session_id}"

    @classmethod
    def create_for_session(
        cls,
        session_id: str,
        user_id: str,
        *,
        room_name: str = "",
        locale: str = "en",
        system_prompt: str = "",
        welcome_message: str = "",
        voice_id: str | None = None,
        stt: ProviderDefinition | None = None,
        llm: ProviderDefinition | None = None,
        tts: ProviderDefinition | None = None,
        chat: ProviderDefinition | None = None,
        tools: list[ToolDefinition] | None = None,
        pipeline_id: str | None = None,
        pipeline_code: str | None = None,
        **metadata: Any,
    ) -> PipelinePacket:
        """
        Factory method to create a packet for a session.

        Args:
            session_id: Unique session ID
            user_id: User/tenant ID
            room_name: LiveKit room name
            locale: Language code
            system_prompt: System prompt for LLM
            welcome_message: Welcome message
            voice_id: TTS voice ID
            stt: STT provider definition
            llm: LLM provider definition
            tts: TTS provider definition
            chat: Chat provider definition
            tools: Tool definitions for agent
            pipeline_id: Pipeline ID from database
            pipeline_code: Pipeline code name
            **metadata: Additional session metadata

        Returns:
            Configured PipelinePacket
        """
        # Use defaults if not provided
        if llm is None:
            llm = ProviderDefinition.llm("openai", model="gpt-4o")
        if stt is None:
            stt = ProviderDefinition.stt("deepgram", model="nova-2", language=locale)

        return cls(
            session=SessionContext(
                session_id=session_id,
                user_id=user_id,
                room_name=room_name,
                locale=locale,
                metadata=metadata,
            ),
            agent=AgentContext(
                system_prompt=system_prompt,
                welcome_message=welcome_message,
                voice_id=voice_id,
            ),
            providers=PipelineProviderConfig(
                stt=stt,
                llm=llm,
                tts=tts,
                chat=chat,
            ),
            tools=tools or [],
            pipeline_id=pipeline_id,
            pipeline_code=pipeline_code,
        )

    def get_enabled_tools(self) -> list[ToolDefinition]:
        """Get only enabled tools."""
        return [t for t in self.tools if t.enabled]

    def get_tools_by_tag(self, tag: str) -> list[ToolDefinition]:
        """Get tools with a specific tag."""
        return [t for t in self.tools if t.enabled and tag in t.tags]
