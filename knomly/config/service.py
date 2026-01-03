"""
Configuration Service for Knomly.

Provides access to MongoDB-stored configuration with caching.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from .schemas import PipelineAuditLog, PromptConfig, UserConfig

logger = logging.getLogger(__name__)


class TTLCache:
    """Simple TTL cache for configuration data."""

    def __init__(self, ttl_seconds: int = 300):
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)

    def get(self, key: str) -> Any | None:
        if key in self._cache:
            value, expires = self._cache[key]
            if datetime.now(UTC) < expires:
                return value
            del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        expires = datetime.now(UTC) + self._ttl
        self._cache[key] = (value, expires)

    def clear(self) -> None:
        self._cache.clear()


class ConfigurationService:
    """
    Service for accessing configuration from MongoDB.

    Collections:
    - prompts: LLM prompt templates
    - users: User settings and Zulip mappings
    - pipeline_audit: Execution logs

    Caching:
    - Prompts cached for 5 minutes
    - User configs cached for 5 minutes
    """

    def __init__(
        self,
        mongodb_url: str,
        database_name: str = "knomly",
        cache_ttl: int = 300,
    ):
        """
        Initialize configuration service.

        Args:
            mongodb_url: MongoDB connection URL
            database_name: Database name
            cache_ttl: Cache TTL in seconds
        """
        self._mongodb_url = mongodb_url
        self._database_name = database_name
        self._client = None
        self._db = None
        self._cache = TTLCache(ttl_seconds=cache_ttl)

    async def connect(self) -> None:
        """Connect to MongoDB."""
        try:
            from motor.motor_asyncio import AsyncIOMotorClient

            self._client = AsyncIOMotorClient(self._mongodb_url)
            self._db = self._client[self._database_name]
            logger.info(f"Connected to MongoDB database: {self._database_name}")
        except ImportError:
            raise ImportError(
                "motor package is required for MongoDB. Install with: pip install motor"
            )

    async def close(self) -> None:
        """Close MongoDB connection."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None

    async def _ensure_connected(self) -> None:
        """Ensure MongoDB connection is established."""
        if self._db is None:
            await self.connect()

    # ==================== Prompts ====================

    async def get_prompt(self, name: str) -> PromptConfig | None:
        """
        Get a prompt configuration by name.

        Args:
            name: Prompt name

        Returns:
            PromptConfig or None if not found
        """
        # Check cache first
        cache_key = f"prompt:{name}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._ensure_connected()

        doc = await self._db.prompts.find_one({"name": name, "active": True})

        if doc is None:
            return None

        config = PromptConfig(**doc)
        self._cache.set(cache_key, config)
        return config

    async def upsert_prompt(self, prompt: PromptConfig) -> None:
        """
        Insert or update a prompt configuration.

        Args:
            prompt: Prompt configuration to save
        """
        await self._ensure_connected()

        prompt.updated_at = datetime.now(UTC)
        await self._db.prompts.update_one(
            {"name": prompt.name},
            {"$set": prompt.model_dump()},
            upsert=True,
        )

        # Invalidate cache
        self._cache.clear()

    async def list_prompts(self, active_only: bool = True) -> list[PromptConfig]:
        """
        List all prompts.

        Args:
            active_only: Only return active prompts

        Returns:
            List of PromptConfig objects
        """
        await self._ensure_connected()

        query = {"active": True} if active_only else {}
        cursor = self._db.prompts.find(query)

        prompts = []
        async for doc in cursor:
            prompts.append(PromptConfig(**doc))

        return prompts

    # ==================== Users ====================

    async def get_user_by_phone(self, phone: str) -> UserConfig | None:
        """
        Get user configuration by phone number.

        Args:
            phone: Phone number (E.164 format, e.g., "+919876543210")

        Returns:
            UserConfig or None if not found
        """
        # Normalize phone number
        normalized = self._normalize_phone(phone)

        # Check cache first
        cache_key = f"user:{normalized}"
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        await self._ensure_connected()

        doc = await self._db.users.find_one({"phone": normalized, "active": True})

        if doc is None:
            return None

        config = UserConfig(**doc)
        self._cache.set(cache_key, config)
        return config

    async def upsert_user(self, user: UserConfig) -> None:
        """
        Insert or update a user configuration.

        Args:
            user: User configuration to save
        """
        await self._ensure_connected()

        user.updated_at = datetime.now(UTC)
        await self._db.users.update_one(
            {"phone": user.phone},
            {"$set": user.model_dump()},
            upsert=True,
        )

        # Invalidate cache
        self._cache.clear()

    async def list_users(self, active_only: bool = True) -> list[UserConfig]:
        """
        List all users.

        Args:
            active_only: Only return active users

        Returns:
            List of UserConfig objects
        """
        await self._ensure_connected()

        query = {"active": True} if active_only else {}
        cursor = self._db.users.find(query)

        users = []
        async for doc in cursor:
            users.append(UserConfig(**doc))

        return users

    # ==================== Audit Logs ====================

    async def log_pipeline_execution(self, audit: PipelineAuditLog) -> None:
        """
        Log a pipeline execution.

        Args:
            audit: Audit log to save
        """
        await self._ensure_connected()
        await self._db.pipeline_audit.insert_one(audit.model_dump())

    async def get_recent_executions(
        self,
        limit: int = 10,
        phone: str | None = None,
    ) -> list[PipelineAuditLog]:
        """
        Get recent pipeline executions.

        Args:
            limit: Maximum number of results
            phone: Optional filter by phone number

        Returns:
            List of PipelineAuditLog objects
        """
        await self._ensure_connected()

        query = {}
        if phone:
            query["sender_phone"] = self._normalize_phone(phone)

        cursor = self._db.pipeline_audit.find(query).sort("started_at", -1).limit(limit)

        logs = []
        async for doc in cursor:
            logs.append(PipelineAuditLog(**doc))

        return logs

    # ==================== Utilities ====================

    def _normalize_phone(self, phone: str) -> str:
        """Normalize phone number to consistent format."""
        # Remove all non-digit characters
        digits = "".join(c for c in phone if c.isdigit())

        # Ensure it has country code
        if len(digits) == 10:
            digits = "91" + digits  # Default to India

        return digits

    async def seed_default_prompts(self) -> None:
        """Seed default prompts if they don't exist."""
        await self._ensure_connected()

        # Standup extraction prompt
        standup_prompt = PromptConfig(
            name="standup_extraction",
            system_prompt="""You are a standup extraction assistant. Your job is to extract structured standup information from voice message transcriptions.

Extract the following from the transcription:
1. today_items: List of tasks planned for today
2. yesterday_items: List of tasks completed yesterday (if mentioned)
3. blockers: List of blockers or issues (if any)
4. summary: Brief 1-sentence summary

Return a JSON object with this structure:
{
    "today_items": ["task 1", "task 2"],
    "yesterday_items": ["completed task"],
    "blockers": ["blocker description"],
    "summary": "Brief summary of the standup"
}

If a category is not mentioned, use an empty list.
Always return valid JSON.""",
            user_template="Transcription:\n{transcription}",
            temperature=0.3,
            max_tokens=512,
            description="Extracts standup items from voice transcription",
        )

        existing = await self.get_prompt("standup_extraction")
        if existing is None:
            await self.upsert_prompt(standup_prompt)
            logger.info("Seeded standup_extraction prompt")
