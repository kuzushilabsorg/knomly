"""
Definition Loaders.

Implementations of the DefinitionLoader protocol for different storage backends.

Design Principle:
    Start simple, scale as needed.
    - Development: FileDefinitionLoader (YAML/JSON files)
    - Testing: MemoryDefinitionLoader (in-memory)
    - Production: MongoDefinitionLoader (implement in your project)

The loaders abstract away WHERE configuration comes from,
allowing the resolver to work with any backend.

Usage:
    # File-based (development)
    loader = FileDefinitionLoader("config/")

    # Memory (testing)
    loader = MemoryDefinitionLoader()
    await loader.add_tool(ToolDefinition(...))

    # MongoDB (production - implement in your project)
    class MongoDefinitionLoader:
        async def get_tools_for_user(self, user_id):
            docs = await self.db.tools.find({"user_id": user_id}).to_list()
            return [ToolDefinition.model_validate(d) for d in docs]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class FileDefinitionLoader:
    """
    Loads definitions from YAML/JSON files.

    This loader is designed for development and testing.
    It reads configuration from a directory structure:

    config/
    ├── tools/
    │   ├── plane.json          # All Plane tools
    │   ├── slack.json          # All Slack tools
    │   └── {user_id}.json      # User-specific tools
    ├── pipelines/
    │   ├── default.json        # Default pipeline
    │   └── {user_id}.json      # User-specific pipeline
    └── providers/
        └── default.json        # Provider definitions

    File Format (tools/*.json):
        [
            {
                "name": "create_task",
                "description": "Create a task",
                "source": "openapi",
                "source_config": {...}
            }
        ]

    File Format (pipelines/*.json):
        {
            "session": {"session_id": "...", "user_id": "..."},
            "agent": {"system_prompt": "..."},
            "providers": {"stt": {...}, "llm": {...}},
            "tools": [...]
        }

    Usage:
        loader = FileDefinitionLoader("config/")
        tools = await loader.get_tools_for_user("user-123")
        pipeline = await loader.get_pipeline_for_session("sess", "user")
    """

    def __init__(
        self,
        base_dir: str | Path,
        *,
        default_pipeline_file: str = "default.json",
    ):
        """
        Initialize loader.

        Args:
            base_dir: Base directory for config files
            default_pipeline_file: Filename for default pipeline
        """
        self._base_dir = Path(base_dir)
        self._default_pipeline_file = default_pipeline_file

    async def get_tools_for_user(
        self,
        user_id: str,
    ) -> list:
        """
        Get tool definitions for a user.

        Loads from:
        1. tools/{user_id}.json (user-specific)
        2. tools/*.json (all tool files, for global tools)

        Args:
            user_id: User identifier

        Returns:
            List of ToolDefinition objects
        """
        from knomly.adapters.schemas import ToolDefinition

        tools: list[ToolDefinition] = []
        tools_dir = self._base_dir / "tools"

        if not tools_dir.exists():
            logger.warning(f"[file_loader] Tools directory not found: {tools_dir}")
            return tools

        # Load user-specific tools first
        user_file = tools_dir / f"{user_id}.json"
        if user_file.exists():
            user_tools = self._load_json(user_file)
            if isinstance(user_tools, list):
                for t in user_tools:
                    tools.append(ToolDefinition.model_validate(t))

        # Load global tools (files not named after a user)
        for file in tools_dir.glob("*.json"):
            if file.name == f"{user_id}.json":
                continue  # Already loaded

            # Optionally skip user-specific files
            # (you could add a naming convention like _global.json)

            file_tools = self._load_json(file)
            if isinstance(file_tools, list):
                for t in file_tools:
                    tool = ToolDefinition.model_validate(t)
                    # Check if tool should be available to this user
                    # Default: all tools are global
                    if tool.enabled:
                        tools.append(tool)

        logger.info(f"[file_loader] Loaded {len(tools)} tools for user={user_id}")
        return tools

    async def get_pipeline_for_session(
        self,
        session_id: str,
        user_id: str,
        **context: Any,
    ):
        """
        Get pipeline configuration for a session.

        Loads from:
        1. pipelines/{user_id}.json (user-specific)
        2. pipelines/default.json (fallback)

        Args:
            session_id: Session identifier
            user_id: User identifier
            **context: Additional context

        Returns:
            PipelinePacket or None
        """
        from knomly.adapters.schemas import PipelinePacket

        pipelines_dir = self._base_dir / "pipelines"

        if not pipelines_dir.exists():
            logger.warning(f"[file_loader] Pipelines directory not found: {pipelines_dir}")
            return None

        # Try user-specific pipeline
        user_file = pipelines_dir / f"{user_id}.json"
        if user_file.exists():
            data = self._load_json(user_file)
            if data:
                # Update session context
                if "session" not in data:
                    data["session"] = {}
                data["session"]["session_id"] = session_id
                data["session"]["user_id"] = user_id

                logger.info(f"[file_loader] Loaded user pipeline: {user_file}")
                return PipelinePacket.model_validate(data)

        # Fall back to default
        default_file = pipelines_dir / self._default_pipeline_file
        if default_file.exists():
            data = self._load_json(default_file)
            if data:
                if "session" not in data:
                    data["session"] = {}
                data["session"]["session_id"] = session_id
                data["session"]["user_id"] = user_id

                logger.info(f"[file_loader] Loaded default pipeline: {default_file}")
                return PipelinePacket.model_validate(data)

        return None

    async def get_provider_definition(
        self,
        provider_type: str,
        provider_code: str,
        user_id: str | None = None,
    ):
        """
        Get provider definition.

        Args:
            provider_type: stt, llm, tts, chat
            provider_code: deepgram, openai, etc.
            user_id: Optional user for user-specific config

        Returns:
            ProviderDefinition or None
        """
        from knomly.adapters.schemas import ProviderDefinition

        providers_dir = self._base_dir / "providers"
        if not providers_dir.exists():
            return None

        # Load providers file
        providers_file = providers_dir / "default.json"
        if not providers_file.exists():
            return None

        data = self._load_json(providers_file)
        if not data:
            return None

        # Find matching provider
        type_providers = data.get(provider_type, {})
        if provider_code in type_providers:
            return ProviderDefinition.model_validate(type_providers[provider_code])

        return None

    def _load_json(self, path: Path) -> Any:
        """Load JSON file."""
        try:
            with path.open() as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"[file_loader] Failed to load {path}: {e}")
            return None


class MemoryDefinitionLoader:
    """
    In-memory definition loader for testing.

    Stores definitions in memory. Useful for unit tests
    and development without files.

    Usage:
        loader = MemoryDefinitionLoader()

        # Add tools
        await loader.add_tool(ToolDefinition(name="test", ...))

        # Add pipeline
        await loader.set_pipeline("user-1", PipelinePacket(...))

        # Use with resolver
        resolver = PipelineResolver(loader=loader, ...)
    """

    def __init__(self):
        self._tools: dict[str, list] = {}  # user_id -> tools
        self._global_tools: list = []
        self._pipelines: dict[str, Any] = {}  # user_id -> pipeline
        self._default_pipeline: Any = None

    async def add_tool(
        self,
        tool,
        user_id: str | None = None,
    ) -> None:
        """
        Add a tool definition.

        Args:
            tool: ToolDefinition to add
            user_id: User ID (None for global)
        """
        if user_id is None:
            self._global_tools.append(tool)
        else:
            if user_id not in self._tools:
                self._tools[user_id] = []
            self._tools[user_id].append(tool)

    async def set_pipeline(
        self,
        user_id: str,
        pipeline,
    ) -> None:
        """Set pipeline for a user."""
        self._pipelines[user_id] = pipeline

    async def set_default_pipeline(self, pipeline) -> None:
        """Set the default pipeline."""
        self._default_pipeline = pipeline

    async def get_tools_for_user(self, user_id: str) -> list:
        """Get tools for a user."""
        user_tools = self._tools.get(user_id, [])
        return [*self._global_tools, *user_tools]

    async def get_pipeline_for_session(
        self,
        session_id: str,
        user_id: str,
        **context: Any,
    ):
        """Get pipeline for a session."""
        # Try user-specific first
        if user_id in self._pipelines:
            return self._pipelines[user_id]

        # Fall back to default
        return self._default_pipeline

    async def get_provider_definition(
        self,
        provider_type: str,
        provider_code: str,
        user_id: str | None = None,
    ):
        """Get provider definition (not implemented for memory loader)."""
        return None

    def clear(self) -> None:
        """Clear all definitions."""
        self._tools.clear()
        self._global_tools.clear()
        self._pipelines.clear()
        self._default_pipeline = None
