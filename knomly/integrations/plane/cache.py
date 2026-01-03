"""
Entity Cache for Plane Integration.

This module provides a lightweight cache of Plane entities (projects, users,
states) to enable context-aware task creation without full agent loops.

Design Principle (ADR-004 compliant):
    - Cache is a SERVICE, not stored in PipelineContext
    - Data flows through FRAMES (via metadata), not context
    - Cache enables "short-term memory" without v2 complexity

Failure Mode (Graceful Degradation):
    - Context is AUXILIARY, not critical
    - If Plane API times out, return empty context
    - Pipeline continues with no context (may fail later, but doesn't crash)
    - "A dumb transcriber is better than a dead transcriber"

Multi-Worker Consideration:
    - Each Gunicorn/Uvicorn worker has its own cache
    - Cache is refreshed lazily on first access
    - Consider Redis for shared state in v2

Usage:
    # Initialize cache
    cache = PlaneEntityCache(client)
    await cache.refresh()

    # Safe access (for context enrichment)
    context = await cache.safe_get_context()  # Never throws

    # Get entity mappings for prompt injection
    projects = cache.get_project_mapping()  # {"mobile": "uuid-1", ...}
    users = cache.get_user_mapping()        # {"steve": "uuid-99", ...}

    # Resolve names to IDs
    project_id = cache.resolve_project("Mobile App")  # Returns "uuid-1"
    user_id = cache.resolve_user("Steve")              # Returns "uuid-99"
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from knomly.integrations.plane import PlaneClient

logger = logging.getLogger(__name__)

# Default timeout for cache refresh (context is auxiliary, not critical)
DEFAULT_REFRESH_TIMEOUT = 5.0  # seconds


@dataclass
class CachedProject:
    """Cached project information."""

    id: str
    name: str
    identifier: str  # Short key like "MOBILE"
    description: str = ""


@dataclass
class CachedUser:
    """Cached user information."""

    id: str
    display_name: str
    email: str = ""
    username: str = ""


@dataclass
class CachedState:
    """Cached workflow state information."""

    id: str
    name: str
    group: str  # "backlog", "unstarted", "started", "completed", "cancelled"
    project_id: str


@dataclass
class PlaneEntityCache:
    """
    In-memory cache of Plane entities for context-aware pipelines.

    This enables the system to resolve "Mobile App" → "uuid-1" without
    implementing full agent loops. The cache is refreshed periodically
    or on-demand.

    Thread-safety: Uses asyncio.Lock for concurrent refresh protection.
    TTL: Default 5 minutes, configurable.
    Timeout: Refresh has strict timeout to avoid blocking pipeline.

    Failure Modes:
        - API timeout: Returns stale/empty cache, logs warning
        - API error: Returns stale/empty cache, logs error
        - Network error: Returns stale/empty cache, logs error
        Pipeline continues in all cases (context is auxiliary).
    """

    client: PlaneClient
    ttl_seconds: float = 300.0  # 5 minutes default
    refresh_timeout: float = DEFAULT_REFRESH_TIMEOUT

    # Internal state
    _projects: dict[str, CachedProject] = field(default_factory=dict)
    _users: dict[str, CachedUser] = field(default_factory=dict)
    _states: dict[str, list[CachedState]] = field(default_factory=dict)  # project_id -> states
    _last_refresh: float = 0.0
    _last_error: str | None = None
    _refresh_lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    @property
    def is_stale(self) -> bool:
        """Check if cache needs refresh."""
        return time.time() - self._last_refresh > self.ttl_seconds

    @property
    def is_empty(self) -> bool:
        """Check if cache has any data."""
        return len(self._projects) == 0

    @property
    def is_healthy(self) -> bool:
        """Check if cache is healthy (has data and no recent errors)."""
        return not self.is_empty and self._last_error is None

    async def refresh(self, force: bool = False) -> bool:
        """
        Refresh cache from Plane API with timeout protection.

        Args:
            force: Refresh even if cache is not stale

        Returns:
            True if refresh succeeded, False otherwise
        """
        if not force and not self.is_stale and not self.is_empty:
            logger.debug("[plane_cache] Cache is fresh, skipping refresh")
            return True

        async with self._refresh_lock:
            # Double-check after acquiring lock
            if not force and not self.is_stale and not self.is_empty:
                return True

            logger.info("[plane_cache] Refreshing entity cache from Plane...")
            start = time.time()

            try:
                # Apply timeout to prevent blocking pipeline
                await asyncio.wait_for(
                    self._do_refresh(),
                    timeout=self.refresh_timeout,
                )

                self._last_refresh = time.time()
                self._last_error = None
                duration = time.time() - start

                logger.info(
                    f"[plane_cache] Refreshed: {len(self._projects)} projects, "
                    f"{sum(len(s) for s in self._states.values())} states "
                    f"in {duration:.2f}s"
                )
                return True

            except TimeoutError:
                self._last_error = f"Timeout after {self.refresh_timeout}s"
                logger.warning(
                    f"[plane_cache] Refresh timed out after {self.refresh_timeout}s. "
                    f"Continuing with {'stale' if not self.is_empty else 'empty'} cache."
                )
                return False

            except Exception as e:
                self._last_error = str(e)
                logger.error(
                    f"[plane_cache] Failed to refresh cache: {e}. "
                    f"Continuing with {'stale' if not self.is_empty else 'empty'} cache."
                )
                return False

    async def _do_refresh(self) -> None:
        """Internal refresh logic (without timeout wrapper)."""
        # Fetch projects
        await self._refresh_projects()

        # Fetch states for each project
        await self._refresh_states()

        # Note: User fetching requires admin permissions
        # For now, we skip it and rely on project memberships

    async def safe_get_context(self, auto_refresh: bool = True) -> dict:
        """
        Safely get context for Frame metadata. NEVER throws.

        This is the primary method for ContextEnrichmentProcessor.
        It handles all errors gracefully and returns empty context on failure.

        Args:
            auto_refresh: Whether to attempt refresh if stale

        Returns:
            Dict suitable for Frame metadata, always valid (may be empty)
        """
        try:
            # Attempt refresh if needed
            if auto_refresh and (self.is_stale or self.is_empty):
                await self.refresh()

            # Return context (may be empty if refresh failed)
            context = self.to_frame_metadata()

            # Add health status to context for observability
            context["plane_context"]["cache_healthy"] = self.is_healthy
            context["plane_context"]["cache_error"] = self._last_error

            return context

        except Exception as e:
            # This should never happen, but if it does, return empty context
            logger.error(f"[plane_cache] Unexpected error in safe_get_context: {e}")
            return self._empty_context()

    def _empty_context(self) -> dict:
        """Return empty context structure for graceful degradation."""
        return {
            "plane_context": {
                "projects": [],
                "users": [],
                "project_mapping": {},
                "user_mapping": {},
                "cache_timestamp": 0,
                "cache_healthy": False,
                "cache_error": self._last_error or "Cache unavailable",
            }
        }

    async def _refresh_projects(self) -> None:
        """Fetch and cache projects."""
        projects = await self.client.list_projects()

        self._projects = {
            p.id: CachedProject(
                id=p.id,
                name=p.name,
                identifier=p.identifier or "",
                description=p.description or "",
            )
            for p in projects
        }

    async def _refresh_states(self) -> None:
        """Fetch and cache workflow states for all projects."""
        self._states = {}

        # Fetch states for each project (in parallel for efficiency)
        tasks = [self._fetch_project_states(project_id) for project_id in self._projects]

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _fetch_project_states(self, project_id: str) -> None:
        """Fetch states for a single project."""
        try:
            # Note: This requires the PlaneClient to have a list_states method
            # For now, we'll leave states empty and add when API method exists
            self._states[project_id] = []
        except Exception as e:
            logger.warning(f"[plane_cache] Failed to fetch states for {project_id}: {e}")
            self._states[project_id] = []

    # =========================================================================
    # Query Methods (for ContextEnrichmentProcessor)
    # =========================================================================

    def get_project_mapping(self) -> dict[str, str]:
        """
        Get mapping of project names/identifiers to IDs.

        Returns:
            Dict mapping lowercase names and identifiers to project IDs
            Example: {"mobile": "uuid-1", "mobile app": "uuid-1", "MOB": "uuid-1"}
        """
        mapping: dict[str, str] = {}

        for project in self._projects.values():
            # Map by name (lowercase)
            mapping[project.name.lower()] = project.id

            # Map by identifier (lowercase)
            if project.identifier:
                mapping[project.identifier.lower()] = project.id

        return mapping

    def get_project_list_for_prompt(self) -> list[dict[str, str]]:
        """
        Get project list formatted for LLM prompt injection.

        Returns:
            List of dicts with name, identifier, and id for each project
        """
        return [
            {
                "name": p.name,
                "identifier": p.identifier,
                "id": p.id,
            }
            for p in self._projects.values()
        ]

    def get_user_mapping(self) -> dict[str, str]:
        """
        Get mapping of user names/emails to IDs.

        Returns:
            Dict mapping lowercase names, emails, usernames to user IDs
        """
        mapping: dict[str, str] = {}

        for user in self._users.values():
            # Map by display name (lowercase)
            mapping[user.display_name.lower()] = user.id

            # Map by email (lowercase)
            if user.email:
                mapping[user.email.lower()] = user.id

            # Map by username (lowercase)
            if user.username:
                mapping[user.username.lower()] = user.id

        return mapping

    def get_user_list_for_prompt(self) -> list[dict[str, str]]:
        """
        Get user list formatted for LLM prompt injection.

        Returns:
            List of dicts with name, email, and id for each user
        """
        return [
            {
                "name": u.display_name,
                "email": u.email,
                "id": u.id,
            }
            for u in self._users.values()
        ]

    # =========================================================================
    # Resolution Methods
    # =========================================================================

    def resolve_project(self, name_or_id: str) -> str | None:
        """
        Resolve a project name, identifier, or ID to a project ID.

        Args:
            name_or_id: Project name, identifier, or UUID

        Returns:
            Project ID if found, None otherwise
        """
        if not name_or_id:
            return None

        # Check if it's already a valid project ID
        if name_or_id in self._projects:
            return name_or_id

        # Look up by name/identifier
        mapping = self.get_project_mapping()
        return mapping.get(name_or_id.lower())

    def resolve_user(self, name_or_id: str) -> str | None:
        """
        Resolve a user name, email, or ID to a user ID.

        Args:
            name_or_id: User name, email, or UUID

        Returns:
            User ID if found, None otherwise
        """
        if not name_or_id:
            return None

        # Check if it's already a valid user ID
        if name_or_id in self._users:
            return name_or_id

        # Look up by name/email
        mapping = self.get_user_mapping()
        return mapping.get(name_or_id.lower())

    def get_best_match_project(self, query: str) -> tuple[str | None, float]:
        """
        Find the best matching project for a fuzzy query.

        Args:
            query: User's description of the project

        Returns:
            Tuple of (project_id, confidence_score)
            Confidence: 1.0 = exact match, 0.0 = no match
        """
        if not query:
            return None, 0.0

        query_lower = query.lower()

        # Exact match
        mapping = self.get_project_mapping()
        if query_lower in mapping:
            return mapping[query_lower], 1.0

        # Partial match (query contained in name)
        for project in self._projects.values():
            if query_lower in project.name.lower():
                return project.id, 0.8
            if project.identifier and query_lower in project.identifier.lower():
                return project.id, 0.8

        # Name contains query
        for project in self._projects.values():
            if project.name.lower() in query_lower:
                return project.id, 0.6

        return None, 0.0

    # =========================================================================
    # Serialization (for Frame metadata)
    # =========================================================================

    def to_frame_metadata(self) -> dict:
        """
        Serialize cache to a dict suitable for Frame metadata.

        This follows ADR-004: context data flows through Frames.

        Returns:
            Dict with projects and users for prompt injection
        """
        return {
            "plane_context": {
                "projects": self.get_project_list_for_prompt(),
                "users": self.get_user_list_for_prompt(),
                "project_mapping": self.get_project_mapping(),
                "user_mapping": self.get_user_mapping(),
                "cache_timestamp": self._last_refresh,
            }
        }
