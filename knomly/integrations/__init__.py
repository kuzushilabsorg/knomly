"""
Knomly Integrations Layer.

This module provides clients for external SaaS integrations.
Each integration follows a consistent pattern:

1. Client: Handles authentication and API communication
2. Schemas: Pydantic models for request/response validation
3. Frames: Integration-specific frame types (in pipeline/frames/integrations/)
4. Processors: Integration processors (in pipeline/processors/integrations/)

Integration Patterns:
- Native Client: Full control, type-safe (used for core integrations)
- SDK Wrapper: Delegates to existing SDK (leverage existing work)
- Protocol: MCP/OpenAPI for universal reach (v2.5)

Directory Structure:
    integrations/
    ├── base.py           # Base classes and protocols
    ├── plane/            # Plane project management
    │   ├── client.py     # PlaneClient
    │   └── schemas.py    # Pydantic models
    └── twenty/           # Twenty CRM (future)
        └── client.py

Usage:
    from knomly.integrations.plane import PlaneClient

    client = PlaneClient(
        api_key="plane_api_xxx",
        workspace_slug="my-workspace",
    )
    work_item = await client.create_work_item(
        project_id="project-123",
        name="Fix login bug",
        priority="high",
    )
"""

from knomly.integrations.base import (
    AuthenticationError,
    IntegrationClient,
    IntegrationConfig,
    IntegrationError,
    RateLimitError,
)

__all__ = [
    "AuthenticationError",
    "IntegrationClient",
    "IntegrationConfig",
    "IntegrationError",
    "RateLimitError",
]
