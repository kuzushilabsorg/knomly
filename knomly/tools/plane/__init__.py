"""
Plane Tools for v2 Agentic Layer.

These tools wrap Plane API operations for agent use:
- PlaneCreateTaskTool: Create tasks in Plane
- PlaneQueryTasksTool: Query/list tasks

Multi-Tenancy (v2.1 - ADR-007):
    Use PlaneToolFactory for per-request tool creation with
    user-specific API keys.

Design Principle (ADR-005):
    Tools wrap existing v1 logic (PlaneClient, PlaneEntityCache).
    Tools do NOT know they are called by an agent.

Usage (single-tenant):
    from knomly.tools.plane import PlaneCreateTaskTool

    tool = PlaneCreateTaskTool(client=plane_client, cache=entity_cache)
    result = await tool.execute({...})

Usage (multi-tenant):
    from knomly.tools.plane import PlaneToolFactory

    factory = PlaneToolFactory(workspace_slug="my-workspace")
    tools = factory.build_tools(context)  # context has user's API key
"""

from .create_task import PlaneCreateTaskTool
from .query_tasks import PlaneQueryTasksTool
from .factory import PlaneToolFactory, CachedPlaneToolFactory

__all__ = [
    "PlaneCreateTaskTool",
    "PlaneQueryTasksTool",
    "PlaneToolFactory",
    "CachedPlaneToolFactory",
]
