"""
Integration-specific Processors.

This module contains Processors for specific SaaS integrations.
Each processor accepts GENERIC domain frames (TaskFrame, ContactFrame, etc.)
and handles platform-specific mapping internally.

Design Principle:
    Frames describe WHAT (the data/intent), not WHERE (the platform).
    Processors handle WHERE (the platform) and HOW (the mapping).

This allows:
    - Swap PlaneProcessor for LinearProcessor without changing upstream code
    - Consistent interface across all integrations
    - Domain-driven design

Design Pattern:
    Integration processors are v1 primitives that:
    1. Accept GENERIC domain frames (TaskFrame, ContactFrame, etc.)
    2. Map to platform-specific API calls
    3. Return GENERIC result frames (TaskResultFrame, etc.)

    They can be:
    - Used directly in v1 pipelines
    - Wrapped by Skills in v2 agent layer
    - Called via MCP/OpenAPI in v2.5

Available integrations:
- plane: PlaneProcessor (task management)
- twenty: TwentyProcessor (CRM) - future
- linear: LinearProcessor (task management) - future

Usage:
    from knomly.pipeline.processors.integrations import PlaneProcessor
    from knomly.pipeline.frames.task import TaskFrame

    # Generic TaskFrame works with any task management processor
    task = TaskFrame(
        name="Fix login bug",
        priority="high",
        project="backend",
    )

    # PlaneProcessor maps to Plane API
    result = await PlaneProcessor(client).process(task, ctx)

    # Same TaskFrame could be processed by LinearProcessor, JiraProcessor, etc.
"""

from knomly.pipeline.processors.integrations.plane import (
    PlaneProcessor,
    TaskCreatorProcessor,
)

__all__ = [
    # Plane processor (accepts generic TaskFrame)
    "PlaneProcessor",
    # Generic task creator (produces TaskFrame from ExtractionFrame)
    "TaskCreatorProcessor",
]
