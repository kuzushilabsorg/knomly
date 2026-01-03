"""
Integration Frame Utilities.

IMPORTANT: Platform-specific frames are DEPRECATED.

The correct pattern for integrations is:

1. Generic domain frames: pipeline/frames/task.py
   - TaskFrame, TaskQueryFrame, TaskResultFrame
   - Platform-agnostic, works with any integration

2. Platform API schemas: integrations/<platform>/schemas.py
   - WorkItem, LinearIssue, JiraTicket, etc.
   - Only used internally by the platform's Processor

3. Platform Processors: pipeline/processors/integrations/<platform>.py
   - Accept generic frames (TaskFrame)
   - Return generic frames (TaskResultFrame)
   - Map between generic and platform-specific internally

Example:
    # Generic frame - no platform coupling
    task = TaskFrame(name="Fix bug", priority=TaskPriority.HIGH)

    # Processor handles platform-specific translation
    result = await PlaneProcessor(client).process(task, ctx)

    # Result is generic - can be consumed by any downstream processor
    assert result.platform == "plane"  # Only metadata reveals platform

This design allows swapping PlaneProcessor for LinearProcessor
without changing any upstream or downstream code.

See ADR-003-integration-pattern.md for full documentation.
"""

# No platform-specific frames exported.
# Use generic frames from pipeline/frames/task.py instead.

__all__: list[str] = []
