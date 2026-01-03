"""
Knomly Runtime Layer.

This layer connects the v3 Adapter schemas to the v1 Pipeline execution.
It's the "Ignition System" that bridges:
- Database/File configuration (PipelinePacket)
- Service instantiation (ServiceFactory)
- Pipeline building (RuntimeBuilder)
- Request handling (PipelineResolver)

Design Principle:
    "Configuration flows down, Execution flows up."

    1. Admin stores config in database
    2. Resolver loads config at request time
    3. Factory builds services from config
    4. Pipeline executes with built services
    5. Response flows back to user

Components:
    - PipelineResolver: Loads config for a user/session
    - RuntimeBuilder: Builds live pipeline from packet
    - DefinitionLoader implementations (File, Mongo, API)

Usage:
    # At application startup
    resolver = PipelineResolver(
        loader=FileDefinitionLoader("config/"),
        factory=GenericServiceFactory(registry),
    )

    # At request time (in webhook handler)
    packet = await resolver.resolve_for_user(user_id)
    pipeline = await resolver.build_pipeline(packet, secrets)
    result = await pipeline.execute(initial_frame)
"""

from .builder import RuntimeBuilder
from .loaders import (
    FileDefinitionLoader,
    MemoryDefinitionLoader,
)
from .resolver import PipelineResolver

__all__ = [
    "FileDefinitionLoader",
    "MemoryDefinitionLoader",
    "PipelineResolver",
    "RuntimeBuilder",
]
