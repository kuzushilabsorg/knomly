"""
Routing Primitives for Knomly Pipeline.

Provides control flow mechanisms for branching, filtering, and parallel execution.
These primitives compose with standard Processors to build complex pipelines.

Design Philosophy:
- Routing is just another Processor (same interface)
- Explicit branching over implicit routing
- Composable primitives that combine for complex flows
- Full observability of routing decisions

See ADR-001 for design decisions.
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Protocol,
    Sequence,
    runtime_checkable,
)
from uuid import UUID

if TYPE_CHECKING:
    from .context import PipelineContext
    from .frames import Frame

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class RoutingError(Exception):
    """Raised when routing cannot determine a branch."""

    def __init__(self, router_name: str, message: str, frame_id: UUID | None = None):
        self.router_name = router_name
        self.frame_id = frame_id
        super().__init__(f"[{router_name}] {message}")


class PipelineExit(Exception):
    """
    Raised to exit the pipeline early with a specific frame.

    Used by Guard and other control-flow primitives to short-circuit
    execution when certain conditions are met.

    Example:
        if not is_authorized(frame):
            raise PipelineExit(UnauthorizedFrame(...))
    """

    def __init__(self, exit_frame: "Frame"):
        self.exit_frame = exit_frame
        super().__init__(f"Pipeline exit with {exit_frame.frame_type}")


# =============================================================================
# Routing Decision Tracking
# =============================================================================


@dataclass(frozen=True)
class RoutingDecision:
    """
    Records a routing decision for observability.

    Stored in PipelineContext.routing_decisions for debugging
    and audit trail.
    """

    timestamp: datetime
    router_name: str
    frame_id: UUID
    frame_type: str
    selected_branch: str
    evaluated_condition: str | None = None
    all_branches: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "router_name": self.router_name,
            "frame_id": str(self.frame_id),
            "frame_type": self.frame_type,
            "selected_branch": self.selected_branch,
            "evaluated_condition": self.evaluated_condition,
            "all_branches": list(self.all_branches),
            "metadata": self.metadata,
        }


def record_decision(
    ctx: "PipelineContext",
    router_name: str,
    frame: "Frame",
    selected_branch: str,
    *,
    condition: str | None = None,
    all_branches: Sequence[str] = (),
    metadata: dict[str, Any] | None = None,
) -> None:
    """Helper to record routing decisions in context."""
    decision = RoutingDecision(
        timestamp=datetime.now(timezone.utc),
        router_name=router_name,
        frame_id=frame.id,
        frame_type=frame.frame_type,
        selected_branch=selected_branch,
        evaluated_condition=condition,
        all_branches=tuple(all_branches),
        metadata=metadata or {},
    )

    # Add to context if it has routing_decisions attribute
    if hasattr(ctx, "routing_decisions"):
        ctx.routing_decisions.append(decision)

    logger.debug(
        f"Routing decision: {router_name} -> {selected_branch} "
        f"(frame={frame.frame_type}, id={str(frame.id)[:8]}...)"
    )


# =============================================================================
# Executable Protocol
# =============================================================================


@runtime_checkable
class Executable(Protocol):
    """
    Protocol for anything that can execute on a frame.

    Both Processor and Pipeline implement this protocol,
    enabling composition of routing branches.
    """

    async def execute(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        """Execute on a frame and return result."""
        ...


# =============================================================================
# Condition Types
# =============================================================================

# Sync condition: (Frame, Context) -> bool
SyncCondition = Callable[["Frame", "PipelineContext"], bool]

# Async condition: (Frame, Context) -> Awaitable[bool]
AsyncCondition = Callable[["Frame", "PipelineContext"], Awaitable[bool]]

# Either sync or async condition
Condition = SyncCondition | AsyncCondition

# Key extractor: (Frame, Context) -> str
KeyExtractor = Callable[["Frame", "PipelineContext"], str]


async def evaluate_condition(
    condition: Condition,
    frame: "Frame",
    ctx: "PipelineContext",
) -> bool:
    """Evaluate a condition, handling both sync and async."""
    result = condition(frame, ctx)
    if asyncio.iscoroutine(result):
        return await result
    return result  # type: ignore[return-value]


# =============================================================================
# Base Router
# =============================================================================


class Router(ABC):
    """
    Base class for routing processors.

    Routers are Processors that direct frames to different branches
    based on conditions, frame types, or other criteria.
    """

    @property
    def name(self) -> str:
        """Router name for logging and metrics."""
        return self.__class__.__name__

    async def _execute_branch(
        self,
        branch: Executable,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        """Execute a branch, handling both Processor and Pipeline."""
        # Check for Pipeline (has execute method that takes initial_frame)
        if hasattr(branch, "execute") and hasattr(branch, "processors"):
            # It's a Pipeline
            from .executor import Pipeline

            if isinstance(branch, Pipeline):
                result = await branch.execute(initial_frame=frame, ctx=ctx)
                return result.output_frames if result.output_frames else None

        # Check for Processor (has process method)
        if hasattr(branch, "process"):
            return await branch.process(frame, ctx)

        # Generic Executable
        return await branch.execute(frame, ctx)


# =============================================================================
# Conditional Router (Binary If/Else)
# =============================================================================


@dataclass
class Conditional(Router):
    """
    Binary conditional routing (if/else).

    Routes frames to one of two branches based on a condition.
    Both branches must be provided.

    Example:
        router = Conditional(
            condition=lambda frame, ctx: frame.confidence > 0.8,
            if_true=HighConfidenceProcessor(),
            if_false=LowConfidenceProcessor(),
        )

    Args:
        condition: Function that returns True or False
        if_true: Branch to execute when condition is True
        if_false: Branch to execute when condition is False
        condition_name: Optional name for logging/debugging
    """

    condition: Condition
    if_true: Executable
    if_false: Executable
    condition_name: str = "condition"

    @property
    def name(self) -> str:
        return f"Conditional({self.condition_name})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        result = await evaluate_condition(self.condition, frame, ctx)

        # Record decision
        selected = "if_true" if result else "if_false"
        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch=selected,
            condition=f"{self.condition_name} = {result}",
            all_branches=("if_true", "if_false"),
        )

        branch = self.if_true if result else self.if_false
        return await self._execute_branch(branch, frame, ctx)


# =============================================================================
# Switch Router (Multi-way Branching)
# =============================================================================


@dataclass
class Switch(Router):
    """
    Multi-way branching based on a key.

    Extracts a key from the frame and routes to the corresponding branch.
    Optionally falls back to a default branch.

    Example:
        router = Switch(
            key=lambda frame, ctx: frame.detected_language,
            cases={
                "en": EnglishProcessor(),
                "es": SpanishProcessor(),
                "hi": HindiProcessor(),
            },
            default=FallbackProcessor(),
        )

    Args:
        key: Function that extracts a string key from the frame
        cases: Mapping from key values to branches
        default: Optional fallback when key doesn't match any case
        key_name: Optional name for logging/debugging
    """

    key: KeyExtractor
    cases: dict[str, Executable]
    default: Executable | None = None
    key_name: str = "key"

    @property
    def name(self) -> str:
        return f"Switch({self.key_name})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        key_value = self.key(frame, ctx)

        # Record decision
        selected = key_value if key_value in self.cases else "default"
        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch=selected,
            condition=f"{self.key_name} = '{key_value}'",
            all_branches=tuple(self.cases.keys()) + (("default",) if self.default else ()),
        )

        # Find matching branch
        branch = self.cases.get(key_value)

        if branch is None:
            if self.default is not None:
                return await self._execute_branch(self.default, frame, ctx)
            else:
                raise RoutingError(
                    router_name=self.name,
                    message=f"No case for key '{key_value}' and no default provided",
                    frame_id=frame.id,
                )

        return await self._execute_branch(branch, frame, ctx)


# =============================================================================
# TypeRouter (Route by Frame Type)
# =============================================================================


@dataclass
class TypeRouter(Router):
    """
    Routes frames based on their type.

    Useful for pipelines that handle multiple frame types
    and need different processing paths for each.

    Example:
        router = TypeRouter(
            routes={
                AudioInputFrame: AudioProcessor(),
                TextInputFrame: TextProcessor(),
                ErrorFrame: ErrorHandler(),
            },
            default=PassthroughProcessor(),
        )

    Args:
        routes: Mapping from frame types to branches
        default: Optional fallback for unmatched types
    """

    routes: dict[type, Executable]
    default: Executable | None = None

    @property
    def name(self) -> str:
        type_names = [t.__name__ for t in self.routes.keys()]
        return f"TypeRouter({', '.join(type_names)})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        frame_type = type(frame)

        # Find matching branch (check exact type first, then bases)
        branch: Executable | None = None
        selected_type: str = "default"

        # Exact match
        if frame_type in self.routes:
            branch = self.routes[frame_type]
            selected_type = frame_type.__name__
        else:
            # Check inheritance
            for route_type, route_branch in self.routes.items():
                if isinstance(frame, route_type):
                    branch = route_branch
                    selected_type = route_type.__name__
                    break

        # Record decision
        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch=selected_type,
            condition=f"type = {frame_type.__name__}",
            all_branches=tuple(t.__name__ for t in self.routes.keys())
            + (("default",) if self.default else ()),
        )

        if branch is not None:
            return await self._execute_branch(branch, frame, ctx)
        elif self.default is not None:
            return await self._execute_branch(self.default, frame, ctx)
        else:
            raise RoutingError(
                router_name=self.name,
                message=f"No route for type {frame_type.__name__} and no default",
                frame_id=frame.id,
            )


# =============================================================================
# Filter (Gate/Predicate)
# =============================================================================


@dataclass
class Filter(Router):
    """
    Gate that allows or blocks frames based on a condition.

    When the condition is True, the frame passes through unchanged.
    When False, returns on_reject (or None to drop the frame).

    Example:
        # Drop low-confidence transcriptions
        filter = Filter(
            condition=lambda f, c: f.confidence > 0.5,
            on_reject=None,  # Drop the frame
        )

        # Return error frame on rejection
        filter = Filter(
            condition=lambda f, c: is_authorized(f, c),
            on_reject=ErrorFrame(error_type="unauthorized"),
        )

    Args:
        condition: Function that returns True to pass, False to reject
        on_reject: Frame to return on rejection (None to drop)
        condition_name: Optional name for logging/debugging
    """

    condition: Condition
    on_reject: "Frame | None" = None
    condition_name: str = "filter"

    @property
    def name(self) -> str:
        return f"Filter({self.condition_name})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        result = await evaluate_condition(self.condition, frame, ctx)

        selected = "pass" if result else "reject"
        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch=selected,
            condition=f"{self.condition_name} = {result}",
            all_branches=("pass", "reject"),
        )

        if result:
            return frame  # Pass through unchanged
        else:
            return self.on_reject  # Return rejection frame or None


# =============================================================================
# Guard (Early Exit)
# =============================================================================


@dataclass
class Guard(Router):
    """
    Guard that exits the pipeline early when condition is True.

    Unlike Filter which continues the pipeline, Guard raises
    PipelineExit to completely stop execution and return
    the exit_frame as the final result.

    Example:
        guard = Guard(
            condition=lambda f, c: f.is_spam,
            exit_frame=SpamDetectedFrame(...),
        )

    Args:
        condition: Function that returns True to exit pipeline
        exit_frame: Frame to return when exiting
        condition_name: Optional name for logging/debugging
    """

    condition: Condition
    exit_frame: "Frame"
    condition_name: str = "guard"

    @property
    def name(self) -> str:
        return f"Guard({self.condition_name})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        result = await evaluate_condition(self.condition, frame, ctx)

        if result:
            record_decision(
                ctx=ctx,
                router_name=self.name,
                frame=frame,
                selected_branch="exit",
                condition=f"{self.condition_name} = True",
                all_branches=("continue", "exit"),
            )
            raise PipelineExit(self.exit_frame)

        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch="continue",
            condition=f"{self.condition_name} = False",
            all_branches=("continue", "exit"),
        )
        return frame  # Continue pipeline


# =============================================================================
# FanOut (Parallel Execution)
# =============================================================================


class FanOutStrategy(Enum):
    """Strategy for handling multiple parallel branch results."""

    ALL = "all"
    """Wait for all branches, fail if any fails."""

    ALL_SETTLED = "all_settled"
    """Wait for all branches, collect both successes and failures."""

    FIRST_SUCCESS = "first_success"
    """Return result from first successful branch."""

    RACE = "race"
    """Return result from first completed branch (success or failure)."""


@dataclass
class FanOut(Router):
    """
    Execute multiple branches in parallel.

    Sends the same frame to all branches concurrently and combines
    results based on the configured strategy.

    Example:
        # Send to multiple notification channels
        fanout = FanOut(
            branches=[
                SlackNotifier(),
                EmailNotifier(),
                PushNotifier(),
            ],
            strategy=FanOutStrategy.ALL_SETTLED,
        )

    Args:
        branches: List of branches to execute in parallel
        strategy: How to combine/select results
        timeout: Optional timeout for parallel execution (seconds)
    """

    branches: Sequence[Executable]
    strategy: FanOutStrategy = FanOutStrategy.ALL_SETTLED
    timeout: float | None = None

    @property
    def name(self) -> str:
        return f"FanOut({self.strategy.value}, n={len(self.branches)})"

    async def process(
        self,
        frame: "Frame",
        ctx: "PipelineContext",
    ) -> "Frame | Sequence[Frame] | None":
        if not self.branches:
            return frame  # No branches, passthrough

        branch_names = [
            getattr(b, "name", b.__class__.__name__) for b in self.branches
        ]

        record_decision(
            ctx=ctx,
            router_name=self.name,
            frame=frame,
            selected_branch=f"parallel({len(self.branches)})",
            condition=f"strategy={self.strategy.value}",
            all_branches=tuple(branch_names),
            metadata={"strategy": self.strategy.value, "timeout": self.timeout},
        )

        # Create tasks for all branches
        tasks = [
            asyncio.create_task(
                self._execute_branch(branch, frame, ctx),
                name=f"fanout_{i}_{branch_names[i]}",
            )
            for i, branch in enumerate(self.branches)
        ]

        try:
            return await self._gather_results(tasks, branch_names)
        except asyncio.TimeoutError:
            # Cancel remaining tasks on timeout
            for task in tasks:
                if not task.done():
                    task.cancel()
            raise RoutingError(
                router_name=self.name,
                message=f"FanOut timed out after {self.timeout}s",
                frame_id=frame.id,
            )

    async def _gather_results(
        self,
        tasks: list[asyncio.Task[Any]],
        branch_names: list[str],
    ) -> "Frame | Sequence[Frame] | None":
        """Gather results based on strategy."""
        if self.strategy == FanOutStrategy.ALL:
            return await self._strategy_all(tasks)
        elif self.strategy == FanOutStrategy.ALL_SETTLED:
            return await self._strategy_all_settled(tasks)
        elif self.strategy == FanOutStrategy.FIRST_SUCCESS:
            return await self._strategy_first_success(tasks, branch_names)
        elif self.strategy == FanOutStrategy.RACE:
            return await self._strategy_race(tasks)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    async def _strategy_all(
        self,
        tasks: list[asyncio.Task[Any]],
    ) -> "Sequence[Frame]":
        """Wait for all, fail if any fails."""
        if self.timeout:
            results = await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=self.timeout,
            )
        else:
            results = await asyncio.gather(*tasks)

        # Flatten results
        return self._flatten_results(results)

    async def _strategy_all_settled(
        self,
        tasks: list[asyncio.Task[Any]],
    ) -> "Sequence[Frame]":
        """Wait for all, collect successes and failures."""
        if self.timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            # Cancel pending on timeout
            for task in pending:
                task.cancel()
        else:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.ALL_COMPLETED)

        results = []
        for task in done:
            try:
                result = task.result()
                if result is not None:
                    if isinstance(result, (list, tuple)):
                        results.extend(result)
                    else:
                        results.append(result)
            except Exception as e:
                # Log error but continue with other results
                logger.warning(f"FanOut branch failed: {e}")

        return results

    async def _strategy_first_success(
        self,
        tasks: list[asyncio.Task[Any]],
        branch_names: list[str],
    ) -> "Frame | Sequence[Frame] | None":
        """Return first successful result."""
        pending = set(tasks)
        errors: list[Exception] = []

        while pending:
            if self.timeout:
                done, pending = await asyncio.wait(
                    pending,
                    timeout=self.timeout,
                    return_when=asyncio.FIRST_COMPLETED,
                )
            else:
                done, pending = await asyncio.wait(
                    pending,
                    return_when=asyncio.FIRST_COMPLETED,
                )

            for task in done:
                try:
                    result = task.result()
                    # Cancel remaining tasks
                    for p in pending:
                        p.cancel()
                    return result
                except Exception as e:
                    errors.append(e)

        # All failed
        raise RoutingError(
            router_name=self.name,
            message=f"All {len(tasks)} branches failed: {errors}",
            frame_id=None,
        )

    async def _strategy_race(
        self,
        tasks: list[asyncio.Task[Any]],
    ) -> "Frame | Sequence[Frame] | None":
        """Return first completed result (success or failure)."""
        if self.timeout:
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )
        else:
            done, pending = await asyncio.wait(
                tasks,
                return_when=asyncio.FIRST_COMPLETED,
            )

        # Cancel remaining
        for task in pending:
            task.cancel()

        if done:
            # Return first completed (may raise if it failed)
            return done.pop().result()

        return None

    def _flatten_results(
        self,
        results: list[Any],
    ) -> list["Frame"]:
        """Flatten nested results into list of frames."""
        flattened: list[Frame] = []
        for result in results:
            if result is None:
                continue
            elif isinstance(result, (list, tuple)):
                flattened.extend(result)
            else:
                flattened.append(result)
        return flattened


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Exceptions
    "RoutingError",
    "PipelineExit",
    # Decision tracking
    "RoutingDecision",
    "record_decision",
    # Protocol
    "Executable",
    # Condition types
    "Condition",
    "SyncCondition",
    "AsyncCondition",
    "KeyExtractor",
    "evaluate_condition",
    # Routers
    "Router",
    "Conditional",
    "Switch",
    "TypeRouter",
    "Filter",
    "Guard",
    "FanOut",
    "FanOutStrategy",
]
