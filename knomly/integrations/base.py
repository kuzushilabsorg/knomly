"""
Base classes for Knomly integrations.

This module defines the foundational abstractions for SaaS integrations,
ensuring consistent patterns across all integration clients.

Design Principles:
1. Async-first: All I/O operations are async
2. Type-safe: Pydantic models for all data
3. Observable: Logging and metrics hooks
4. Resilient: Built-in retry with exponential backoff
5. Testable: Easy to mock and test

Retry Strategy:
    - Retryable errors: timeouts, network errors, 429, 5xx
    - Non-retryable: 4xx (except 429), auth errors
    - Backoff: exponential with jitter
"""

from __future__ import annotations

import asyncio
import logging
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import httpx

logger = logging.getLogger(__name__)


# =============================================================================
# Exceptions
# =============================================================================


class IntegrationError(Exception):
    """Base exception for integration errors."""

    def __init__(
        self,
        message: str,
        integration: str,
        *,
        status_code: int | None = None,
        response_body: str | None = None,
        retryable: bool = False,
    ):
        super().__init__(message)
        self.integration = integration
        self.status_code = status_code
        self.response_body = response_body
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [f"[{self.integration}] {self.args[0]}"]
        if self.status_code:
            parts.append(f"(status={self.status_code})")
        return " ".join(parts)


class AuthenticationError(IntegrationError):
    """Raised when authentication fails (401/403)."""

    def __init__(self, message: str, integration: str, **kwargs):
        super().__init__(message, integration, retryable=False, **kwargs)


class RateLimitError(IntegrationError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(
        self,
        message: str,
        integration: str,
        *,
        retry_after: float | None = None,
        **kwargs,
    ):
        super().__init__(message, integration, retryable=True, **kwargs)
        self.retry_after = retry_after


class NotFoundError(IntegrationError):
    """Raised when a resource is not found (404)."""

    def __init__(self, message: str, integration: str, **kwargs):
        super().__init__(message, integration, retryable=False, **kwargs)


class ValidationError(IntegrationError):
    """Raised when request validation fails (400/422)."""

    def __init__(
        self,
        message: str,
        integration: str,
        *,
        validation_errors: list[dict[str, Any]] | None = None,
        **kwargs,
    ):
        super().__init__(message, integration, retryable=False, **kwargs)
        self.validation_errors = validation_errors or []


# =============================================================================
# Configuration
# =============================================================================


@dataclass(frozen=True, slots=True)
class IntegrationConfig:
    """Configuration for an integration client."""

    # Authentication
    api_key: str | None = None
    api_secret: str | None = None
    access_token: str | None = None

    # Connection
    base_url: str = ""
    timeout: float = 30.0

    # Rate limiting
    max_retries: int = 3
    retry_delay: float = 1.0

    # Observability
    log_requests: bool = False
    log_responses: bool = False


# =============================================================================
# Response Types
# =============================================================================

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class IntegrationResponse(Generic[T]):
    """Standard response wrapper for integration calls."""

    success: bool
    data: T | None = None
    error: str | None = None
    status_code: int | None = None
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def ok(cls, data: T, status_code: int = 200, headers: dict[str, str] | None = None) -> "IntegrationResponse[T]":
        """Create a successful response."""
        return cls(success=True, data=data, status_code=status_code, headers=headers or {})

    @classmethod
    def fail(cls, error: str, status_code: int | None = None) -> "IntegrationResponse[T]":
        """Create a failed response."""
        return cls(success=False, error=error, status_code=status_code)


@dataclass(frozen=True, slots=True)
class PaginatedResponse(Generic[T]):
    """Paginated response for list operations."""

    items: list[T]
    total_count: int
    page: int = 1
    per_page: int = 50
    has_next: bool = False
    next_cursor: str | None = None

    @property
    def total_pages(self) -> int:
        """Calculate total pages."""
        if self.per_page == 0:
            return 0
        return (self.total_count + self.per_page - 1) // self.per_page


# =============================================================================
# Base Client
# =============================================================================


class IntegrationClient(ABC):
    """
    Abstract base class for integration clients.

    Provides common functionality:
    - HTTP client management
    - Authentication header injection
    - Error handling and mapping
    - Request/response logging
    - Rate limit handling

    Subclasses must implement:
    - name: Integration identifier
    - _get_auth_headers(): Return authentication headers
    """

    def __init__(self, config: IntegrationConfig):
        """
        Initialize the integration client.

        Args:
            config: Integration configuration
        """
        self.config = config
        self._client: httpx.AsyncClient | None = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name for this integration."""
        ...

    @abstractmethod
    def _get_auth_headers(self) -> dict[str, str]:
        """Return authentication headers for requests."""
        ...

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    **self._get_auth_headers(),
                },
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Make an HTTP request with retry and exponential backoff.

        Args:
            method: HTTP method (GET, POST, PATCH, DELETE)
            path: URL path (will be appended to base_url)
            params: Query parameters
            json: JSON body
            headers: Additional headers

        Returns:
            httpx.Response

        Raises:
            IntegrationError: On any non-retryable error or after max retries
        """
        last_error: IntegrationError | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._do_request(
                    method, path, params=params, json=json, headers=headers
                )
            except IntegrationError as e:
                last_error = e

                if not e.retryable:
                    # Non-retryable errors fail immediately
                    raise

                if attempt >= self.config.max_retries:
                    # Max retries reached
                    logger.warning(
                        f"[{self.name}] Max retries ({self.config.max_retries}) "
                        f"reached for {method} {path}"
                    )
                    raise

                # Calculate backoff with jitter
                backoff = self._calculate_backoff(attempt, e)
                logger.info(
                    f"[{self.name}] Retry {attempt + 1}/{self.config.max_retries} "
                    f"for {method} {path} after {backoff:.2f}s"
                )
                await asyncio.sleep(backoff)

        # Should never reach here, but satisfy type checker
        if last_error:
            raise last_error
        raise IntegrationError("Unknown error", self.name)

    def _calculate_backoff(
        self,
        attempt: int,
        error: IntegrationError,
    ) -> float:
        """
        Calculate backoff delay with exponential growth and jitter.

        Args:
            attempt: Current attempt number (0-indexed)
            error: The error that triggered the retry

        Returns:
            Delay in seconds
        """
        # Check for Retry-After header (from 429 responses)
        if isinstance(error, RateLimitError) and error.retry_after:
            return error.retry_after

        # Exponential backoff: delay * (2 ^ attempt)
        base_delay = self.config.retry_delay * (2 ** attempt)

        # Add jitter (±25%) to prevent thundering herd
        jitter = base_delay * 0.25 * (2 * random.random() - 1)

        # Cap at 60 seconds
        return min(base_delay + jitter, 60.0)

    async def _do_request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> httpx.Response:
        """
        Execute a single HTTP request.

        This is the internal method that _request wraps with retry logic.
        """
        client = await self._get_client()

        if self.config.log_requests:
            logger.debug(f"[{self.name}] {method} {path} params={params} body={json}")

        try:
            response = await client.request(
                method=method,
                url=path,
                params=params,
                json=json,
                headers=headers,
            )

            if self.config.log_responses:
                logger.debug(
                    f"[{self.name}] Response: status={response.status_code} "
                    f"body={response.text[:500] if response.text else 'empty'}"
                )

            # Handle errors
            self._check_response(response)

            return response

        except httpx.TimeoutException as e:
            raise IntegrationError(
                f"Request timeout: {e}",
                self.name,
                retryable=True,
            ) from e
        except httpx.NetworkError as e:
            raise IntegrationError(
                f"Network error: {e}",
                self.name,
                retryable=True,
            ) from e

    def _check_response(self, response: httpx.Response) -> None:
        """
        Check response for errors and raise appropriate exceptions.

        Args:
            response: HTTP response to check

        Raises:
            AuthenticationError: For 401/403
            RateLimitError: For 429
            NotFoundError: For 404
            ValidationError: For 400/422
            IntegrationError: For other errors
        """
        if response.is_success:
            return

        status = response.status_code
        body = response.text

        if status == 401 or status == 403:
            raise AuthenticationError(
                f"Authentication failed: {body}",
                self.name,
                status_code=status,
                response_body=body,
            )

        if status == 429:
            retry_after = response.headers.get("Retry-After")
            raise RateLimitError(
                "Rate limit exceeded",
                self.name,
                status_code=status,
                response_body=body,
                retry_after=float(retry_after) if retry_after else None,
            )

        if status == 404:
            raise NotFoundError(
                f"Resource not found: {body}",
                self.name,
                status_code=status,
                response_body=body,
            )

        if status == 400 or status == 422:
            raise ValidationError(
                f"Validation error: {body}",
                self.name,
                status_code=status,
                response_body=body,
            )

        # Generic error
        raise IntegrationError(
            f"Request failed: {body}",
            self.name,
            status_code=status,
            response_body=body,
            retryable=status >= 500,
        )

    async def health_check(self) -> bool:
        """
        Check if the integration is healthy/reachable.

        Returns:
            True if healthy, False otherwise
        """
        try:
            # Subclasses should override with integration-specific check
            return True
        except Exception:
            return False

    async def __aenter__(self) -> "IntegrationClient":
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
