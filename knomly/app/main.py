"""
Knomly - Voice-First AI Operations Assistant

FastAPI application entry point.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from knomly.app.api.webhooks import twilio_router
from knomly.app.dependencies import get_settings, initialize_services, shutdown_services

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager.

    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting Knomly services...")
    try:
        await initialize_services()
        logger.info("Knomly services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}", exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Knomly services...")
    try:
        await shutdown_services()
        logger.info("Knomly services shut down successfully")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}", exc_info=True)


# Create FastAPI application
settings = get_settings()

app = FastAPI(
    title="Knomly",
    description="Voice-First AI Operations Assistant - Build modular pipelines for voice and messaging applications",
    version="0.1.0",
    lifespan=lifespan,
    debug=settings.debug,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(twilio_router, prefix="/api/v1")


@app.get("/", tags=["root"])
async def root() -> dict[str, str]:
    """Root endpoint with service info."""
    return {
        "service": "knomly",
        "version": "0.1.0",
        "status": "running",
    }


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, Any]:
    """
    Health check endpoint.

    Returns service health status including:
    - Provider availability
    - MongoDB connection status
    """
    try:
        from knomly.app.dependencies import get_config_service, get_providers

        providers = get_providers()
        config = get_config_service()

        return {
            "status": "healthy",
            "providers": providers.list_providers(),
            "database": "connected" if config._db is not None else "disconnected",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }


@app.get("/api/v1/prompts", tags=["config"])
async def list_prompts() -> dict[str, Any]:
    """List all configured prompts."""
    from knomly.app.dependencies import get_config_service

    config = get_config_service()
    prompts = await config.list_prompts()
    return {
        "prompts": [p.model_dump() for p in prompts],
    }


@app.get("/api/v1/users", tags=["config"])
async def list_users() -> dict[str, Any]:
    """List all configured users."""
    from knomly.app.dependencies import get_config_service

    config = get_config_service()
    users = await config.list_users()
    return {
        "users": [u.model_dump() for u in users],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "knomly.app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
