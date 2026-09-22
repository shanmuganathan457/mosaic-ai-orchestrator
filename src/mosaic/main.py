"""MOSAIC FastAPI Application Entrypoint."""

import logging
from typing import Dict

from fastapi import FastAPI

from mosaic.api import router as support_router
from mosaic.config import settings

from fastapi.middleware.cors import CORSMiddleware

# Configure structured logging
logging.basicConfig(
    level=settings.LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mosaic")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Multi-Intent Orchestration & State-Aware Intelligent Coordination API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(support_router)


@app.get("/health", tags=["Health"])
async def health_check() -> Dict[str, str]:
    """Health check endpoint to verify backend operational readiness."""
    logger.debug("Health check ping received.")
    return {
        "status": "healthy",
        "project": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("mosaic.main:app", host="0.0.0.0", port=8000, reload=True)
